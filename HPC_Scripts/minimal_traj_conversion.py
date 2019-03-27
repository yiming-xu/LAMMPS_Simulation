# Modified by Yiming Xu from source below.
#
# lammps.py (2011/03/29)
# An ASE calculator for the LAMMPS classical MD code available from
#       http://lammps.sandia.gov/
# The environment variable LAMMPS_COMMAND must be defined to point to the
# LAMMPS binary.
#
# Copyright (C) 2009 - 2011 Joerg Meyer, joerg.meyer@ch.tum.de
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this file; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA or see <http://www.gnu.org/licenses/>.

import decimal as dec
import os
import shlex
import shutil
from io import StringIO
from multiprocessing import Event, Manager, Process, Queue, Value
from subprocess import PIPE, Popen

import numpy as np
import pandas as pd
from ase import Atoms
from ase.parallel import paropen

__all__ = ['LAMMPS']

# "End mark" used to indicate that the calculation is done
CALCULATION_END_MARK = '__end_of_ase_invoked_calculation__'
CUSTOM_LOG_FORMAT = 'id type x y z vx vy vz fx fy fz\n'


class LAMMPS:
    def __init__(self, label='lammps', tmp_dir=None, parameters={},
                 specorder=None, files=[], always_triclinic=False,
                 keep_alive=True, keep_tmp_files=False,
                 no_data_file=False):
        """The LAMMPS calculators object

        files: list
            Short explanation XXX
        parameters: dict
            List of LAMMPS input parameters that will be passed into the
            input file.
        always_triclinic: bool
            Force use of a triclinic cell in LAMMPS, even if the cell is
            a perfect parallelepiped.
        """

        self.label = label
        self.trajectory_out = None

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        self.prism = Prism(atoms.get_cell())

    def read_lammps_trj(self, lammps_trj=None, n_cpus=8):
        """Method which reads a LAMMPS dump file."""

        def df_producer(in_queue, out_queue, write_progress, write_event):
            while True:
                step_str = in_queue.get(timeout=1800)

                if step_str == 'DONE':
                    out_queue.put('DONE')
                    break
                else:
                    step_number = step_str[0]
                    atoms_str = step_str[-1]
                    atoms_df = pd.read_csv(StringIO(CUSTOM_LOG_FORMAT + atoms_str),
                                           delim_whitespace=True,
                                           index_col=0)

                    while step_number != write_progress.value:
                        write_event.wait(timeout=None)

                    write_event.clear()
                    atoms_df.sort_index(inplace=True)
                    out_queue.put([atoms_df, step_number])

                    with write_progress.get_lock():
                        write_progress.value += 1
                    write_event.set()

        def step_reader(f, in_queue, n_procs):
            """ This function reads 1 step of the trajectory file and sends it to
            in_queue to be further processed
            """
            for line in f:
                assert line.startswith('ITEM: TIMESTEP')
                _step_number = int(next(f).strip())
                assert next(f).startswith('ITEM: NUMBER OF ATOMS')
                n_atoms = int(next(f).strip())
                assert next(f).startswith('ITEM: BOX BOUNDS')
                box_str = [next(f).rstrip().split(' ') for x in range(3)]
                assert next(f).startswith('ITEM: ATOMS')
                atoms_str = ''.join([next(f) for x in range(n_atoms)])

                # Add these information to in_queue
                in_queue.put([_step_number, n_atoms, box_str, atoms_str])

            [in_queue.put('DONE') for _ in range(n_procs)]

        def step_writer(trajectory_out, out_queue, n_procs):
            """ This function gets 1 step from out_queue, calculates
            the needful and writes it to file.
            """
            write_counter = 0

            while write_counter < n_procs:
                msg = out_queue.get(timeout=1800) # time out in 30 minutes if nothing is written

                if isinstance(msg, str):
                    write_counter += 1
                else:
                    atoms_df = msg[0]
                    _step_number = msg[1]
                    # print(_step_number)
                cell_atoms = self.atoms.get_cell()
                rotation_lammps2ase = np.linalg.inv(self.prism.R)
                type_atoms = self.atoms.get_atomic_numbers()
                cols = atoms_df.columns

                positions_atoms = np.dot(atoms_df.to_numpy()[:, cols.get_loc(
                    'x'):cols.get_loc('z')+1], rotation_lammps2ase)
                velocities_atoms = np.dot(atoms_df.to_numpy()[:, cols.get_loc(
                    'vx'):cols.get_loc('vz')+1], rotation_lammps2ase)

                if trajectory_out is not None:
                    tmp_atoms = Atoms(type_atoms, positions=positions_atoms,
                                      cell=cell_atoms)
                    tmp_atoms.set_velocities(velocities_atoms)
                    trajectory_out.write(tmp_atoms)

        # Leave some processes empty
        n_procs = n_cpus - 4

        # Initializing multiprocessing of dataframes
        in_queue = Queue(maxsize=60)
        out_queue = Manager().Queue(maxsize=60)

        write_progress = Value('L', 0)  # This is the NEXT step to be writen
        write_event = Event()

        # Reading files
        f = paropen(lammps_trj, 'r')

        # Starts the processes for dataframe processing
        processes = [Process(target=df_producer, args=(
            in_queue, out_queue, write_progress, write_event,)) for _ in range(n_procs)]
        # Start a process for read
        processes.append(Process(target=step_reader,
                                 args=(f, in_queue, n_procs,)))
        # And another process to write
        processes.append(Process(target=step_writer, args=(
            self.trajectory_out, out_queue, n_procs,)))

        for p in processes:
            p.daemon = True
            p.start()

        # close all the processes and exit
        for p in processes:
            p.join()
        f.close()


class Prism(object):

    def __init__(self, cell, pbc=(True, True, True), digits=10):
        """Create a lammps-style triclinic prism object from a cell

        The main purpose of the prism-object is to create suitable
        string representations of prism limits and atom positions
        within the prism.
        When creating the object, the digits parameter (default set to 10)
        specify the precision to use.
        lammps is picky about stuff being within semi-open intervals,
        e.g. for atom positions (when using create_atom in the in-file),
        x must be within [xlo, xhi).
        """
        a, b, c = cell
        an, bn, cn = [np.linalg.norm(v) for v in cell]

        alpha = np.arccos(np.dot(b, c) / (bn * cn))
        beta = np.arccos(np.dot(a, c) / (an * cn))
        gamma = np.arccos(np.dot(a, b) / (an * bn))

        xhi = an
        xyp = np.cos(gamma) * bn
        yhi = np.sin(gamma) * bn
        xzp = np.cos(beta) * cn
        yzp = (bn * cn * np.cos(alpha) - xyp * xzp) / yhi
        zhi = np.sqrt(cn**2 - xzp**2 - yzp**2)

        # Set precision
        self.car_prec = dec.Decimal('10.0') ** \
            int(np.floor(np.log10(max((xhi, yhi, zhi)))) - digits)
        self.dir_prec = dec.Decimal('10.0') ** (-digits)
        self.acc = float(self.car_prec)
        self.eps = np.finfo(xhi).eps

        # For rotating positions from ase to lammps
        Apre = np.array(((xhi, 0, 0),
                         (xyp, yhi, 0),
                         (xzp, yzp, zhi)))
        self.R = np.dot(np.linalg.inv(cell), Apre)

        # Actual lammps cell may be different from what is used to create R
        def fold(vec, pvec, i):
            p = pvec[i]
            x = vec[i] + 0.5 * p
            n = (np.mod(x, p) - x) / p
            return [float(self.f2qdec(a)) for a in (vec + n * pvec)]

        Apre[1, :] = fold(Apre[1, :], Apre[0, :], 0)
        Apre[2, :] = fold(Apre[2, :], Apre[1, :], 1)
        Apre[2, :] = fold(Apre[2, :], Apre[0, :], 0)

        self.A = Apre
        self.Ainv = np.linalg.inv(self.A)

        if self.is_skewed() and \
                (not (pbc[0] and pbc[1] and pbc[2])):
            raise RuntimeError('Skewed lammps cells MUST have '
                               'PBC == True in all directions!')

    def f2qdec(self, f):
        return dec.Decimal(repr(f)).quantize(self.car_prec, dec.ROUND_DOWN)

    def f2qs(self, f):
        return str(self.f2qdec(f))

    def f2s(self, f):
        return str(dec.Decimal(repr(f)).quantize(self.car_prec,
                                                 dec.ROUND_HALF_EVEN))

    def dir2car(self, v):
        """Direct to cartesian coordinates"""
        return np.dot(v, self.A)

    def car2dir(self, v):
        """Cartesian to direct coordinates"""
        return np.dot(v, self.Ainv)

    def fold_to_str(self, v):
        """Fold a position into the lammps cell (semi open)

        Returns tuple of str.
        """
        # Two-stage fold, first into box, then into semi-open interval
        # (within the given precision).
        d = [x % (1 - self.dir_prec) for x in
             map(dec.Decimal,
                 map(repr, np.mod(self.car2dir(v) + self.eps, 1.0)))]
        return tuple([self.f2qs(x) for x in
                      self.dir2car(list(map(float, d)))])

    def get_lammps_prism(self):
        A = self.A
        return A[0, 0], A[1, 1], A[2, 2], A[1, 0], A[2, 0], A[2, 1]

    def get_lammps_prism_str(self):
        """Return a tuple of strings"""
        p = self.get_lammps_prism()
        return tuple([self.f2s(x) for x in p])

    def positions_to_lammps_strs(self, positions):
        """Rotate an ase-cell position to the lammps cell orientation

        Returns tuple of str.
        """
        rot_positions = np.dot(positions, self.R)
        return [tuple([self.f2s(x) for x in position])
                for position in rot_positions]

    def pos_to_lammps_fold_str(self, position):
        """Rotate and fold an ase-cell position into the lammps cell

        Returns tuple of str.
        """
        return self.fold_to_str(np.dot(position, self.R))

    def is_skewed(self):
        acc = self.acc
        prism = self.get_lammps_prism()
        axy, axz, ayz = [np.abs(x) for x in prism[3:]]
        return (axy >= acc) or (axz >= acc) or (ayz >= acc)
