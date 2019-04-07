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
from re import IGNORECASE
from re import compile as re_compile
from subprocess import PIPE, Popen
from tempfile import NamedTemporaryFile, mkdtemp
from tempfile import mktemp as uns_mktemp
from threading import Thread

import numpy as np
from ase import Atoms
from ase.data import (atomic_masses, atomic_names, atomic_numbers,
                      covalent_radii, vdw_radii)
from ase.parallel import paropen
from ase.units import Ang, GPa, fs
from ase.utils import basestring

__all__ = ['LAMMPS', 'write_lammps_data']

# "End mark" used to indicate that the calculation is done
CALCULATION_END_MARK = '__end_of_ase_invoked_calculation__'


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
        specorder: list
            Short explanation XXX
        keep_tmp_files: bool
            Retain any temporary files created. Mostly useful for debugging.
        tmp_dir: str
            path/dirname (default None -> create automatically).
            Explicitly control where the calculator object should create
            its files. Using this option implies 'keep_tmp_files'
        no_data_file: bool
            Controls whether an explicit data file will be used for feeding
            atom coordinates into lammps. Enable it to lessen the pressure on
            the (tmp) file system. THIS OPTION MIGHT BE UNRELIABLE FOR CERTAIN
            CORNER CASES (however, if it fails, you will notice...).
        keep_alive: bool
            When using LAMMPS as a spawned subprocess, keep the subprocess
            alive (but idling when unused) along with the calculator object.
        always_triclinic: bool
            Force use of a triclinic cell in LAMMPS, even if the cell is
            a perfect parallelepiped.
        """

        self.label = label
        self.parameters = parameters
        self.specorder = specorder
        self.files = files
        self.always_triclinic = always_triclinic
        self.calls = 0
        self.forces = None
        self.keep_alive = keep_alive
        self.keep_tmp_files = keep_tmp_files
        self.no_data_file = no_data_file

        # if True writes velocities from atoms.get_velocities() to LAMMPS input
        self.write_velocities = False

        # file object, if is not None the trajectory will be saved in it
        self.trajectory_out = None

        # period of system snapshot saving (in MD steps)
        self.dump_period = 1
        if tmp_dir is not None:
            # If tmp_dir is pointing somewhere, don't remove stuff!
            self.keep_tmp_files = True
        self._lmp_handle = None        # To handle the lmp process

        # read_log depends on that the first (three) thermo_style custom args
        # can be capitilized and matched against the log output. I.e.
        # don't use e.g. 'ke' or 'cpu' which are labeled KinEng and CPU.
        self._custom_thermo_args = ['step', 'temp', 'press', 'cpu',
                                    'pxx', 'pyy', 'pzz', 'pxy', 'pxz', 'pyz',
                                    'ke', 'pe', 'etotal',
                                    'vol', 'lx', 'ly', 'lz', 'atoms']
        self._custom_thermo_mark = ' '.join([x.capitalize() for x in
                                             self._custom_thermo_args[0:3]])

        # Match something which can be converted to a float
        f_re = r'([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|nan|inf))'
        n = len(self._custom_thermo_args)
        # Create a re matching exactly N white space separated floatish things
        self._custom_thermo_re = re_compile(
            r'^\s*' + r'\s+'.join([f_re] * n) + r'\s*$', flags=IGNORECASE)
        # thermo_content contains data "written by" thermo_style.
        # It is a list of dictionaries, each dict (one for each line
        # printed by thermo_style) contains a mapping between each
        # custom_thermo_args-argument and the corresponding
        # value as printed by lammps. thermo_content will be
        # re-populated by the read_log method.
        self.thermo_content = []

        if tmp_dir is None:
            self.tmp_dir = mkdtemp(prefix='LAMMPS-')
        else:
            self.tmp_dir = os.path.realpath(tmp_dir)
            if not os.path.isdir(self.tmp_dir):
                os.mkdir(self.tmp_dir, 0o755)

        for f in files:
            shutil.copy(f, os.path.join(self.tmp_dir, os.path.basename(f)))

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        self.prism = Prism(atoms.get_cell())

    def clean(self, force=False):

        self._lmp_end()

        if not self.keep_tmp_files:
            shutil.rmtree(self.tmp_dir)

    def get_potential_energy(self, atoms):
        self.update(atoms)
        return self.thermo_content[-1]['pe']

    def get_forces(self, atoms):
        self.update(atoms)
        return self.forces

    def get_stress(self, atoms):
        self.update(atoms)
        tc = self.thermo_content[-1]
        # 1 bar (used by lammps for metal units) = 1e-4 GPa
        return np.array([tc[i] for i in ('pxx', 'pyy', 'pzz', 'pyz', 'pxz',
                                         'pxy')]) * (-1e-4 * GPa)

    def update(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.calculate(atoms)

    def calculate(self, atoms):
        self.atoms = atoms.copy()
        pbc = self.atoms.get_pbc()
        if all(pbc):
            cell = self.atoms.get_cell()
        elif not any(pbc):
            # large enough cell for non-periodic calculation -
            # LAMMPS shrink-wraps automatically via input command
            #       "periodic s s s"
            # below
            cell = 2 * np.max(np.abs(self.atoms.get_positions())) * np.eye(3)
        else:
            print("WARNING: semi-periodic ASE cell detected - translation")
            print("         to proper LAMMPS input cell might fail")
            cell = self.atoms.get_cell()
        self.prism = Prism(cell)
        self.run()

    def _lmp_alive(self):
        # Return True if this calculator is currently handling a running
        # lammps process
        return self._lmp_handle and not isinstance(
            self._lmp_handle.poll(), int)

    def _lmp_end(self):
        # Close lammps input and wait for lammps to end. Return process
        # return value
        if self._lmp_alive():
            self._lmp_handle.stdin.close()
            return self._lmp_handle.wait()

    def run(self, set_atoms=False):
        """Method which explicitly runs LAMMPS."""

        self.calls += 1

        # set LAMMPS command from environment variable
        if 'LAMMPS_COMMAND' in os.environ:
            lammps_cmd_line = shlex.split(os.environ['LAMMPS_COMMAND'],
                                          posix=(os.name == 'posix'))
            if len(lammps_cmd_line) == 0:
                self.clean()
                raise RuntimeError('The LAMMPS_COMMAND environment variable '
                                   'must not be empty')
            # want always an absolute path to LAMMPS binary when calling from
            # self.dir
            lammps_cmd_line[0] = os.path.abspath(lammps_cmd_line[0])

        else:
            self.clean()
            raise RuntimeError(
                'Please set LAMMPS_COMMAND environment variable')
        if 'LAMMPS_OPTIONS' in os.environ:
            lammps_options = shlex.split(os.environ['LAMMPS_OPTIONS'],
                                         posix=(os.name == 'posix'))
        else:
            lammps_options = shlex.split('-echo log -screen none',
                                         posix=(os.name == 'posix'))

        # change into subdirectory for LAMMPS calculations
        cwd = os.getcwd()
        os.chdir(self.tmp_dir)

        # setup file names for LAMMPS calculation
        label = '{0}{1:>06}'.format(self.label, self.calls)
        lammps_in = uns_mktemp(prefix='in_' + label, dir=self.tmp_dir)
        lammps_log = uns_mktemp(prefix='log_' + label, dir=self.tmp_dir)
        lammps_trj_fd = NamedTemporaryFile(
            prefix='trj_' + label, dir=self.tmp_dir,
            delete=(not self.keep_tmp_files))
        lammps_trj = lammps_trj_fd.name
        if self.no_data_file:
            lammps_data = None
        else:
            lammps_data_fd = NamedTemporaryFile(
                prefix='data_' + label, dir=self.tmp_dir,
                delete=(not self.keep_tmp_files))
            self.write_lammps_data(lammps_data=lammps_data_fd)
            lammps_data = lammps_data_fd.name
            lammps_data_fd.flush()

        # see to it that LAMMPS is started
        if not self._lmp_alive():
            # Attempt to (re)start lammps
            self._lmp_handle = Popen(
                #### lammps_cmd_line + lammps_options + ['-log', '/dev/stdout'],
                lammps_cmd_line + lammps_options,
                stdin=PIPE, stdout=PIPE)
        lmp_handle = self._lmp_handle

        # Create thread reading lammps stdout (for reference, if requested,
        # also create lammps_log, although it is never used)
        if self.keep_tmp_files:
            lammps_log_fd = open(lammps_log, 'wb')
            fd = SpecialTee(lmp_handle.stdout, lammps_log_fd)
        else:
            fd = lmp_handle.stdout
        thr_read_log = Thread(target=self.read_lammps_log, args=(fd,))
        thr_read_log.start()

        # write LAMMPS input (for reference, also create the file lammps_in,
        # although it is never used)
        if self.keep_tmp_files:
            lammps_in_fd = open(lammps_in, 'wb')
            fd = SpecialTee(lmp_handle.stdin, lammps_in_fd)
        else:
            fd = lmp_handle.stdin
        self.write_lammps_in(lammps_in=fd, lammps_trj=lammps_trj,
                             lammps_data=lammps_data)

        if self.keep_tmp_files:
            lammps_in_fd.close()

        # Wait for log output to be read (i.e., for LAMMPS to finish)
        # and close the log file if there is one
        thr_read_log.join()
        if self.keep_tmp_files:
            lammps_log_fd.close()

        if not self.keep_alive:
            self._lmp_end()

        exitcode = lmp_handle.poll()
        if exitcode and exitcode != 0:
            cwd = os.getcwd()
            raise RuntimeError(
                'LAMMPS exited in {0} with exit code: {1}.'.format(cwd, exitcode))

        # A few sanity checks
        if len(self.thermo_content) == 0:
            raise RuntimeError('Failed to retrieve any thermo_style-output')
        if int(self.thermo_content[-1]['atoms']) != len(self.atoms):
            # This obviously shouldn't happen, but if prism.fold_...() fails,
            # it could
            raise RuntimeError('Atoms have gone missing')

        self.read_lammps_trj(lammps_trj=lammps_trj, set_atoms=set_atoms)
        lammps_trj_fd.close()
        if not self.no_data_file:
            lammps_data_fd.close()

        os.chdir(cwd)

    def write_lammps_data(self, lammps_data=None):
        """Method which writes a LAMMPS data file with atomic structure."""
        if lammps_data is None:
            lammps_data = 'data.' + self.label
        write_lammps_data(
            lammps_data, self.atoms, self.specorder,
            force_skew=self.always_triclinic, prismobj=self.prism,
            velocities=self.write_velocities)

    def write_lammps_in(self, lammps_in=None, lammps_trj=None,
                        lammps_data=None):
        """Write a LAMMPS in_ file with run parameters and settings."""

        def write_var(f, parameter, default=None):
            if parameter in self.parameters:
                value = self.parameters[parameter]
                if isinstance(value, list):
                    for v in value:
                        f.write("{:14} {} \n".format(parameter, v).encode('utf-8'))
                else:
                    f.write("{:14} {} \n".format(parameter, value).encode('utf-8'))
            elif default:
                # f.write("# Default values for {} used.\n".format(parameter).encode('utf-8'))
                f.write("{:14} {} \n".format(parameter, default).encode('utf-8'))
            else:
                # f.write('# !!!Parameter {} not found!\n'.format(parameter).encode('utf-8'))
                pass

        def write_box_and_atoms(f):
            if self.keep_tmp_files:
                f.write('## Original ase cell\n'.encode('utf-8'))
                f.write(''.join(['# {0:.16} {1:.16} {2:.16}\n'.format(*x)
                                 for x in self.atoms.get_cell()]
                                ).encode('utf-8'))
            write_var(f, 'lattice', 'sc 1.0')
            xhi, yhi, zhi, xy, xz, yz = self.prism.get_lammps_prism_str()
            if self.always_triclinic or self.prism.is_skewed():
                write_var(f, 'region', 'asecell prism 0.0 {0} 0.0 {1} 0.0 {2} {3} {4} {5} side in units box'.format(xhi, yhi, zhi, xy, xz, yz))
            else:
                write_var(f, 'region', 'asecell block 0.0 {0} 0.0 {1} 0.0 {2} side in units box'.format(xhi, yhi, zhi))

            symbols = self.atoms.get_chemical_symbols()
            if self.specorder is None:
                # By default, atom types in alphabetic order
                species = sorted(set(symbols))
            else:
                # By request, specific atom type ordering
                species = self.specorder

            n_atom_types = len(species)
            species_i = dict([(s, i + 1) for i, s in enumerate(species)])

            write_var(f, 'create_box', '{0} asecell'.format(n_atom_types))

            f.write('\n# By default, atom types in alphabetic order\n'.encode('utf-8'))

            for s, pos in zip(symbols, self.atoms.get_positions()):
                if self.keep_tmp_files:
                    f.write('# atom pos in ase cell: {0:.16} {1:.16} {2:.16}'
                            '\n'.format(*tuple(pos)).encode('utf-8'))

                write_var(f, 'create_atoms', '{0} single {1} {2} {3} units box\n'.format(
                            *((species_i[s],) + self.prism.pos_to_lammps_fold_str(pos))))

        if isinstance(lammps_in, basestring):
            f = paropen(lammps_in, 'wb')
            close_in_file = True
        else:
            # Expect lammps_in to be a file-like object
            f = lammps_in
            close_in_file = False

        if self.keep_tmp_files:
            f.write('# (written by ASE)\n'.encode('utf-8'))

        # Write variables
        f.write(('clear\n'
                 'variable\t dump_file string "{0}"\n'
                 'variable\t data_file string "{1}"\n'
                 ).format(lammps_trj, lammps_data).encode('utf-8'))

        # Writing commands by category: https://lammps.sandia.gov/doc/Commands_category.html
        # This is the general structure of how commands flow
        # TODO: customization within each section with a flag

        # Initialization/Settings
        f.write('\n # Initialization/Settings \n'.encode('utf-8'))
        write_var(f, 'newton')
        write_var(f, 'package')
        write_var(f, 'units')
        write_var(f, 'neighbor')
        write_var(f, 'neigh_modify')

        # Atoms Settings
        f.write('\n # Atoms \n'.encode('utf-8'))
        write_var(f, 'atom_style', 'metal')
        write_var(f, 'atom_modify')
        write_var(f, 'boundary', '{0} {1} {2}'.format(*tuple('sp'[x] for x in self.atoms.get_pbc())))

        # If not using data file, write the simulation box and the atoms
        f.write('\n # Basic Simulation Box and Atoms \n'.encode('utf-8'))
        if self.no_data_file:
            write_box_and_atoms(f)
        else:
            write_var(f, 'read_data', lammps_data)

        # Additional Setup Simulation Box and Atoms Setup
        f.write('\n # Additional Setup Simulation Box and Atoms Setup \n'.encode('utf-8'))
        write_var(f, 'region')
        write_var(f, 'group')
        write_var(f, 'mass')
        write_var(f, 'velocity')
        write_var(f, 'replicate')

        # Interaction Setup
        f.write('\n # Interactions Setup \n'.encode('utf-8'))
        write_var(f, 'pair_style')
        write_var(f, 'pair_coeff')
        write_var(f, 'pair_modify')
        write_var(f, 'kspace_style')

        write_var(f, 'bond_style')
        write_var(f, 'bond_coeff')

        write_var(f, 'angle_style')
        write_var(f, 'angle_coeff')

        # Simulation Run Setup
        f.write('\n # Simulation Run Setup \n'.encode('utf-8'))
        write_var(f, 'fix', 'fix_nve all nve')
        write_var(f, 'dump',  'dump_all all custom {1} "{0}" id type x y z vx vy vz fx fy fz'.format(lammps_trj, self.dump_period))

        write_var(f, 'thermo_style', 'custom {0}'.format(' '.join(self._custom_thermo_args)))
        write_var(f, 'thermo_modify', 'flush yes')
        write_var(f, 'thermo', '1')

        write_var(f, 'timestep', '1')
        write_var(f, 'restart')

        write_var(f, 'min_style')
        write_var(f, 'min_modify')
        write_var(f, 'minimize')
        
        write_var(f, 'run')

        f.write('print "{0}" \n'.format(CALCULATION_END_MARK).encode('utf-8'))
        # Force LAMMPS to flush log
        #### f.write('log /dev/stdout\n'.encode('utf-8'))

        f.flush()
        if close_in_file:
            f.close()

    def read_lammps_log(self, lammps_log=None, PotEng_first=False):
        """Method which reads a LAMMPS output log file."""

        if lammps_log is None:
            lammps_log = self.label + '.log'

        if isinstance(lammps_log, basestring):
            f = paropen(lammps_log, 'rb')
            close_log_file = True
        else:
            # Expect lammps_in to be a file-like object
            f = lammps_log
            close_log_file = False

        thermo_content = []
        line = f.readline().decode('utf-8')
        while line and line.strip() != CALCULATION_END_MARK:
            # get thermo output
            if line.startswith(self._custom_thermo_mark):
                m = True
                while m:
                    line = f.readline().decode('utf-8')
                    m = self._custom_thermo_re.match(line)
                    if m:
                        # create a dictionary between each of the
                        # thermo_style args and it's corresponding value
                        thermo_content.append(
                            dict(zip(self._custom_thermo_args,
                                     map(float, m.groups()))))
            else:
                line = f.readline().decode('utf-8')

        if close_log_file:
            f.close()

        self.thermo_content = thermo_content

    def read_lammps_trj(self, lammps_trj=None, set_atoms=False):
        """Method which reads a LAMMPS dump file."""
        import pandas as pd
        from io import StringIO
        if lammps_trj is None:
            lammps_trj = self.label + '.lammpstrj'

        f = paropen(lammps_trj, 'r')
        for line in f:
            # Read the file into memory
            if line.startswith('ITEM: TIMESTEP'):
                _step_number = int(next(f).strip())
            elif line.startswith('ITEM: NUMBER OF ATOMS'):
                n_atoms = int(next(f).strip())
            elif line.startswith('ITEM: BOX BOUNDS'):
                _box_str = [next(f).rstrip().split(' ') for x in range(3)]
                # box_df = pd.DataFrame(
                #     box_str, columns=['lo', 'hi', 'tilt'], dtype=float)
            elif line.startswith('ITEM: ATOMS'):
                # atoms_str = [next(f).rstrip().split(' ')
                #              for x in range(n_atoms)]
                atoms_str = ''.join([next(f) for x in range(n_atoms)])
                atoms_df = pd.read_csv(StringIO(line[12:] + atoms_str), delim_whitespace=True, index_col=0)
                atoms_df.sort_index(inplace=True)
                
                #atoms_df.sort_values(by='id', inplace=True)
                # Create appropriate atoms object
                # Determine cell tilt for triclinic case
                # if not box_df.tilt.isnull().values.any():
                #     # Assuming default order
                #     xy = box_df.tilt[0]
                #     xz = box_df.tilt[1]
                #     yz = box_df.tilt[2]

                # xhilo = (box_df.hi[0] - box_df.lo[0]) - xy - xz
                # yhilo = (box_df.hi[1] - box_df.lo[1]) - yz
                # zhilo = (box_df.hi[2] - box_df.lo[2])

                if self.atoms:
                    cell_atoms = self.atoms.get_cell()

                    # BEWARE: reconstructing the rotation from the LAMMPS
                    #         output trajectory file fails in case of shrink
                    #         wrapping for a non-periodic direction
                    #      -> hence rather obtain rotation from prism object
                    #         used to generate the LAMMPS input
                    # rotation_lammps2ase = np.dot(
                    #               np.linalg.inv(np.array(cell)), cell_atoms)
                    rotation_lammps2ase = np.linalg.inv(self.prism.R)

                    type_atoms = self.atoms.get_atomic_numbers()
                    cols = atoms_df.columns
                    positions_atoms = np.dot(atoms_df.to_numpy()[:, cols.get_loc('x'):cols.get_loc('z')+1], rotation_lammps2ase)
                    velocities_atoms = np.dot(atoms_df.to_numpy()[:, cols.get_loc('vx'):cols.get_loc('vz')+1], rotation_lammps2ase)
                    forces_atoms = np.dot(atoms_df.to_numpy()[:, cols.get_loc('fx'):cols.get_loc('fz')+1], rotation_lammps2ase)

                if set_atoms:
                    # assume periodic boundary conditions here (as in
                    # write_lammps)
                    self.atoms = Atoms(type_atoms, positions=positions_atoms,
                                       cell=cell_atoms)
                    self.atoms.set_velocities(velocities_atoms
                                              * (Ang/(fs*1000.)))

                self.forces = forces_atoms
                if self.trajectory_out is not None:
                    tmp_atoms = Atoms(type_atoms, positions=positions_atoms,
                                      cell=cell_atoms)
                    tmp_atoms.set_velocities(velocities_atoms)
                    self.trajectory_out.write(tmp_atoms)
        f.close()


class SpecialTee(object):
    """A special purpose, with limited applicability, tee-like thing.

    A subset of stuff read from, or written to, orig_fd,
    is also written to out_fd.
    It is used by the lammps calculator for creating file-logs of stuff
    read from, or written to, stdin and stdout, respectively.
    """

    def __init__(self, orig_fd, out_fd):
        self._orig_fd = orig_fd
        self._out_fd = out_fd
        self.name = orig_fd.name

    def write(self, data):
        self._orig_fd.write(data)
        self._out_fd.write(data)
        self.flush()

    def read(self, *args, **kwargs):
        data = self._orig_fd.read(*args, **kwargs)
        self._out_fd.write(data)
        return data

    def readline(self, *args, **kwargs):
        data = self._orig_fd.readline(*args, **kwargs)
        self._out_fd.write(data)
        return data

    def readlines(self, *args, **kwargs):
        data = self._orig_fd.readlines(*args, **kwargs)
        self._out_fd.write(''.join(data))
        return data

    def flush(self):
        self._orig_fd.flush()
        self._out_fd.flush()


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


def write_lammps_data(fileobj, atoms, specorder=None, force_skew=False,
                      prismobj=None, velocities=False, charges=False):
    """Write atomic structure data to a LAMMPS data_file."""
    if isinstance(fileobj, basestring):
        f = paropen(fileobj, 'wb')
        close_file = True
    else:
        # Presume fileobj acts like a fileobj
        f = fileobj
        close_file = False

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                'Can only write one configuration to a lammps data file!')
        atoms = atoms[0]

    f.write('{0} (written by ASE) \n\n'.format(f.name).encode('utf-8'))

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    f.write('{0} atoms \n'.format(n_atoms).encode('utf-8'))

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        species = sorted(set(symbols))
    else:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
    n_atom_types = len(species)
    f.write('{0} atom types\n'.format(n_atom_types).encode('utf-8'))

    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj
    xhi, yhi, zhi, xy, xz, yz = p.get_lammps_prism_str()

    f.write('0.0 {0}  xlo xhi\n'.format(xhi).encode('utf-8'))
    f.write('0.0 {0}  ylo yhi\n'.format(yhi).encode('utf-8'))
    f.write('0.0 {0}  zlo zhi\n'.format(zhi).encode('utf-8'))

    if force_skew or p.is_skewed():
        f.write('{0} {1} {2}  xy xz yz\n'.format(xy, xz, yz).encode('utf-8'))
    f.write('\n'.encode('utf-8'))

    f.write('Masses \n\n'.encode('utf-8'))
    for s in species:
        i = species.index(s) + 1
        f.write('{0:>2d} {1:>.4f}\n'.format(
            i, atomic_masses[atomic_numbers[s]]).encode('utf-8'))
    f.write('\n'.encode('utf-8'))

    f.write('Atoms \n\n'.encode('utf-8'))

    # Temporary support for adding a charge
    if charges:
        for i, (r, q) in enumerate(zip(p.positions_to_lammps_strs(atoms.get_positions()), atoms.get_initial_charges())):
            s = species.index(symbols[i]) + 1
            f.write('{0:>6} {1:>3} {5:>10.6f} {2:>14} {3:>14} {4:>14}\n'.format(
                    *(i + 1, s) + tuple(r), q).encode('utf-8'))
    else:
        for i, r in enumerate(p.positions_to_lammps_strs(atoms.get_positions())):
            s = species.index(symbols[i]) + 1
            f.write('{0:>6} {1:>3} {2:>14} {3:>14} {4:>14}\n'.format(
                    *(i + 1, s) + tuple(r)).encode('utf-8'))

    if velocities and atoms.get_velocities() is not None:
        f.write('\n\nVelocities \n\n'.encode('utf-8'))
        for i, v in enumerate(atoms.get_velocities() / (Ang/(fs*1000.))):
            f.write('{0:>6} {1:>.10f} {2:>.10f} {3:>.10f}\n'.format(
                    *(i + 1,) + tuple(v)).encode('utf-8'))

    f.flush()
    if close_file:
        f.close()


if __name__ == '__main__':
    pair_style = 'eam'
    Pd_eam_file = 'Pd_u3.eam'
    pair_coeff = ['* * ' + Pd_eam_file]
    parameters = {'pair_style': pair_style, 'pair_coeff': pair_coeff}
    files = [Pd_eam_file]
    calc = LAMMPS(parameters=parameters, files=files)
    a0 = 3.93
    b0 = a0 / 2.0
    if True:
        bulk = Atoms(
            ['Pd'] * 4,
            positions=[(0, 0, 0), (b0, b0, 0), (b0, 0, b0), (0, b0, b0)],
            cell=[a0] * 3, pbc=True)
        # test get_forces
        print('forces for a = {0}'.format(a0))
        print(calc.get_forces(bulk))
        # single points for various lattice constants
        bulk.set_calculator(calc)
        for n in range(-5, 5, 1):
            a = a0 * (1 + n / 100.0)
            bulk.set_cell([a] * 3)
            print('a : {0} , total energy : {1}'.format(
                a, bulk.get_potential_energy()))

    calc.clean()
