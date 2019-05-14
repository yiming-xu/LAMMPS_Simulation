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

import os
from multiprocessing import Event, Manager, Process, Queue, Value

import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read
from ase.parallel import paropen


class bonds_analysis:
    def __init__(self, atom_file=None):
        """The bond analysis class

        bond_file: 
            The bond file usually with the extension .tatb
        atom_file:
            A file that can be normally parsed by ASE as an atoms object.
            .extxyz is generally expected
        """
        if atom_file:
            self.atoms = read(atom_file, 'r')

    def read_reaxff_bond(self, bond_in=None, bond_out=None, n_cpus=8):
        """Method to read, parse and save the output in a convenient format"""

        def df_producer(in_queue, out_queue, write_progress, write_event):
            while True:
                step_str = in_queue.get()

                if isinstance(step_str, str):
                    out_queue[0].put('DONE')
                    out_queue[1].put('DONE')
                    break
                else:
                    step_count = step_str[0]
                    step_data = step_str[1]

                    atom_table = []
                    connectivity_table = []

                    # As a reminder, each step is in the form:
                    # id type nb id_1...id_nb mol bo_1...bo_nb abo nlp q
                    # Due to the varying number of points, we unfortunately
                    # have to parse and analyse line by line
                    for line in step_data:
                        atom_id = line[0]
                        atom_type = line[1]
                        charge = line[-1]

                        # need to cast n_bonds to int to know how many variables there are
                        n_bonds = int(line[2])

                        mol_id = line[3+n_bonds]  # tends to be 0
                        b = line[3:3+n_bonds]
                        bo = line[4+n_bonds:4+2*n_bonds]

                        atom_table.append([atom_id, atom_type, mol_id, charge])
                        for i, j in zip(b, bo):
                            connectivity_table.append([atom_id, i, j])

                    atom_df = pd.DataFrame(atom_table, columns=[
                                           'atom_id', 'atom_type', 'mol_id', 'charge'])
                    atom_df = atom_df.astype({'atom_id': np.uint32,
                                              'atom_type': np.uint8,
                                              'mol_id': np.uint8,
                                              'charge': np.float32})
                    atom_df.set_index('atom_id', inplace=True)
                    atom_df.sort_index(inplace=True)

                    connectivity_df = pd.DataFrame(connectivity_table, columns=[
                                                   'atom_a', 'atom_b', 'bond_order'])
                    connectivity_df = connectivity_df.astype({'atom_a': np.uint32,
                                                              'atom_b': np.uint32,
                                                              'bond_order': np.float32})

                    while step_count != write_progress.value:
                        write_event.wait(timeout=None)

                    write_event.clear()

                    out_queue[0].put((step_count, atom_df))
                    out_queue[1].put((step_count, connectivity_df))

                    with write_progress.get_lock():
                        write_progress.value += 1
                    write_event.set()

        def step_reader(f, in_queue, n_procs):
            """ This function reads 1 step of the trajectory file and sends it to
            in_queue to be further processed
            """
            step_count = 0
            try:
                for line in f:
                    _step_number = int(line.split()[-1])
                    assert next(f).startswith('#')
                    n_atoms = int(next(f).split()[-1])
                    assert all([next(f).startswith('#') for _ in range(4)])

                    # Read the whole step and add to in_queue
                    in_queue.put((step_count, [next(f).split()
                                               for _ in range(n_atoms)]))
                    # Each step ends with '#'
                    assert next(f).startswith('#')
                    step_count += 1
            except Exception as e:
                print(e)
                [in_queue.put('DONE') for _ in range(n_procs)]

            # Poison pill when there is no more line to read
            [in_queue.put('DONE') for _ in range(n_procs)]

        def step_writer_atoms(bond_out, out_queue, n_procs):
            """ This function gets 1 step from out_queue, calculates
            the needful and writes it to file.
            """
            # with pd.HDFStore(bond_out, 'w') as store:
            write_counter = 0
            atoms_store = pd.HDFStore(bond_out+'_atoms.hdf5', mode='w')
            while write_counter < n_procs:
                msg = out_queue.get()

                if isinstance(msg, str):
                    write_counter += 1
                else:
                    step_number = msg[0]
                    atom_df = msg[1]
                    atoms_store.append(key='step{}'.format(step_number),
                                       value=atom_df,
                                       complevel=9,
                                       complib='blosc:snappy')
            atoms_store.close()

        def step_writer_connectivity(bond_out, out_queue, n_procs):
            """ This function gets 1 step from out_queue, calculates
            the needful and writes it to file.
            """
            # with pd.HDFStore(bond_out, 'w') as store:
            write_counter = 0
            connectivity_store = pd.HDFStore(
                bond_out+'_connectivity.hdf5', mode='w')
            while write_counter < n_procs:
                msg = out_queue.get()

                if isinstance(msg, str):
                    write_counter += 1
                else:
                    step_number = msg[0]
                    connectivity_df = msg[1]
                    connectivity_store.append(key='step{}'.format(step_number),
                                              value=connectivity_df,
                                              complevel=9,
                                              complib='blosc:snappy',
                                              index=False)
            connectivity_store.close()

        # Leave some processes empty
        n_procs = n_cpus - 3

        # Initializing multiprocessing of dataframes
        in_queue = Queue(maxsize=60)
        out_queue_atoms = Manager().Queue(maxsize=60)
        out_queue_connectivity = Manager().Queue(maxsize=60)

        write_progress = Value('L', 0)  # This is the NEXT step to be writen
        write_event = Event()

        # Reading and files
        f_in = paropen(bond_in, 'r')

        # Starts the processes for dataframe processing
        processes = [Process(target=df_producer, args=(
            in_queue, (out_queue_atoms, out_queue_connectivity), write_progress, write_event,)) for _ in range(n_procs)]

        # Start a process for read
        processes.append(Process(target=step_reader,
                                 args=(f_in, in_queue, n_procs,)))
        # And processes to write
        processes.append(Process(target=step_writer_atoms,
                                 args=(bond_out, out_queue_atoms, n_procs,)))
        processes.append(Process(target=step_writer_connectivity,
                                 args=(bond_out, out_queue_connectivity, n_procs,)))
        for p in processes:
            p.daemon = True
            p.start()

        # close all the processes and exit
        for p in processes:
            p.join()

        # close all the files
        f_in.close()
