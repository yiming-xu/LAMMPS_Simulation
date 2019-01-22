# Author:
#   Yiming Xu, yiming.xu15@imperial.ac.uk
#
# This is a list of scripts that are somewhat useful
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy.random import rand

from ase import Atoms, units
from ase.build import molecule
from ase.data import atomic_numbers, covalent_radii, vdw_radii
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
from lammpsrun import LAMMPS, write_lammps_data


def read_float_or_fraction(number: str) -> float:
    # This is needed because sometimes GULP coordinates are printed in fractions
    if number.isdecimal():
        return float(number)
    else:
        split_by_dot = number.split('.')
        split_by_slash = number.split('/')
        if len(split_by_dot) == 2 and split_by_dot[0].isdecimal() and split_by_dot[1].isdecimal:
            return float(number)
        elif len(split_by_slash) == 2 and split_by_slash[0].isdecimal() and split_by_slash[1].isdecimal:
            return float(split_by_slash[0])/float(split_by_slash[1])
        else:
            raise ValueError


def read_gulp_trajectory(traj_file_list):
    """Reads a list of GULP opti trajectory files, and extracts the starting
    geometry and coordinate information into a list of ASE atoms."""

    trajectory = []
    for step in traj_file_list:
        cell = []
        position_str = []
        with open(step) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                # As the output is well structured, and we need only the structure information.
                # However, the end of coordinates is not well defined, so this is used instead.
                # More stopping criteria can be added.
                if line.startswith("species") or line.startswith("reaxFFtol"):
                    break

                if line.startswith("cell"):
                    pos_line = f.readline()
                    pos_line = pos_line.split()
                    cell = [read_float_or_fraction(x) for x in pos_line]

                elif line.startswith('frac'):
                    coord_line = f.readline()
                    # Trying to ensure that only relevant lines are read.
                    while "core" in coord_line.lower() or "shel" in coord_line.lower() or "bshe" in coord_line.lower():
                        position_str.append(coord_line.split())
                        coord_line = f.readline()

        # convert information to ASE atoms and append to trajectory
        atomic_symbols = [x[0] for x in position_str]
        positions = [[read_float_or_fraction(x)
                      for x in y[2:5]] for y in position_str]
        ase_atoms = Atoms(atomic_symbols, cell=cell, pbc=True)
        ase_atoms.set_scaled_positions(positions)
        trajectory.append(ase_atoms)
    return trajectory


def rotate_plot_atoms(atoms, radii=1.0, rotation_list=None, interval=40):

    if rotation_list is None:
        rotation_list = ['90x, {}y, 0z'.format(x) for x in range(360)]

    def update_plot_atoms(num):
        cur_rotation = rotation_list[num]
        ax.cla()
        ax.set_axis_off()
        plot_atoms(atoms, ax=ax, radii=radii, rotation=cur_rotation)
        return ax,

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ani = animation.FuncAnimation(fig, update_plot_atoms, len(rotation_list),
                                  interval=40, blit=False, repeat=True)
    return ani


def reaxff_params_generator(sim_box, job_name, input_fd="", write=False, **kwargs):

    list_of_elements = sorted(list(set(sim_box.get_chemical_symbols())))

    if 'potential' in kwargs.keys():
        potential = kwargs['potential']
    else:
        potential = 'ffield.reax.Fe_O_C_H_combined'

    # Default Parameters
    reaxff_params = {
        # Initialization
        "units": "real",
        "atom_style": "charge",
        # "velocity": ["all create 300.0 1050027 rot yes dist gaussian"],

        # Forcefield definition
        "pair_style": "reax/c NULL safezone 16",
        "pair_coeff": ['* * ' + '{0} '.format(potential) + ' '.join(list_of_elements)],
        "neighbor": "2.0 bin",
        "neighbor_modify": "delay 10 check yes",

        # Run and Minimization
        # "run": "1",
        "timestep": 1,
        "fix": ["all_nve all nve",
                "qeqreax all qeq/reax 1 0.0 10.0 1e-6 reax/c"]
    }

    for key in kwargs.keys():
        reaxff_params[key] = kwargs[key]

    write_lammps_data(os.path.join(input_fd, job_name + ".lammpsdata",),
                      sim_box, charges=True, force_skew=True)
    calc = LAMMPS(parameters=reaxff_params, always_triclinic=True)
    sim_box.set_calculator(calc)
    if write:
        calc.write_lammps_in(lammps_in=os.path.join(input_fd, "{0}.lammpsin".format(job_name)),
                             lammps_trj="{0}.lammpstrj".format(job_name),
                             lammps_data="{0}.lammpsdata".format(job_name))

    return calc


def get_coordination_number(sim_box, coordination_indices, cut_off=1.0, vdw_cut_off=1.0, cov_cut_off=1.0):
    i, j = nl_cutoff_cov_vdw(sim_box, cut_off, vdw_cut_off, cov_cut_off)

    indices, coord = np.unique(i, return_counts=True)
    coord_numbers = dict(zip(indices, coord))
    return [coord_numbers[x] for x in coordination_indices]


def identify_connection(i, j, num, connections=None):
    'Given a neighbor_list i, j, identifies all indices connected to num'
    if not connections:
        connections = [num]

    new_connections = j[i == num]
    to_check = [x for x in new_connections if x not in connections]
    connections.extend(to_check)

    for n in to_check:
        identify_connection(i, j, n, connections)

    return connections


def nl_cutoff_cov_vdw(sim_box, cut_off, vdw_cut_off=1.0, cov_cut_off=1.0):
    overlap_vdwr_sphere = [vdw_radii[atomic_numbers[x]]
                           for x in sim_box.get_chemical_symbols()]
    overlap_covr_sphere = [covalent_radii[atomic_numbers[x]]
                           for x in sim_box.get_chemical_symbols()]
    overlap_sphere = cut_off * \
        (vdw_cut_off * np.array(overlap_vdwr_sphere) +
         cov_cut_off * np.array(overlap_covr_sphere))/2

    i, j = neighbor_list('ij', sim_box, overlap_sphere, self_interaction=False)
    return i, j


def replace_molecule(sim_box, source_index, new_mol, cut_off=1.0, seed=None):
    'Replaces a molecule attached to an atom at source_index of the sim_box and replace it by new_mol'
    new_box = sim_box.copy()
    old_pbc = new_box.get_pbc()

    i, j = nl_cutoff_cov_vdw(new_box, cut_off)

    mol_to_delete_indices = identify_connection(i, j, num=source_index)
    mol_to_delete_COM = new_box[mol_to_delete_indices].get_center_of_mass()

    new_mol.translate(mol_to_delete_COM)
    del new_box[mol_to_delete_indices]

    new_box.extend(new_mol)
    new_box.set_pbc(old_pbc)

    return new_box


def create_water_region(cell):
    'Creates a region of water with approximate density at 300K'

    aq_cell = cell
    H2O_volume = (1e+27)/(1000/18 * units.mol)
    Atoms_grid = Atoms('Ne', pbc=True, cell=[H2O_volume**(1/3)]*3)

    Atoms_grid = Atoms_grid.repeat(
        tuple([int(x/(H2O_volume**(1/3))) for x in aq_cell]))

    Atoms_grid.set_cell(aq_cell)
    Atoms_grid.center()
    Atoms_grid.rattle(stdev=H2O_volume**(1/3)/10)

    H2O_bulk = molecule('H2O')
    H2O_bulk.euler_rotate(rand()*360, rand()*360, rand()*360, center="COM")
    H2O_bulk.translate(Atoms_grid.get_positions()[0])
    H2O_bulk.set_cell(Atoms_grid.get_cell())
    H2O_bulk.set_pbc(True)

    for new_pos in Atoms_grid.get_positions()[1:]:
        new_molecule = molecule('H2O')
        new_molecule.euler_rotate(
            rand()*360, rand()*360, rand()*360, center="COM")
        new_molecule.translate(new_pos)

        H2O_bulk += new_molecule.copy()
    H2O_bulk.center()

    return H2O_bulk
