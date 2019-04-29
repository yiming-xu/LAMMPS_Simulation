# Author:
#   Yiming Xu, yiming.xu15@imperial.ac.uk
#
# This is a list of scripts that are somewhat useful
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from numpy.random import rand
from fractions import Fraction

from ase import Atoms, units
from ase.build import molecule
from ase.data import atomic_numbers, covalent_radii, vdw_radii
from ase.neighborlist import neighbor_list
from ase.visualize.plot import plot_atoms
from ase.io import read, write

ephemeral = r"/rds/general/user/yx6015/ephemeral/"
convert_scripts_dir = r"/rds/general/user/yx6015/home/LAMMPS_Simulation/HPC_Scripts/"
sim_path = r"/rds/general/user/yx6015/home/LAMMPS_Simulation/HPC_Jupyter/"

class LAMMPScalculation:
    """ This class automates and stores some common functions used to set-up a LAMMPS calculation.
    """

    def __init__(self, atoms, label, temp_fd="temp_lammps", **kwargs):
        from lammpsrun import Prism
        from lammps import IPyLammps
        from ase.data import atomic_numbers, atomic_masses

        self.atoms = atoms.copy()
        symbols = self.atoms.get_chemical_symbols()
        species = sorted(set(symbols))

        n_atom_types = len(species)

        self.prism = Prism(self.atoms.get_cell())
        xhi, yhi, zhi, xy, xz, yz = self.prism.get_lammps_prism_str()
        species_i = dict([(s, i + 1) for i, s in enumerate(species)])

        if 'potential' in kwargs.keys():
            potential = kwargs['potential']
        else:
            potential = 'ffield.reax.Fe_O_C_H_combined'

        self.label = label
        self.temp_fd = os.path.join(os.getcwd(), temp_fd)

        self.dump_file = os.path.join(
            self.temp_fd, "{}.lammpstrj".format(label))
        self.log_file = os.path.join(self.temp_fd, "{}.log".format(label))

        reaxff_params = {"periodicity": ["p", "p", "p"],
                         "timestep": 1}
        for key in kwargs.keys():
            reaxff_params[key] = kwargs[key]

        # Setting up calculation
        # These parameter should not change
        self.calc = IPyLammps()
        self.calc.log(self.log_file)

        self.calc.units("real")
        self.calc.atom_style("charge")

        self.calc.boundary(*reaxff_params["periodicity"])
        self.calc.neighbor(2.0, "bin")
        self.calc.neigh_modify("delay", 10, "check", "yes")

        self.calc.lattice("sc", 1.0)
        # xlo xhi ylo yhi zlo zhi
        self.calc.region("asecell", "prism", 0.0, xhi, 0.0, yhi,
                         0.0, zhi, xy, xz, yz, 'side in', 'units box')
        self.calc.create_box(n_atom_types, "asecell")

        # Writing atomic attributes
        for s, pos in zip(symbols, self.atoms.get_positions()):
            self.calc.create_atoms(
                species_i[s], "single", *self.prism.pos_to_lammps_fold_str(pos))
        for k, v in species_i.items():
            self.calc.mass(v, atomic_masses[atomic_numbers[k]])

        self.calc.timestep(reaxff_params['timestep'])
        self.calc.pair_style("reax/c ", "NULL", "safezone", 20, "mincap", 100)
        self.calc.pair_coeff("* *", potential, *species_i)

    def set_fixes(self):
        self.calc.fix("qeqreax", "all", "qeq/reax",
                      1, 0.0, 10.0, "1e-6", "reax/c")

    def set_output(self):
        self.calc.dump("dump_all", "all", "custom", 1, self.dump_file,
                       "id type x y z vx vy vz fx fy fz")
        self.calc.thermo_style(
            "custom", "step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms")
        self.calc.thermo_modify("flush", "yes")
        self.calc.thermo(1)
        self.calc.info("all")

    def read_lammps_trj(self, lammps_trj=None) -> list:
        """Method which reads a LAMMPS dump file."""
        if not lammps_trj:
            lammps_trj = os.path.join(self.dump_file)

        trajectory = []
        with open(lammps_trj, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break

                if 'ITEM: TIMESTEP' in line:
                    n_atoms = 0
                    lo = []
                    hi = []
                    tilt = []
                    atom_id = []
                    atom_type = []
                    positions = []
                    velocities = []
                    forces = []

                elif 'ITEM: NUMBER OF ATOMS' in line:
                    line = f.readline()
                    n_atoms = int(line.split()[0])

                elif 'ITEM: BOX BOUNDS' in line:
                    tilt_items = line.split()[3:]
                    for i in range(3):
                        line = f.readline()
                        fields = line.split()
                        lo.append(float(fields[0]))
                        hi.append(float(fields[1]))
                        if len(fields) >= 3:
                            tilt.append(float(fields[2]))

                if 'ITEM: ATOMS' in line:
                    atom_attributes = {}
                    for (i, x) in enumerate(line.split()[2:]):
                        atom_attributes[x] = i
                    for _n in range(n_atoms):
                        line = f.readline()
                        fields = line.split()
                        atom_id.append(int(fields[atom_attributes['id']]))
                        atom_type.append(int(fields[atom_attributes['type']]))
                        positions.append([float(fields[atom_attributes[x]])
                                          for x in ['x', 'y', 'z']])
                        velocities.append([float(fields[atom_attributes[x]])
                                           for x in ['vx', 'vy', 'vz']])
                        forces.append([float(fields[atom_attributes[x]])
                                       for x in ['fx', 'fy', 'fz']])

                    # Re-order items according to their 'id' since running in
                    # parallel can give arbitrary ordering.
                    atom_type = [x for _, x in sorted(zip(atom_id, atom_type))]
                    positions = [x for _, x in sorted(zip(atom_id, positions))]
                    velocities = [x for _, x in sorted(
                        zip(atom_id, velocities))]
                    forces = [x for _, x in sorted(zip(atom_id, forces))]

                    # determine cell tilt (triclinic case!)
                    if len(tilt) >= 3:
                        # for >=lammps-7Jul09 use labels behind "ITEM: BOX BOUNDS"
                        # to assign tilt (vector) elements ...
                        if len(tilt_items) >= 3:
                            xy = tilt[tilt_items.index('xy')]
                            xz = tilt[tilt_items.index('xz')]
                            yz = tilt[tilt_items.index('yz')]
                        # ... otherwise assume default order in 3rd column
                        # (if the latter was present)
                        else:
                            xy = tilt[0]
                            xz = tilt[1]
                            yz = tilt[2]
                    else:
                        xy = xz = yz = 0

                    # Error with LAMMPS traj file output
                    # Temporary work around with this:
                    xhilo = (hi[0] - lo[0]) + xy - xz
                    # This is what it is supposed to be
                    # xhilo = (hi[0] - lo[0]) - xy - xz
                    yhilo = (hi[1] - lo[1]) - yz
                    zhilo = (hi[2] - lo[2])

                    cell = [[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]]

                    # These have been put into the correct order
                    cell_atoms = np.array(cell)
                    type_atoms = np.array(atom_type)

                    if self.atoms:
                        # BEWARE: reconstructing the rotation from the LAMMPS
                        #         output trajectory file fails in case of shrink
                        #         wrapping for a non-periodic direction
                        #      -> hence rather obtain rotation from prism object
                        #         used to generate the LAMMPS input
                        # rotation_lammps2ase = np.dot(
                        #               np.linalg.inv(np.array(cell)), cell_atoms)
                        rotation_lammps2ase = np.linalg.inv(self.prism.R)

                        type_atoms = self.atoms.get_atomic_numbers()
                        positions_atoms = np.dot(
                            positions, rotation_lammps2ase)
                        velocities_atoms = np.dot(
                            velocities, rotation_lammps2ase)
                        forces_atoms = np.dot(forces, rotation_lammps2ase)

                    tmp_atoms = Atoms(type_atoms,
                                      positions=positions_atoms,
                                      cell=cell_atoms)
                    tmp_atoms.set_velocities(velocities_atoms)
                    tmp_atoms.set_pbc(self.atoms.get_pbc())
                    trajectory.append(tmp_atoms)
        return trajectory, forces_atoms


class GULPcalculation:
    """ This class automates and stores some common functions used to set-up a GULP calculation.
    """

    def __init__(self, atoms, label, temp_fd="temp", keywords='conp opti', options=None, calc=None, **kwargs):
        from gulp import GULP

        self.label = label
        self.temp_fd = temp_fd
        self.keywords = keywords
        self.options = ['dump every 1 noover {}.grs'.format(
            os.path.join(temp_fd, label))]

        if options:
            self.options.extend(options)
        if calc:
            self.calc = calc
        else:
            self.calc = GULP(label=label,
                             keywords=self.keywords,
                             options=self.options,
                             **kwargs)

        self.atoms = atoms.copy()
        self.atoms.calc = self.calc
        if 'opti' in self.keywords:
            self.opt = self.calc.get_optimizer(self.atoms)

    def read_intermediate_geometry(self):
        energy = []
        with open(self.label+'.got') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if line.startswith("  Cycle:"):
                    energy.append(float(line.split()[3]))

        traj_file_list = [os.path.join(
            self.temp_fd, self.label+'_{}.grs'.format(i+1)) for i in range(len(energy))]
        trajectory = self.calc.read_trajectory(traj_file_list)

        return energy, trajectory


def rotate_plot_atoms(atoms, radii=1.0, rotation_list=None, interval=40, jsHTML=False):
    """Plots the *atoms* object through the *rotation_list*. If properly structured, 
    this would result in the atoms rotating.

    This function makes use of ASE's plot_atoms function. In Jupyter notebook, the
    *notebook* backend should be used. (%matplotlib notebook)

    Parameters:

    atoms: Atoms instance
    radii: float
    interval: int
    cut_off: float
    rotation_list: list
        A list of rotation coordinates accepted by plot_atoms.
    """

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
    if jsHTML:
        from IPython.display import HTML
        ani = HTML(ani.to_jshtml())
    return ani

def trajectory_convert(ephemeral_out_dir, job_names):
    from PBSJobSubmissionScript import PBS_Submitter

    convert_commands = [["timeout 11h python convert_script_nocopy_fast.py {} {} 32".format(y, os.path.join(ephemeral, x, y+'.lammpstrj'))] for x, y in zip(ephemeral_out_dir, job_names)]
    convert_source_files = [[os.path.join(ephemeral, x, y+".extxyz"),
                             os.path.join(convert_scripts_dir, "convert_script_nocopy_fast.py"),
                             os.path.join(convert_scripts_dir, "minimal_traj_conversion.py")] for x, y in zip(job_names, ephemeral_out_dir)]
    convert_names = [x+"_convert" for x in job_names]
    convert_PBS = PBS_Submitter(job_names=convert_names,
                                job_commands=convert_commands,
                                modules=["anaconda3/personal"],
                                walltime="12:00:00",
                                proc_nodes=1,
                                proc_cpus=32, #mpiprocs x threads = cpus
                                proc_mpiprocs=32, 
                                memory=60,
                                source_files=convert_source_files)

    return convert_PBS

def bond_convert(ephemeral_out_dir, job_names):
    from PBSJobSubmissionScript import PBS_Submitter
    bond_commands = ["timeout 23h python convert_bond.py 8"]
    bond_source_files = [[os.path.join(ephemeral, x, "bonds.tatb"),
                         os.path.join(convert_scripts_dir, "bonds_analysis.py"),
                         os.path.join(convert_scripts_dir, "convert_bond.py")] for x in ephemeral_out_dir]
    bond_names = [x+"_bonds" for x in job_names]
    bond_PBS = PBS_Submitter(job_names=bond_names,
                             job_commands=bond_commands,
                             modules=["anaconda3/personal"],
                             walltime="24:00:00",
                             proc_nodes=1,
                             proc_cpus=8, #mpiprocs x threads = cpus
                             proc_mpiprocs=8, 
                             memory=46,
                             source_files=bond_source_files)
    return bond_PBS

def reaxff_params_generator(sim_box, job_name, input_fd="", write_input=False, always_triclinic=True, dump_period=1, **kwargs):
    from lammpsrun import LAMMPS, write_lammps_data

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

        # Forcefield definition
        "pair_style": "reax/c NULL safezone 16",
        "pair_coeff": ['* * ' + '{0} '.format(potential) + ' '.join(list_of_elements)],

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
    calc = LAMMPS(parameters=reaxff_params, always_triclinic=always_triclinic)
    calc.dump_period = dump_period
    sim_box.set_calculator(calc)
    if write_input:
        write(os.path.join(input_fd, job_name + ".extxyz",),
              sim_box, format="extxyz")
        calc.write_lammps_in(lammps_in=os.path.join(input_fd, "{0}.lammpsin".format(job_name)),
                             lammps_trj="{0}.lammpstrj".format(job_name),
                             lammps_data="{0}.lammpsdata".format(job_name))
    calc.clean()
    return calc


def get_coordination_number(sim_box, atom_index, cut_off=1.0, vdw_cut_off=1.0, cov_cut_off=1.0):
    """Returns the coordination number of atoms of index *atom_index*
    in *sim_box*.

    This function makes use of ASE's vdw radius and covalent radius data, and
    accepts individual cut-offs of such radius. These are then passed to the neighbor
    list function to get the neighbour list.

    Parameters:

    sim_box: Atoms | list of Atoms
    atom_index: list
    cut_off: float
    vdw_cut_off: float
    cov_cut_off: float
    """

    # pylint: disable=unused-variable
    i, j = nl_cutoff_cov_vdw(
        sim_box, cut_off, vdw_cut_off, cov_cut_off)
    # pylint: enable=unused-variable
    indices, coord = np.unique(i, return_counts=True)
    coord_numbers = dict(zip(indices, coord))
    return [coord_numbers[x] for x in atom_index]


def identify_connection(i, j, atom_index, connections=None):
    """Given a neighbor_list i, j, identifies all indices connected to atom_index recursively"""
    if not connections:
        connections = [atom_index]

    new_connections = j[i == atom_index]
    to_check = [x for x in new_connections if x not in connections]
    connections.extend(to_check)

    for n in to_check:
        identify_connection(i, j, n, connections)

    return connections


def nl_cutoff_cov_vdw(sim_box, cut_off, vdw_cut_off=1.0, cov_cut_off=1.0):
    overlap_vdwr_sphere = np.array([vdw_radii[atomic_numbers[x]]
                                    for x in sim_box.get_chemical_symbols()])
    overlap_covr_sphere = np.array([covalent_radii[atomic_numbers[x]]
                                    for x in sim_box.get_chemical_symbols()])
    overlap_sphere = cut_off * \
        (vdw_cut_off * overlap_vdwr_sphere +
         cov_cut_off * overlap_covr_sphere)/2

    # for atoms without vdwr
    overlap_sphere[np.isnan(overlap_vdwr_sphere)
                   ] = cut_off * cov_cut_off * overlap_covr_sphere[np.isnan(overlap_vdwr_sphere)]

    i, j = neighbor_list('ij', sim_box, overlap_sphere, self_interaction=False)
    return i, j


def replace_molecule(sim_box, source_index, new_mol, cut_off=1.0, seed=None):
    'Replaces a molecule attached to an atom at source_index of the sim_box and replace it by new_mol'
    new_box = sim_box.copy()
    old_pbc = new_box.get_pbc()

    i, j = nl_cutoff_cov_vdw(new_box, cut_off)

    if isinstance(source_index, int):
        mol_to_delete_indices = [identify_connection(i, j, atom_index=source_index)]
    else: # assuming source_index is list-like
        mol_to_delete_indices = [identify_connection(i, j, atom_index=x) for x in source_index]

    mol_to_delete_COM = [new_box[x].get_center_of_mass() for x in mol_to_delete_indices]

    for com in mol_to_delete_COM:
        add_mol = new_mol.copy()
        add_mol.translate(com)
        new_box.extend(add_mol)
    
    flat_list = [item for sublist in mol_to_delete_indices for item in sublist]
    del new_box[flat_list]

    new_box.set_pbc(old_pbc)

    return new_box


def create_water_region(cell):
    """Creates a region of water for the cell specified. The density is assumed to be that
    of normal water at 1atm and 300K.

    Internally, this first creates a grid of Ne atoms, and then rattle them very slightly
    (std = 1/10 of a H2O molecule size). The Ne atoms are then replaced with H2O molecules
    centered on Ne. Be _very_ careful that the atoms are not out of bounds. It can happen
    after some rotations. Recommend to put in a slightly smaller cell to give some margins.

    Parameters:

    cell: list | numpy array

    Returns:

    H2O_bulk: Atoms | list of Atom
    """

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
