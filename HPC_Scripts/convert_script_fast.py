from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.io import write, read
from lammpsrun import LAMMPS, Prism
import sys
import mkl

print("Original Max Threads:", mkl.get_max_threads())
mkl.set_num_threads(8)
print("New Max Threads:", mkl.get_max_threads())

file_name = sys.argv[1]

mol_file = file_name + ".extxyz"
lammps_traj_file = file_name + ".lammpstrj"
ase_traj_file = file_name + ".traj"

calc = LAMMPS()
mol = read(mol_file)
mol.set_calculator(calc)
calc.atoms = mol
calc.prism = Prism(mol.get_cell())

print("Writing ASE trajectory to ", ase_traj_file)
calc.trajectory_out = Trajectory(ase_traj_file, 'w')

print("Reading LAMMPS Trajectory from", lammps_traj_file)
calc.read_lammps_trj(lammps_trj = lammps_traj_file)
calc.trajectory_out.close()