import sys

import mkl
from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory

from minimal_traj_conversion import LAMMPS, Prism

file_name = sys.argv[1]
lammps_traj_file = sys.argv[2]
n_cpus = 8
if len(sys.argv) > 3:
    n_cpus = int(sys.argv[3])

print("Original Max Threads:", mkl.get_max_threads())
mkl.set_num_threads(n_cpus)
print("New Max Threads:", mkl.get_max_threads())

mol_file = file_name + ".extxyz"
ase_traj_file = file_name + ".traj"

calc = LAMMPS()
mol = read(mol_file)
mol.set_calculator(calc)

print("Writing ASE trajectory to ", ase_traj_file)
calc.trajectory_out = Trajectory(ase_traj_file, 'w')

print("Reading LAMMPS Trajectory from", lammps_traj_file)
calc.read_lammps_trj(lammps_trj=lammps_traj_file, n_cpus=n_cpus)
calc.trajectory_out.close()
