import sys

import mkl
from ase import Atoms
from ase.io import read
from ase.io.trajectory import Trajectory

from bonds_analysis import bonds_analysis

n_cpus = 8
if len(sys.argv) > 1:
    n_cpus = int(sys.argv[1])

print("Original Max Threads:", mkl.get_max_threads())
mkl.set_num_threads(n_cpus)
print("New Max Threads:", mkl.get_max_threads())

bond_in = "bonds.tatb"
bond_out = "bonds"

ba = bonds_analysis()

ba.read_reaxff_bond(bond_in=bond_in, bond_out=bond_out, n_cpus=10)
