from ase import Atoms, units
from ase.visualize import view
from ase.visualize.plot import plot_atoms
from ase.io.trajectory import Trajectory
from ase.io import write, read
from ase.calculators.lammpsrun import LAMMPS, Prism

CO2_10_aq_calc = LAMMPS()
CO2_10_aq = read("CO2_solvation.extxyz")
CO2_10_aq.set_calculator(CO2_10_aq_calc)
CO2_10_aq_calc.atoms = CO2_10_aq
CO2_10_aq_calc.prism = Prism(CO2_10_aq.get_cell())

CO2_10_aq_calc.trajectory_out = Trajectory("CO2_solvation_300K.traj", 'w')
CO2_10_aq_calc.read_lammps_trj(lammps_trj = "CO2_solvation_300K.lammpstrj")
CO2_10_aq_calc.trajectory_out.close()