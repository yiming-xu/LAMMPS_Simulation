# Author:
#   Yiming Xu, yiming.xu15@imperial.ac.uk
# 
# This is a list of scripts that are somewhat useful
from ase import Atoms, units
from ase.build import molecule
from numpy.random import rand

def create_water_region(cell):
    'Creates a region of water with approximate density at 300K'

    aq_cell = cell
    H2O_volume = (1e+27)/(1000/18 * units.mol)
    Atoms_grid = Atoms('Ne', pbc=True, cell=[H2O_volume**(1/3)]*3)

    Atoms_grid = Atoms_grid.repeat(tuple([int(x/(H2O_volume**(1/3))) for x in aq_cell]))

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
        new_molecule.euler_rotate(rand()*360, rand()*360, rand()*360, center="COM")
        new_molecule.translate(new_pos)

        H2O_bulk += new_molecule.copy()
    H2O_bulk.center()
    
    return H2O_bulk