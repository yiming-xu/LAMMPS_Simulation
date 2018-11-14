# Author:
#   Yiming Xu, yiming.xu15@imperial.ac.uk
# 
# This is a list of scripts that are somewhat useful
from ase import Atoms, units
from ase.build import molecule
import numpy as np
from numpy.random import rand
from ase.data import *
from ase.neighborlist import neighbor_list

def identify_connection(i, j, num, connections = None):
    'Given a neighbor_list i, j, identifies all indices connected to num'
    if not connections:
        connections = [num]
        
    new_connections = j[i == num]
    to_check = [x for x in new_connections if x not in connections]
    connections.extend(to_check)
    
    for n in to_check:
        identify_connection(i, j, n, connections)
        
    return connections
    
def replace_molecule(sim_box, source_index, new_mol, cut_off=1.0, seed=None):
    'Replaces a molecule attached to an atom at source_index of the sim_box and replace it by new_mol'
    new_box = sim_box.copy()
    old_pbc = new_box.get_pbc()
    
    overlap_vdwr_sphere = [vdw_radii[atomic_numbers[x]] for x in new_box.get_chemical_symbols()]
    overlap_covr_sphere = [covalent_radii[atomic_numbers[x]] for x in new_box.get_chemical_symbols()]
    overlap_sphere = cut_off*(np.array(overlap_vdwr_sphere)+np.array(overlap_covr_sphere))/2
    
    i,j = neighbor_list('ij', new_box, overlap_sphere, self_interaction=False)
    
    mol_to_delete_indices = identify_connection(i, j, num = source_index)
    mol_to_delete_COM = new_box[mol_to_delete_indices].get_center_of_mass()
    
    #print(mol_to_delete_indices)
    
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