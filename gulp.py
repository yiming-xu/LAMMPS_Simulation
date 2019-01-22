# gulp.py customised for the MgO TE labs
"""This module defines an ASE interface to GULP.

Written by:

Andy Cuko <andi.cuko@upmc.fr>
Antoni Macia <tonimacia@gmail.com>

EXPORT ASE_GULP_COMMAND="/path/to/gulp < PREFIX.gin > PREFIX.got"

Keywords
Options

"""
import os
import re

import numpy as np
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.units import Ang, eV


class GULPOptimizer:
    def __init__(self, atoms, calc):
        self.atoms = atoms
        self.calc = calc

    def todict(self):
        return {'type': 'optimization',
                'optimizer': 'GULPOptimizer'}

    def run(self, fmax=None, steps=None, **gulp_kwargs):
        if fmax is not None:
            gulp_kwargs['gmax'] = fmax
        if steps is not None:
            gulp_kwargs['maxcyc'] = steps

        self.calc.set(**gulp_kwargs)
        self.atoms.calc = self.calc
        self.atoms.get_potential_energy()
        self.atoms.positions[:] = self.calc.get_atoms().positions


class GULP(FileIOCalculator):
    implemented_properties = ['energy', 'forces']
    command = 'gulp < PREFIX.gin > PREFIX.got'
    default_parameters = dict(
        keywords='conp gradients',
        options=[],
        shel=[],
        library="ffsioh.lib",
        conditions=None,
        symmetry=None
    )

    def get_optimizer(self, atoms):
        gulp_keywords = self.parameters.keywords.split()
        if 'opti' not in gulp_keywords:
            raise ValueError('Can only create optimizer from GULP calculator '
                             'with "opti" keyword.  Current keywords: {}'
                             .format(gulp_keywords))

        opt = GULPOptimizer(atoms, self)
        return opt

#conditions=[['O', 'default', 'O1'], ['O', 'O2', 'H', '<', '1.6']]

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='gulp', atoms=None, optimized=None,
                 Gnorm=1000.0, steps=1000, conditions=None, **kwargs):
        """Construct GULP-calculator object."""
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)
        self.optimized = optimized
        self.Gnorm = Gnorm
        self.steps = steps
        self.conditions = conditions
        self.library_check()
        self.atom_types = []

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters

        # Create a primitive cell for calculating symmetry
        if p.symmetry:
            assert self.atoms.info['unit_cell'] == 'conventional'
            sg = self.atoms.info['spacegroup']

            from ase.build import cut
            prim_cell = sg.scaled_primitive_cell
            prim_atoms = cut(
                self.atoms, a=prim_cell[0], b=prim_cell[1], c=prim_cell[2])

        # Build string to hold .gin input file:
        s = p.keywords
        s += '\ntitle\nASE calculation\nend\n\n'

        if all(self.atoms.pbc):
            if p.symmetry:
                coords = prim_atoms.get_scaled_positions()
                cell_params = prim_atoms.get_cell_lengths_and_angles()
            else:
                coords = self.atoms.get_scaled_positions()
                cell_params = self.atoms.get_cell_lengths_and_angles()

            s += 'cell\n{0} {1} {2} {3} {4} {5}\n'.format(
                *(np.around(cell_params, 5)))
            s += 'frac\n'

        if self.conditions is not None:
            c = self.conditions
            labels = c.get_atoms_labels()
            self.atom_types = c.get_atom_types()
        else:
            if p.symmetry:
                labels = prim_atoms.get_chemical_symbols()
            else:
                labels = self.atoms.get_chemical_symbols()

        for xyz, symbol in zip(coords, labels):

            s += ' {0:2} core'.format(symbol)

            def print_xyz(xyz) -> str:
                # so we need to convert coords to fraction notation for precision purposes.
                from fractions import Fraction

                xyz_string = ''
                for i in xyz:
                    # if it's 0 it can stay as 0 no worries
                    if i == 0:
                        xyz_string += '  {:8}'.format(np.around(i, 5))
                    else:
                        # 50 seems a good number. would be adjustable
                        new_frac = Fraction(i).limit_denominator(50)
                        # ensuring a very small difference in actual value
                        # note default symprec=1e-05
                        if abs(new_frac.numerator/new_frac.denominator - i)/i < 1e-6:
                            xyz_string += '       {}'.format(new_frac)
                        else:
                            xyz_string += '  {:8}'.format(np.around(i, 5))
                return xyz_string

            s += print_xyz(xyz) + '\n'

            if symbol in p.shel:
                s += ' {0:2} shel'.format(symbol)
                s += print_xyz(xyz) + '\n'

        if p.symmetry:
            s += '\nspace\n'
            s += '%s\n' % p.symmetry
        s += '\nlibrary {0}\n'.format(p.library)
        if p.options:
            for t in p.options:
                s += '%s\n' % t
        with open(self.prefix + '.gin', 'w') as f:
            f.write(s)

    def read_results(self):
        FileIOCalculator.read(self, self.label)
        if not os.path.isfile(self.label + '.got'):
            raise ReadError

        with open(self.label + '.got') as f:
            lines = f.readlines()

        cycles = -1
        self.optimized = None
        for i, line in enumerate(lines):
            if line.startswith("  Final energy ="):
                energy = float(line.split()[-2])
                self.results['energy'] = energy
                self.results['free_energy'] = energy

            elif line.find('Optimisation achieved') != -1:
                self.optimized = True

            elif line.find('Final Gnorm') != -1:
                self.Gnorm = float(line.split()[-1])

            elif line.find('Cycle:') != -1:
                cycles += 1

            elif line.find('Final Cartesian derivatives') != -1:
                s = i + 5
                forces = []
                while(True):
                    s = s + 1
                    if lines[s].find("------------") != -1:
                        break
                    if lines[s].find(" s ") != -1:
                        continue
                    g = lines[s].split()[3:6]
                    G = [-float(x) * eV / Ang for x in g]
                    forces.append(G)
                forces = np.array(forces)
                self.results['forces'] = forces

            elif line.find('Final cartesian coordinates of atoms') != -1:
                s = i + 5
                positions = []
                while True:
                    s = s + 1
                    if lines[s].find("------------") != -1:
                        break
                    if lines[s].find(" s ") != -1:
                        continue
                    xyz = lines[s].split()[3:6]
                    XYZ = [float(x) * Ang for x in xyz]
                    positions.append(XYZ)
                positions = np.array(positions)
                self.atoms.set_positions(positions)

        self.steps = cycles

    def get_opt_state(self):
        return self.optimized

    def get_opt_steps(self):
        return self.steps

    def get_Gnorm(self):
        return self.Gnorm

    def library_check(self):
        if self.parameters['library'] is not None:
            if 'GULP_LIB' not in os.environ:
                raise RuntimeError("Be sure to have set correctly $GULP_LIB "
                                   "or to have the force field library.")


class Conditions:
    """Atomic labels for the GULP calculator.

    This class manages an array similar to
    atoms.get_chemical_symbols() via get_atoms_labels() method, but
    with atomic labels in stead of atomic symbols.  This is useful
    when you need to use calculators like GULP or lammps that use
    force fields. Some force fields can have different atom type for
    the same element.  In this class you can create a set_rule()
    function that assigns labels according to structural criteria."""

    def __init__(self, atoms):
        self.atoms = atoms
        self.atoms_symbols = atoms.get_chemical_symbols()
        self.atoms_labels = atoms.get_chemical_symbols()
        self.atom_types = []

    def min_distance_rule(self, sym1, sym2,
                          ifcloselabel1=None, ifcloselabel2=None,
                          elselabel1=None, max_distance=3.0):
        """Find pairs of atoms to label based on proximity.

        This is for, e.g., the ffsioh or catlow force field, where we
        would like to identify those O atoms that are close to H
        atoms.  For each H atoms, we must specially label one O atom.

        This function is a rule that allows to define atom labels (like O1,
        O2, O_H etc..)  starting from element symbols of an Atoms
        object that a force field can use and according to distance
        parameters.

        Example:
        atoms = read('some_xyz_format.xyz')
        a = Conditions(atoms)
        a.set_min_distance_rule('O', 'H', ifcloselabel1='O2',
                                ifcloselabel2='H', elselabel1='O1')
        new_atoms_labels = a.get_atom_labels()

        In the example oxygens O are going to be labeled as O2 if they
        are close to a hydrogen atom othewise are labeled O1.

        """

        if ifcloselabel1 is None:
            ifcloselabel1 = sym1
        if ifcloselabel2 is None:
            ifcloselabel2 = sym2
        if elselabel1 is None:
            elselabel1 = sym1

        # self.atom_types is a list of element types  used instead of element
        # symbols in orger to track the changes made. Take care of this because
        # is very important.. gulp_read function that parse the output
        # has to know which atom_type it has to associate with which
        # atom_symbol
        #
        # Example: [['O','O1','O2'],['H', 'H_C', 'H_O']]
        # this beacuse Atoms oject accept only atoms symbols
        self.atom_types.append([sym1, ifcloselabel1, elselabel1])
        self.atom_types.append([sym2, ifcloselabel2])

        dist_mat = self.atoms.get_all_distances()
        index_assigned_sym1 = []
        index_assigned_sym2 = []

        for i in range(len(self.atoms_symbols)):
            if self.atoms_symbols[i] == sym2:
                dist_12 = 1000
                index_assigned_sym2.append(i)
                for t in range(len(self.atoms_symbols)):
                    if (self.atoms_symbols[t] == sym1
                        and dist_mat[i, t] < dist_12
                            and t not in index_assigned_sym1):
                        dist_12 = dist_mat[i, t]
                        closest_sym1_index = t
                index_assigned_sym1.append(closest_sym1_index)

        for i1, i2 in zip(index_assigned_sym1, index_assigned_sym2):
            if dist_mat[i1, i2] > max_distance:
                raise ValueError('Cannot unambiguously apply minimum-distance '
                                 'rule because pairings are not obvious.  '
                                 'If you wish to ignore this, then increase '
                                 'max_distance.')

        for s in range(len(self.atoms_symbols)):
            if s in index_assigned_sym1:
                self.atoms_labels[s] = ifcloselabel1
            elif s not in index_assigned_sym1 and self.atoms_symbols[s] == sym1:
                self.atoms_labels[s] = elselabel1
            elif s in index_assigned_sym2:
                self.atoms_labels[s] = ifcloselabel2

    def get_atom_types(self):
        return self.atom_types

    def get_atoms_labels(self):
        labels = np.array(self.atoms_labels)
        return labels
