##################################################
# This script stores the relevant forcefield data
# required by reaxFF, primarily in the LAMMPS
# format.
##################################################
# MIT License
##################################################
# Author: Yiming Xu
# Copyright:
# Credits:
# License:
# Version:
# Maintainer:
# Email:
# Status:
##################################################
import pandas as pd
from itertools import product, combinations_with_replacement, combinations


class reaxFF_data:
    # Courtesy of Christopher Sewell, aiida-gulp
    _paramkeys = ['Overcoordination 1', 'Overcoordination 2', 'Valency angle conjugation 1',
                  'Triple bond stabilisation 1', 'Triple bond stabilisation 2', 'C2-correction',
                  'Undercoordination 1', 'Triple bond stabilisation', 'Undercoordination 2',
                  'Undercoordination 3', 'Triple bond stabilization energy', 'Lower Taper-radius',
                  'Upper Taper-radius', 'Not used 1', 'Valency undercoordination', 'Valency angle/lone pair',
                  'Valency angle 1', 'Valency angle 2', 'Not used 2', 'Double bond/angle',
                  'Double bond/angle: overcoord 1', 'Double bond/angle: overcoord 2', 'Not used 3',
                  'Torsion/BO', 'Torsion overcoordination 1', 'Torsion overcoordination 2', 'Not used 4',
                  'Conjugation', 'vdWaals shielding', 'bond order cutoff', 'Valency angle conjugation 2',
                  'Valency overcoordination 1', 'Valency overcoordination 2', 'Valency/lone pair',
                  'Not used 5', 'Not used 6', 'Not used 7', 'Not used 8', 'Valency angle conjugation 3']

    _speckeys = ['reaxff1_radii1', 'reaxff1_valence1', 'mass', 'reaxff1_morse3', 'reaxff1_morse2',
                 'reaxff_gamma', 'reaxff1_radii2', 'reaxff1_valence3', 'reaxff1_morse1', 'reaxff1_morse4',
                 'reaxff1_valence4', 'reaxff1_under', 'dummy1', 'reaxff_chi', 'reaxff_mu', 'dummy2',
                 'reaxff1_radii3', 'reaxff1_lonepair2', 'dummy3', 'reaxff1_over2', 'reaxff1_over1',
                 'reaxff1_over3', 'dummy4', 'dummy5', 'reaxff1_over4', 'reaxff1_angle1', 'dummy11',
                 'reaxff1_valence2', 'reaxff1_angle2', 'dummy6', 'dummy7', 'dummy8']

    _bondkeys = ['reaxff2_bond1', 'reaxff2_bond2', 'reaxff2_bond3', 'reaxff2_bond4','reaxff2_bo5',
                 'reaxff2_bo7', 'reaxff2_bo6', 'reaxff2_over', 'reaxff2_bond5', 'reaxff2_bo3',
                 'reaxff2_bo4', 'dummy1', 'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo8', 'reaxff2_bo9']

    _odkeys = ['reaxff2_morse1', 'reaxff2_morse3', 'reaxff2_morse2',
               'reaxff2_morse4', 'reaxff2_morse5', 'reaxff2_morse6']

    _anglekeys = ['reaxff3_angle1', 'reaxff3_angle2', 'reaxff3_angle3',
                  'reaxff3_conj', 'reaxff3_angle5', 'reaxff3_penalty', 'reaxff3_angle4']

    _torkeys = ['reaxff4_torsion1', 'reaxff4_torsion2', 'reaxff4_torsion3',
                'reaxff4_torsion4', 'reaxff4_torsion5', 'dummy1', 'dummy2']

    _hbkeys = ['reaxff3_hbond1', 'reaxff3_hbond2',
               'reaxff3_hbond3', 'reaxff3_hbond4']

    def __init__(self, species=None):
        self.species = species
        if species:
            self.params = self._gen_empty_params()

    def __repr__(self):
        outstr = ''
        for i, k in self.params.items():
            outstr += '**********************\n'
            outstr += i
            outstr += '\n**********************\n'
            outstr += k.__repr__()
            outstr += '\n'
        return outstr

    def read_lammps(self, file_name):
        # Reads into self.params data from a standard LAMMPS reaxFF input file
        def _read_general_parameters(rf):
            n_params = int(rf.readline().split()[0])
            assert n_params == 39

            # We expect 39 parameters
            params = []
            for _i in range(n_params):
                params.append(float(rf.readline().split()[0]))

            self.params['general'].loc['default'] = params

        def _read_species_parameters(rf):
            n_params = int(rf.readline().split()[0])

            # Discard 3 lines
            for _i in range(3):
                rf.readline()

            for atom_no in range(1, n_params+1):
                params = []

                # Each set of atom parameters spans 4 lines
                for _i in range(4):
                    params.extend(rf.readline().split())

                current_atom = params[0]

                # Check for placeholder atom
                if current_atom == 'X':
                    self.params['species']['X'] = None
                    species2id['X'] = 0
                else:
                    species2id[current_atom] = atom_no

                if 'X' not in species2id.keys():
                    species2id['X'] = 0
                self.params['species'].loc[current_atom] = [
                    float(x) for x in params[1:]]

        def _read_connectivity_parameters(rf, connectivity: str, param_lines: int, number_atoms: int):
            n_params = int(rf.readline().split()[0])

            # Discard *param_lines-1* lines
            for _i in range(param_lines-1):
                rf.readline()

            for _i in range(n_params):
                params = []

                # Each set of atom parameters spans *param_lines* lines
                for _i in range(param_lines):
                    params.extend(rf.readline().split())

                current_ids = [int(x) for x in params[0:number_atoms]]
                current_atoms = [species2id.inverse[x][0] for x in current_ids]

                if 'X' in current_atoms:
                    self.params[connectivity].loc['-'.join(current_atoms)] = [
                        float(x) for x in params[number_atoms:]]
                elif '-'.join(current_atoms) in self.params[connectivity].index:
                    self.params[connectivity].loc['-'.join(current_atoms)] = [
                        float(x) for x in params[number_atoms:]]
                elif '-'.join(reversed(current_atoms)) in self.params[connectivity].index:
                    self.params[connectivity].loc['-'.join(reversed(current_atoms))] = [
                        float(x) for x in params[number_atoms:]]
                else:
                    raise Exception('This should never happen.')

        with open(file_name) as rf:
            self.params['description'] = rf.readline()
            species2id = bidict({sp: None for sp in self.species})

            _read_general_parameters(rf)
            _read_species_parameters(rf)
            lammps_params_format = [{'connectivity': 'bonds', 'param_lines': 2, 'number_atoms': 2},
                                    {'connectivity': 'off_diagonal',
                                        'param_lines': 1, 'number_atoms': 2},
                                    {'connectivity': 'angles',
                                        'param_lines': 1, 'number_atoms': 3},
                                    {'connectivity': 'torsions',
                                        'param_lines': 1, 'number_atoms': 4},
                                    {'connectivity': 'h_bonds', 'param_lines': 1, 'number_atoms': 3}]

            for par in lammps_params_format:
                _read_connectivity_parameters(rf, **par)

    def _gen_empty_params(self) -> pd.DataFrame:
        """Generates an empty parameter dict with all possible parameters.
        """
        bonds = ['-'.join(x)
                 for x in combinations_with_replacement(self.species, r=2)]
        off_diags = ['-'.join(x) for x in combinations(self.species, r=2)]

        angles = []
        for angle in product(self.species, repeat=3):
            if angle[::-1] not in angles:
                angles.append('-'.join(angle))

        torsions = []
        for torsion in product(self.species, repeat=4):
            if torsion[::-1] not in torsions:
                torsions.append('-'.join(torsion))

        # Assuming hydrogen bonding between only H and O. Can be extended in the future with input params.
        h_bonds = ['O-H-O']

        params = {'description': None,
                  'general': pd.DataFrame(columns=self._paramkeys),
                  'species': pd.DataFrame(columns=self._speckeys,
                                          index=self.species),
                  'bonds': pd.DataFrame(columns=self._bondkeys,
                                        index=bonds),
                  'off_diagonal': pd.DataFrame(columns=self._odkeys,
                                               index=off_diags),
                  'angles': pd.DataFrame(columns=self._anglekeys,
                                         index=angles),
                  'torsions': pd.DataFrame(columns=self._torkeys,
                                           index=torsions),
                  'h_bonds': pd.DataFrame(columns=self._hbkeys,
                                          index=h_bonds),
                  'remark': None}

        return params


class bidict(dict):
    """ A two way mapping function, where the forward keys are 
    unique, but the inverse gives a list of keys with the same
    value.
    """

    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        if key in self:
            self.inverse[self[key]].remove(key)
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
