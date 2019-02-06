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
from warnings import warn
from itertools import product, combinations_with_replacement, combinations


class reaxFF_data:
    # Courtesy of Christopher Sewell, aiida-gulp
    _paramkeys_gulp = ['Overcoordination 1', 'Overcoordination 2', 'Valency angle conjugation 1',
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

    _paramkeys = ['Overcoordination parameter', 'Overcoordination parameter', 'Valency angle conjugation parameter',
                  'Triple bond stabilisation parameter', 'Triple bond stabilisation parameter', 'C2-correction',
                  'Undercoordination parameter', 'Triple bond stabilisation parameter', 'Undercoordination parameter',
                  'Undercoordination parameter', 'Triple bond stabilization energy', 'Lower Taper-radius', 'Upper Taper-radius',
                  'Not used', 'Valency undercoordination', 'Valency angle/lone pair parameter', 'Valency angle',
                  'Valency angle parameter', 'Not used', 'Double bond/angle parameter', 'Double bond/angle parameter: overcoord',
                  'Double bond/angle parameter: overcoord', 'Not used', 'Torsion/BO parameter', 'Torsion overcoordination',
                  'Torsion overcoordination', 'Conjugation 0 (not used)', 'Conjugation', 'vdWaals shielding',
                  'Cutoff for bond order (*100)', 'Valency angle conjugation parameter', 'Overcoordination parameter',
                  'Overcoordination parameter', 'Valency/lone pair parameter', 'Not used', 'Not used', 'Molecular energy (not used)',
                  'Molecular energy (not used)', 'Valency angle conjugation parameter']

    _speckeys_gulp = ['reaxff1_radii1', 'reaxff1_valence1', 'mass', 'reaxff1_morse3', 'reaxff1_morse2',
                      'reaxff_gamma', 'reaxff1_radii2', 'reaxff1_valence3', 'reaxff1_morse1', 'reaxff1_morse4',
                      'reaxff1_valence4', 'reaxff1_under', 'dummy1', 'reaxff_chi', 'reaxff_mu', 'dummy2',
                      'reaxff1_radii3', 'reaxff1_lonepair2', 'dummy3', 'reaxff1_over2', 'reaxff1_over1',
                      'reaxff1_over3', 'dummy4', 'dummy5', 'reaxff1_over4', 'reaxff1_angle1', 'dummy11',
                      'reaxff1_valence2', 'reaxff1_angle2', 'dummy6', 'dummy7', 'dummy8']

    _speckeys = ['cov.r', 'valency', 'a.m', 'Rvdw', 'Evdw', 'gammaEEM', 'cov.r2', '#el', 'alfa',
                 'gammavdW', 'valency', 'Eunder', 'n.u.', 'chiEEM', 'etaEEM', 'n.u.', 'cov.r3', 'Elp',
                 'Heat inc.', '13BO1', '13BO2', '13BO3', 'n.u.', 'n.u.', 'ov/un', 'val1', 'n.u.', 'val3',
                 'vval4', 'n.u.', 'n.u.', 'n.u.']

    _bondkeys_gulp = ['reaxff2_bond1', 'reaxff2_bond2', 'reaxff2_bond3', 'reaxff2_bond4', 'reaxff2_bo5',
                      'reaxff2_bo7', 'reaxff2_bo6', 'reaxff2_over', 'reaxff2_bond5', 'reaxff2_bo3',
                      'reaxff2_bo4', 'dummy1', 'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo8', 'reaxff2_bo9']

    _bondkeys = ['Edis1', 'LPpen', 'n.u.', 'pbe1', 'pbo5', '13corr', 'pbo6', 'kov',
                 'pbe2', 'pbo3', 'pbo4', 'Etrip', 'pbo1', 'pbo2', 'ovcorr', 'n.u.']

    _odkeys_gulp = ['reaxff2_morse1', 'reaxff2_morse3', 'reaxff2_morse2',
                    'reaxff2_morse4', 'reaxff2_morse5', 'reaxff2_morse6']

    _odkeys = ['Ediss', 'Ro', 'gamma', 'rsigma', 'rpi', 'rpi2 ']

    _anglekeys_gulp = ['reaxff3_angle1', 'reaxff3_angle2', 'reaxff3_angle3',
                       'reaxff3_conj', 'reaxff3_angle5', 'reaxff3_penalty', 'reaxff3_angle4']

    _angleskeys = ['Theta,o', 'ka', 'kb', 'pv1', 'pv2', 'kpenal', 'pv3']

    _torkeys_gulp = ['reaxff4_torsion1', 'reaxff4_torsion2', 'reaxff4_torsion3',
                     'reaxff4_torsion4', 'reaxff4_torsion5', 'dummy1', 'dummy2']

    _torkeys = ['V1', 'V2', 'V3', 'V2(BO)', 'vconj', 'n.u.', 'n.u.']

    _hbkeys_gulp = ['reaxff3_hbond1', 'reaxff3_hbond2',
                    'reaxff3_hbond3', 'reaxff3_hbond4']

    _hbkeys = ['Rhb', 'Dehb', 'vhb1', 'vhb2']

    def __init__(self, species=None, params=None):
        self.species = species
        self.description = None
        self.remark = None
        self.params = params

        if not params and species:
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

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        """Returns a deep copy of itself."""
        params_copy = self.params.copy()
        for i, k in params_copy.items():
            params_copy[i] = k.copy(deep=True)
        return reaxFF_data(species=self.species.copy(), params=params_copy)

    def clean_params(self):
        for k in self.params.values():
            k.dropna(inplace=True)

    def read_lammps(self, file_name, drop_new_species=True):
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

                if drop_new_species and current_atom not in self.species:
                    continue

                # Check for placeholder atom
                if current_atom == 'X':
                    self.params['species']['X'] = None
                    species2id['X'] = 0
                else:
                    if current_atom not in self.species:
                        warn('{} is not found in initilizing species. Added anyway.'.format(
                            current_atom))

                    species2id[current_atom] = atom_no

                if 'X' not in species2id.keys():
                    species2id['X'] = 0

                self.params['species'].loc[current_atom] = [
                    float(x) for x in params[1:]]

        def _read_connectivity_parameters(rf, connectivity: str, param_lines: int, number_atoms: int):
            # Sometimes the h-bond section is missing
            init_line = rf.readline()
            if not init_line:
                return

            n_params = int(init_line.split()[0])

            # Discard *param_lines-1* lines
            for _i in range(param_lines-1):
                rf.readline()

            for _i in range(n_params):
                params = []

                # Each set of atom parameters spans *param_lines* lines
                for _i in range(param_lines):
                    params.extend(rf.readline().split())

                current_ids = [int(x) for x in params[0:number_atoms]]

                if drop_new_species and any(x not in species2id.values() for x in current_ids):
                    continue

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
                    warn('One of the atoms in {} is not found in initilizing species. Added anyway.'.format(
                        '-'.join(current_atoms)))
                    self.params[connectivity].loc['-'.join(current_atoms)] = [
                        float(x) for x in params[number_atoms:]]

        with open(file_name) as rf:
            self.description = rf.readline()
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
        general_df = pd.DataFrame(columns=self._paramkeys, index=['default'])

        species_df = pd.DataFrame(columns=self._speckeys)
        species_df['symbols'] = self.species
        species_df.set_index('symbols', inplace=True)

        bonds = ['-'.join(x)
                 for x in combinations_with_replacement(self.species, r=2)]
        bonds_df = pd.DataFrame(columns=self._bondkeys)
        bonds_df['symbols'] = bonds
        bonds_df.set_index('symbols', inplace=True)

        off_diags = ['-'.join(x) for x in combinations(self.species, r=2)]
        off_diags_df = pd.DataFrame(columns=self._odkeys)
        off_diags_df['symbols'] = off_diags
        off_diags_df.set_index('symbols', inplace=True)

        angles = []
        for angle in product(self.species, repeat=3):
            if angle[::-1] not in angles:
                angles.append('-'.join(angle))
        angles_df = pd.DataFrame(columns=self._anglekeys)
        angles_df['symbols'] = angles
        angles_df.set_index('symbols', inplace=True)

        torsions = []
        for torsion in product(self.species, repeat=4):
            if torsion[::-1] not in torsions:
                torsions.append('-'.join(torsion))
        torsions_df = pd.DataFrame(columns=self._torkeys)
        torsions_df['symbols'] = torsions
        torsions_df.set_index('symbols', inplace=True)

        # Assuming hydrogen bonding between only these atoms.
        # Can be extended in the future with input params.
        h_bond_acceptors = ['O', 'N', 'Cl', 'F', 'S', 'Br']
        h_bonds = []
        for h_bond in product(self.species, repeat=3):
            # The middle atom must be hydrogen
            if h_bond[1] == 'H' and (h_bond[0] in h_bond_acceptors or h_bond[2] in h_bond_acceptors):
                if h_bond[::-1] not in h_bonds:
                    h_bonds.append('-'.join(h_bond))
        h_bonds_df = pd.DataFrame(columns=self._hbkeys)
        h_bonds_df['symbols'] = h_bonds
        h_bonds_df.set_index('symbols', inplace=True)

        params = {'general': general_df,
                  'species': species_df,
                  'bonds': bonds_df,
                  'off_diagonal': off_diags_df,
                  'angles': angles_df,
                  'torsions': torsions_df,
                  'h_bonds': h_bonds_df}

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
