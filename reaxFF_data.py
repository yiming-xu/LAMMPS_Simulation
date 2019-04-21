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

    _paramkeys = ['Overcoordination parameter 1', 'Overcoordination parameter 2', 'Valency angle conjugation parameter 1',
                  'Triple bond stabilisation parameter 1', 'Triple bond stabilisation parameter 2', 'C2-correction',
                  'Undercoordination parameter 1', 'Triple bond stabilisation parameter 3', 'Undercoordination parameter 2',
                  'Undercoordination parameter 3', 'Triple bond stabilization energy', 'Lower Taper-radius', 'Upper Taper-radius',
                  'Not used1', 'Valency undercoordination', 'Valency angle/lone pair parameter', 'Valency angle',
                  'Valency angle parameter', 'Not used 2', 'Double bond/angle parameter', 'Double bond/angle parameter: overcoord 1',
                  'Double bond/angle parameter: overcoord 2', 'Not used 3', 'Torsion/BO parameter', 'Torsion overcoordination 1',
                  'Torsion overcoordination 2', 'Conjugation 0 (not used)', 'Conjugation', 'vdWaals shielding',
                  'Cutoff for bond order (*100)', 'Valency angle conjugation parameter 2', 'Overcoordination parameter 3',
                  'Overcoordination parameter 4', 'Valency/lone pair parameter', 'Not used 4', 'Not used 5', 'Molecular energy (not used) 1',
                  'Molecular energy (not used) 2', 'Valency angle conjugation parameter 3']

    _speckeys_gulp = ['reaxff1_radii1', 'reaxff1_valence1', 'mass', 'reaxff1_morse3', 'reaxff1_morse2',
                      'reaxff_gamma', 'reaxff1_radii2', 'reaxff1_valence3', 'reaxff1_morse1', 'reaxff1_morse4',
                      'reaxff1_valence4', 'reaxff1_under', 'dummy1', 'reaxff_chi', 'reaxff_mu', 'dummy2',
                      'reaxff1_radii3', 'reaxff1_lonepair2', 'dummy3', 'reaxff1_over2', 'reaxff1_over1',
                      'reaxff1_over3', 'dummy4', 'dummy5', 'reaxff1_over4', 'reaxff1_angle1', 'dummy11',
                      'reaxff1_valence2', 'reaxff1_angle2', 'dummy6', 'dummy7', 'dummy8']

    _speckeys = ['cov.r', 'valency1', 'a.m', 'Rvdw', 'Evdw', 'gammaEEM', 'cov.r2', '#el', 'alfa',
                 'gammavdW', 'valency2', 'Eunder', 'n.u.1', 'chiEEM', 'etaEEM', 'n.u.2', 'cov.r3', 'Elp',
                 'Heat inc.', '13BO1', '13BO2', '13BO3', 'n.u.3', 'n.u.4', 'ov/un', 'val1', 'n.u.5', 'val3',
                 'vval4', 'n.u.6', 'n.u.7', 'n.u.8']

    _bondkeys_gulp = ['reaxff2_bond1', 'reaxff2_bond2', 'reaxff2_bond3', 'reaxff2_bond4', 'reaxff2_bo5',
                      'reaxff2_bo7', 'reaxff2_bo6', 'reaxff2_over', 'reaxff2_bond5', 'reaxff2_bo3',
                      'reaxff2_bo4', 'dummy1', 'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo8', 'reaxff2_bo9']

    _bondkeys = ['Edis1', 'LPpen', 'n.u.1', 'pbe1', 'pbo5', '13corr', 'pbo6', 'kov',
                 'pbe2', 'pbo3', 'pbo4', 'Etrip', 'pbo1', 'pbo2', 'ovcorr', 'n.u.2']

    _odkeys_gulp = ['reaxff2_morse1', 'reaxff2_morse3', 'reaxff2_morse2',
                    'reaxff2_morse4', 'reaxff2_morse5', 'reaxff2_morse6']

    _odkeys = ['Ediss', 'Ro', 'gamma', 'rsigma', 'rpi', 'rpi2 ']

    _anglekeys_gulp = ['reaxff3_angle1', 'reaxff3_angle2', 'reaxff3_angle3',
                       'reaxff3_conj', 'reaxff3_angle5', 'reaxff3_penalty', 'reaxff3_angle4']

    _anglekeys = ['Theta,o', 'ka', 'kb', 'pv1', 'pv2', 'kpenal', 'pv3']

    _torkeys_gulp = ['reaxff4_torsion1', 'reaxff4_torsion2', 'reaxff4_torsion3',
                     'reaxff4_torsion4', 'reaxff4_torsion5', 'dummy1', 'dummy2']

    _torkeys = ['V1', 'V2', 'V3', 'V2(BO)', 'vconj', 'n.u.1', 'n.u.2']

    _hbkeys_gulp = ['reaxff3_hbond1', 'reaxff3_hbond2',
                    'reaxff3_hbond3', 'reaxff3_hbond4']

    _hbkeys = ['Rhb', 'Dehb', 'vhb1', 'vhb2']

    _lammps_params_format = [{'connectivity': 'species', 'param_lines': 4, 'number_atoms': 1},
                             {'connectivity': 'bonds',
                                 'param_lines': 2, 'number_atoms': 2},
                             {'connectivity': 'off_diagonal',
                              'param_lines': 1, 'number_atoms': 2},
                             {'connectivity': 'angles',
                              'param_lines': 1, 'number_atoms': 3},
                             {'connectivity': 'torsions',
                              'param_lines': 1, 'number_atoms': 4},
                             {'connectivity': 'h_bonds', 'param_lines': 1, 'number_atoms': 3}]

    def __init__(self, species=None, params=None, species2id=None):
        self.species = species
        self.description = None
        self.remark = None
        self.params = params
        self.species2id = species2id

        if not params and species:
            self.params = self._gen_empty_params()

    def __repr__(self):
        outstr = ''
        outstr += 'Description: {}\n'.format(self.description)
        outstr += 'Initializing Species: {}\n'.format(self.species)

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
        return reaxFF_data(species=self.species.copy(), params=params_copy, species2id=self.species2id)

    def gen_species2id(self):
        """Generates a set of id for the existing species. Includes the placeholder *X*
           for the sake of consistency.
        """
        indices = range(0, len(self.species)+1)
        species = ['X'] + self.species
        self.species2id = bidict({sp: i for i, sp in zip(indices, species)})

    def clean_params(self):
        for k in self.params.values():
            k.dropna(inplace=True)

    def write_lammps(self) -> str:
        """write reaxff data in original input format
        """
        def _write_connectivity_parameters(connectivity: str, param_lines: int, number_atoms: int) -> str:
            c2l = {'species': 'atoms',
                   'bonds': 'bonds',
                   'off_diagonal': 'off-diagonal terms',
                   'angles': 'angles',
                   'torsions': 'torsions',
                   'h_bonds': 'hydrogen bonds'}
            paramkey = {'species': self._speckeys,
                        'bonds': self._bondkeys,
                        'off_diagonal': self._odkeys,
                        'angles': self._anglekeys,
                        'torsions': self._torkeys,
                        'h_bonds': self._hbkeys}
            outstr = ''

            c2l_length = len(c2l[connectivity])
            pars_length = p[connectivity].shape[1]//param_lines

            # Write parameter section
            outstr += '{:>3} ! Nr of {};'.format(
                p[connectivity].shape[0], c2l[connectivity])

            print_pars = paramkey[connectivity][:pars_length]
            outstr += regexp(len(print_pars)).format(*print_pars) + '\n'
            for i in range(1, param_lines):
                print_pars = paramkey[connectivity][pars_length *
                                                    i:pars_length*(i+1)]
                outstr += ' '*(13+c2l_length) + \
                    regexp(len(print_pars)).format(*print_pars) + '\n'

            # Writing values section
            for index, row in p[connectivity].iterrows():
                # Remove the 'source' column
                cur_row = row[row.index!='source']
                connectivity_symbol = index.split('-')

                # For species, print the chemical symbols directly
                if connectivity != 'species':
                    connectivity_symbol = [self.species2id[x]
                                           for x in connectivity_symbol]

                cstr = " ".join(
                    ["{:>2}"]*number_atoms).format(*connectivity_symbol)
                outstr += cstr

                print_vals = cur_row.values[:pars_length]
                outstr += ' '*(13+c2l_length-len(cstr)) + \
                    regex(len(print_vals)).format(*print_vals) + '\n'

                for i in range(1, param_lines):
                    print_vals = cur_row.values[pars_length*i:pars_length*(i+1)]
                    outstr += ' '*(13+c2l_length) + \
                        regex(len(print_vals)).format(*print_vals) + '\n'

            return outstr

        def regex(x): return " ".join(["{:10.4f}"]*x)

        def regexp(x): return ";".join(["{:>10}"]*x)

        outstr = ""
        p = self.params
        if self.description:
            outstr += ("{}".format(self.description.rstrip()))
        outstr += "\n"

        # Print general parameters
        # Assuming a final column of 'source', then let's not print that
        outstr += "{:8d} ! Number of general parameters\n".format(
            p['general'].shape[1]-('source' in p['general'].columns))

        # Index is a bit meaningless for general parameters
        for _i, row in p['general'].iterrows():
            for value, name in zip(row.values, row.index):
                # Skip Printing Sources
                if name != 'source':
                    outstr += "{0:8.4f} ! {1}\n".format(value, name)

        for par in self._lammps_params_format:
            outstr += _write_connectivity_parameters(**par)

        return outstr

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
                    self.species2id['X'] = 0
                else:
                    if current_atom not in self.species:
                        warn('{} is not found in initilizing species. Added anyway.'.format(
                            current_atom))

                    self.species2id[current_atom] = atom_no

                if 'X' not in self.species2id.keys():
                    self.species2id['X'] = 0

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

                if drop_new_species and any(x not in self.species2id.values() for x in current_ids):
                    continue

                current_atoms = [self.species2id.inverse[x][0]
                                 for x in current_ids]

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
            self.species2id = bidict({sp: None for sp in self.species})

            _read_general_parameters(rf)
            _read_species_parameters(rf)

            for par in self._lammps_params_format:
                if par['connectivity'] != 'species':
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
        tor_species = self.species + ['X']
        for torsion in product(tor_species, repeat=4):
            if torsion[::-1] not in torsions:
                # if torsion[1] != 'X' and torsion[2] != 'X':
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
