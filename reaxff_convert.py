##################################################
# This script converts a ReaxFF force field
# parameter file from the original format into
# one used by GULP.
##################################################
# MIT License
##################################################
# Author: Christopher Sewell
# Copyright: Copyright 2018, aiida-gulp
## Credits: [Christopher Sewell]
## License: MIT License
## Version: {major}.{minor}.{rel}
## Maintainer: Christopher Sewell
## Email: chrisj_sewell@hotmail.com
## Status: {dev_status}
##################################################


import copy
import re
from decimal import Decimal

import numpy as np

# TODO can X be in middle of species?


def skip(f, numlines):
    for _ in range(numlines):
        f.readline()


def readnumline(f, vtype):
    line = f.readline().split()
    if line[1] != '!':
        raise Exception(
            'not on a line containing ! (as expected) while reading {0}'.
            format(vtype))
    return int(line[0])


def split_numbers(string, as_decimal=False):
    """ get a list of numbers from a string (even with no spacing)
    :type string: str
    :type as_decimal: bool
    :param as_decimal: if True return floats as decimal.Decimal objects
    :rtype: list
    :Example:
    >>> split_numbers("1")
    [1.0]
    >>> split_numbers("1 2")
    [1.0, 2.0]
    >>> split_numbers("1.1 2.3")
    [1.1, 2.3]
    >>> split_numbers("1e-3")
    [0.001]
    >>> split_numbers("-1-2")
    [-1.0, -2.0]
    >>> split_numbers("1e-3-2")
    [0.001, -2.0]
    """
    _match_number = re.compile(
        r'-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *[+-]?\ *[0-9]+)?')
    string = string.replace(" .", " 0.")
    string = string.replace("-.", "-0.")
    return [
        Decimal(s) if as_decimal else float(s)
        for s in re.findall(_match_number, string)
    ]


def readlist(f):
    return split_numbers(f.readline())


def getvals(key, vals, names):
    out = []
    for name in names:
        out.append(vals[key.index(name)])
    return out


def transpose(lst):
    return list(map(list, zip(*lst)))


_paramkeys = [
    'Overcoordination 1', 'Overcoordination 2', 'Valency angle conjugation 1',
    'Triple bond stabilisation 1', 'Triple bond stabilisation 2',
    'C2-correction', 'Undercoordination 1', 'Triple bond stabilisation',
    'Undercoordination 2', 'Undercoordination 3',
    'Triple bond stabilization energy', 'Lower Taper-radius',
    'Upper Taper-radius', 'Not used 1', 'Valency undercoordination',
    'Valency angle/lone pair', 'Valency angle 1', 'Valency angle 2',
    'Not used 2', 'Double bond/angle', 'Double bond/angle: overcoord 1',
    'Double bond/angle: overcoord 2', 'Not used 3', 'Torsion/BO',
    'Torsion overcoordination 1', 'Torsion overcoordination 2', 'Not used 4',
    'Conjugation', 'vdWaals shielding', 'bond order cutoff',
    'Valency angle conjugation 2', 'Valency overcoordination 1',
    'Valency overcoordination 2', 'Valency/lone pair', 'Not used 5',
    'Not used 6', 'Not used 7', 'Not used 8', 'Valency angle conjugation 3'
]

_speckeys = [
    'idx', 'symbol', 'reaxff1_radii1', 'reaxff1_valence1', 'mass',
    'reaxff1_morse3', 'reaxff1_morse2', 'reaxff_gamma', 'reaxff1_radii2',
    'reaxff1_valence3', 'reaxff1_morse1', 'reaxff1_morse4', 'reaxff1_valence4',
    'reaxff1_under', 'dummy1', 'reaxff_chi', 'reaxff_mu', 'dummy2',
    'reaxff1_radii3', 'reaxff1_lonepair2', 'dummy3', 'reaxff1_over2',
    'reaxff1_over1', 'reaxff1_over3', 'dummy4', 'dummy5', 'reaxff1_over4',
    'reaxff1_angle1', 'dummy11', 'reaxff1_valence2', 'reaxff1_angle2',
    'dummy6', 'dummy7', 'dummy8'
]

_bondkeys = [
    'idx1', 'idx2', 'reaxff2_bond1', 'reaxff2_bond2', 'reaxff2_bond3',
    'reaxff2_bond4', 'reaxff2_bo5', 'reaxff2_bo7', 'reaxff2_bo6',
    'reaxff2_over', 'reaxff2_bond5', 'reaxff2_bo3', 'reaxff2_bo4', 'dummy1',
    'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo8', 'reaxff2_bo9'
]

_odkeys = [
    'idx1', 'idx2', 'reaxff2_morse1', 'reaxff2_morse3', 'reaxff2_morse2',
    'reaxff2_morse4', 'reaxff2_morse5', 'reaxff2_morse6'
]

_anglekeys = [
    'idx1', 'idx2', 'idx3', 'reaxff3_angle1', 'reaxff3_angle2',
    'reaxff3_angle3', 'reaxff3_conj', 'reaxff3_angle5', 'reaxff3_penalty',
    'reaxff3_angle4'
]

_torkeys = [
    'idx1', 'idx2', 'idx3', 'idx4', 'reaxff4_torsion1', 'reaxff4_torsion2',
    'reaxff4_torsion3', 'reaxff4_torsion4', 'reaxff4_torsion5', 'dummy1',
    'dummy2'
]

_hbkeys = [
    'idx1', 'idx2', 'idx3', 'reaxff3_hbond1', 'reaxff3_hbond2',
    'reaxff3_hbond3', 'reaxff3_hbond4'
]


def read_reaxff_file(inpath, reaxfftol=None):
    """
    :param inpath: path to reaxff file (in standard (lammps) format)
    :param reaxfftol: additional tolerance parameters
    :return:
    """

    toldict = _create_tol_dict(reaxfftol)

    with open(inpath, 'r') as f:
        # Descript Initial Line
        descript = f.readline()
        # Read Parameters
        # ----------
        if int(f.readline().split()[0]) != len(_paramkeys):
            raise IOError('Expecting {} general parameters'.format(
                len(_paramkeys)))
        reaxff_par = {}
        for key in _paramkeys:
            reaxff_par[key] = float(f.readline().split()[0])

        # Read Species Information
        # --------------------
        spec_dict = _read_species_info(f)

        # Read Bond Information
        # --------------------
        bond_dict = _read_bond_info(f)

        # Read Off-Diagonal Information
        # --------------------
        od_dict = _read_off_diagonal_info(f)

        # Read Angle Information
        # --------------------
        angle_dict = _read_angle_info(f)

        # Read Torsion Information
        # --------------------
        torsion_dict = _read_torsion_info(f)

        # Read HBond Information
        # --------------------
        hbond_dict = _read_hbond_info(f)

        return {
            "descript": descript.strip(),
            "tolerances": toldict,
            "params": reaxff_par,
            "species":
            spec_dict,  # spec_df.reset_index().to_dict(orient='list'),
            "bonds":
            bond_dict,  # bond_df.reset_index().to_dict(orient='list'),
            "off-diagonals":
            od_dict,  # od_df.reset_index().to_dict(orient='list'),
            "hbonds":
            hbond_dict,  # hbond_df.reset_index().to_dict(orient='list'),
            "torsions":
            torsion_dict,  # torsion_df.reset_index().to_dict(orient='list'),
            "angles":
            angle_dict,  # angle_df.reset_index().to_dict(orient='list')
        }


def _read_species_info(f):
    nspec = readnumline(f, 'species')
    skip(f, 3)
    spec_values = []
    for i in range(nspec):
        values = f.readline().split()
        symbol = values.pop(0)
        idx = 0 if symbol == 'X' else i + 1
        spec_values.append([idx, symbol] + [float(v) for v in values] +
                           readlist(f) + readlist(f) + readlist(f))
        if len(spec_values[i]) != len(_speckeys):
            raise Exception(
                'number of values different than expected for species {0}'.
                format(symbol))

    # spec_df = pd.DataFrame(spec_values, columns=speckey).set_index('idx')
    # spec_df['reaxff1_lonepair1'] = 0.5 * (spec_df.reaxff1_valence3 - spec_df.reaxff1_valence1)
    spec_dict = {k: v for k, v in zip(_speckeys, transpose(spec_values))}
    spec_dict['reaxff1_lonepair1'] = (
        0.5 * (np.array(spec_dict["reaxff1_valence3"]) - np.array(
            spec_dict["reaxff1_valence1"]))).tolist()
    return spec_dict


def _read_bond_info(f):
    # bondcode = ['idx1', 'idx2', 'Edis1', 'Edis2', 'Edis3',
    #             'pbe1', 'pbo5', '13corr', 'pbo6', 'kov',
    #             'pbe2', 'pbo3', 'pbo4', 'nu', 'pbo1', 'pbo2',
    #             'ovcorr', 'nu']
    # bonddescript = ['idx1', 'idx2', 'Sigma-bond dissociation energy', 'Pi-bond dissociation energy',
    #                 'Double pi-bond dissociation energy',
    #                 'Bond energy', 'Double pi bond order', '1,3-Bond order correction', 'Double pi bond order',
    #                 'Overcoordination penalty',
    #                 'Bond energy', 'Pi bond order', 'Pi bond order', 'dummy', 'Sigma bond order',
    #                 'Sigma bond order',
    #                 'Overcoordination BO correction', 'dummy']
    # bond_lookup_df = pd.DataFrame(np.array([bondcode, bonddescript]).T, index=bondkey)
    nbond = readnumline(f, 'bonds')
    skip(f, 1)
    bond_values = []
    for i in range(nbond):
        values = readlist(f)
        id1 = values.pop(0)
        id2 = values.pop(0)
        bond_values.append([int(id1), int(id2)] + values + readlist(f))
        if len(bond_values[i]) != len(_bondkeys):
            raise Exception(
                'number of values different than expected for bond')
    # bond_df = pd.DataFrame(bond_values, columns=bondkey).set_index(['idx1', 'idx2'])
    bond_dict = {k: v for k, v in zip(_bondkeys, transpose(bond_values))}
    return bond_dict


def _read_off_diagonal_info(f):
    nod = readnumline(f, 'off-diagonal')
    od_values = []
    for i in range(nod):
        values = readlist(f)
        id1 = int(values.pop(0))
        id2 = int(values.pop(0))
        od_values.append([id1, id2] + values)
        if len(od_values[i]) != len(_odkeys):
            raise Exception(
                'number of values different than expected for off-diagonal')
    # od_df = pd.DataFrame(od_values, columns=odkey).set_index(['idx1', 'idx2'])
    od_dict = {k: v for k, v in zip(_odkeys, transpose(od_values))}
    return od_dict


def _read_angle_info(f):
    nangle = readnumline(f, 'angle')
    angle_values = []
    for i in range(nangle):
        values = readlist(f)
        id1 = int(values.pop(0))
        id2 = int(values.pop(0))
        id3 = int(values.pop(0))
        angle_values.append([id1, id2, id3] + values)
        if len(angle_values[i]) != len(_anglekeys):
            raise Exception(
                'number of values different than expected for angle')
    # angle_df = pd.DataFrame(angle_values, columns=anglekey).set_index(['idx1', 'idx2', 'idx3'])
    angle_dict = {k: v for k, v in zip(_anglekeys, transpose(angle_values))}
    return angle_dict


def _read_torsion_info(f):
    ntors = readnumline(f, 'torsion')
    torsion_values = []
    for i in range(ntors):
        values = readlist(f)
        species1 = int(values.pop(0))
        species2 = int(values.pop(0))
        species3 = int(values.pop(0))
        species4 = int(values.pop(0))
        torsion_values.append([species1, species2, species3, species4] +
                              values)
        if len(torsion_values[i]) != len(_torkeys):
            raise Exception(
                'number of values different than expected for torsion')
    # torsion_df = pd.DataFrame(torsion_values, columns=torkey).set_index(['idx1', 'idx2', 'idx3', 'idx4'])
    torsion_dict = {k: v for k, v in zip(_torkeys, transpose(torsion_values))}
    return torsion_dict


def _read_hbond_info(f):
    nhb = readnumline(f, 'hbond')
    hbond_values = []
    for i in range(nhb):
        values = readlist(f)
        species1 = int(values.pop(0))
        species2 = int(values.pop(0))
        species3 = int(values.pop(0))
        hbond_values.append([species1, species2, species3] + values)
        if len(hbond_values[i]) != len(_hbkeys):
            raise Exception(
                'number of values different than expected for hbond {0},{1},{2}'
                .format(species1, species2, species3))
    # hbond_df = pd.DataFrame(hbond_values, columns=hbkey).set_index(['idx1', 'idx2', 'idx3'])
    hbond_dict = {k: v for k, v in zip(_hbkeys, transpose(hbond_values))}
    return hbond_dict


_tolerance_defaults = {
    "anglemin": 0.001,
    "angleprod": 0.000001,
    "hbondmin": 0.01,
    "hbonddist": 7.5,
    "torsionprod":
    0.000000001  # NB: needs to be lower to get comparable energy to lammps, but then won't optimize
}


def _create_tol_dict(reaxfftol):
    reaxfftol = {} if reaxfftol is None else reaxfftol.copy()
    toldict = {}
    for key, val in _tolerance_defaults.items():
        if key in reaxfftol:
            toldict[key] = reaxfftol[key]
        else:
            toldict[key] = val
    return toldict


# pylint: disable=too-many-locals
def write_lammps(data):
    """write reaxff data in original input format
    :param data: dictionary of data
    :rtype: str
    """
    outstr = ""

    def regex(x): return " ".join(["{:10.4f}"] * x)

    outstr += ("{}".format(data["descript"]))
    # if species_filter:
    #     outstr += ("#  (Filtered by: {})\n".format(species_filter))
    outstr += "\n"

    outstr += "{:8d} ! Number of general parameters\n".format(len(_paramkeys))
    for key in _paramkeys:
        outstr += "{0:8.4f} ! {1}\n".format(data["params"][key], key)

    outstr += '{0} ! Nr of atoms; cov.r; valency;a.m;Rvdw;Evdw;gammaEEM;cov.r2;#\n'.format(
        len(data["species"][_speckeys[0]]))
    outstr += 'alfa;gammavdW;valency;Eunder;Eover;chiEEM;etaEEM;n.u.\n'
    outstr += 'cov r3;Elp;Heat inc.;n.u.;n.u.;n.u.;n.u.\n'
    outstr += 'ov/un;val1;n.u.;val3,vval4\n'

    spec_data = transpose(
        [data['species'][key] for key in _speckeys if key != 'idx'])

    for spec in spec_data:
        outstr += '{0:3}'.format(spec[0]) + regex(8).format(*spec[1:9]) + '\n'
        outstr += '   ' + regex(8).format(*spec[9:17]) + '\n'
        outstr += '   ' + regex(8).format(*spec[17:25]) + '\n'
        outstr += '   ' + regex(8).format(*spec[25:33]) + '\n'

    outstr += '{0} ! Nr of bonds; Edis1;LPpen;n.u.;pbe1;pbo5;13corr;pbo6\n'.format(
        len(data["bonds"][_bondkeys[0]]))
    outstr += 'pbe2;pbo3;pbo4;n.u.;pbo1;pbo2;ovcorr\n'

    bond_data = transpose([data['bonds'][key] for key in _bondkeys])

    for bond in bond_data:
        outstr += '{:2} {:2} '.format(
            bond[0], bond[1]) + regex(8).format(*bond[2:10]) + '\n'
        outstr += '      ' + regex(8).format(*bond[10:18]) + '\n'

    outstr += '{0} ! Nr of off-diagonal terms; Ediss;Ro;gamma;rsigma;rpi;rpi2\n'.format(
        len(data["off-diagonals"][_odkeys[0]]))

    od_data = transpose([data['off-diagonals'][key] for key in _odkeys])

    for od in od_data:
        outstr += '{:2} {:2} '.format(od[0],
                                  od[1]) + regex(6).format(*od[2:8]) + '\n'

    outstr += '{0} ! Nr of angles;at1;at2;at3;Thetao,o;ka;kb;pv1;pv2\n'.format(
        len(data["angles"][_anglekeys[0]]))

    angle_data = transpose([data['angles'][key] for key in _anglekeys])

    for angle in angle_data:
        outstr += '{:2} {:2} {:2} '.format(*angle[0:3]) + regex(7).format(
            *angle[3:10]) + '\n'

    outstr += '{0} ! Nr of torsions;at1;at2;at3;at4;;V1;V2;V3;V2(BO);vconj;n.u;n\n'.format(
        len(data["torsions"][_torkeys[0]]))

    torsion_data = transpose([data['torsions'][key] for key in _torkeys])

    for tor in torsion_data:
        outstr += '{:2} {:2} {:2} {:2} '.format(*tor[0:4]) + regex(7).format(
            *tor[4:11]) + '\n'

    outstr += '{0} ! Nr of hydrogen bonds;at1;at2;at3;Rhb;Dehb;vhb1\n'.format(
        len(data["hbonds"][_hbkeys[0]]))

    hbond_data = transpose([data['hbonds'][key] for key in _hbkeys])

    for hbond in hbond_data:
        outstr += '{:2} {:2} {:2} '.format(*hbond[0:3]) + regex(4).format(
            *hbond[3:8]) + '\n'

    return outstr


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# pylint: disable=too-many-locals
def write_gulp(data, species_filter=None):
    """write reaxff data in GULP input format
    :param data: dictionary of data
    :param species_filter: list of symbols to filter
    :rtype: str
    """
    data = copy.deepcopy(data)

    descript = data["descript"]
    tol_par = data["tolerances"]
    reaxff_par = data["params"]
    id_sym_dict = {
        k: v
        for k, v in zip(data["species"]['idx'], data["species"]['symbol'])
    }
    spec_df = data["species"]
    spec_df['idxs'] = zip(spec_df['idx'])
    bond_df = data["bonds"]
    bond_df['idxs'] = zip(bond_df['idx1'], bond_df['idx2'])
    od_df = data["off-diagonals"]
    od_df['idxs'] = zip(od_df['idx1'], od_df['idx2'])
    hbond_df = data["hbonds"]
    hbond_df['idxs'] = zip(hbond_df['idx2'], hbond_df['idx1'],
                           hbond_df['idx3'])
    angle_df = data["angles"]
    angle_df['idxs'] = zip(angle_df['idx2'], angle_df['idx1'],
                           angle_df['idx3'])
    torsion_df = data["torsions"]
    torsion_df['idxs'] = zip(torsion_df['idx1'], torsion_df['idx2'],
                             torsion_df['idx3'], torsion_df['idx4'])

    # If reaxff2_bo3 = 1 needs to be set to 0 for GULP since this is a dummy value
    def gulp_conv1(val):
        return 0.0 if abs(val - 1) < 1e-12 else val

    bond_df['reaxff2_bo3'] = [gulp_conv1(i) for i in bond_df["reaxff2_bo3"]]

    # If reaxff2_bo(5,n) < 0 needs to be set to 0 for GULP since this is a dummy value
    def gulp_conv2(val):
        return 0.0 if val < 0.0 else val

    bond_df['reaxff2_bo5'] = [gulp_conv2(i) for i in bond_df["reaxff2_bo5"]]

    # TODO, this wasn't part of the original script, and should be better understood
    # but without it, the energies greatly differ to LAMMPS (approx equal otherwise)
    def gulp_conv3(val):
        return 0.0 if val > 0.0 else val

    spec_df['reaxff1_radii3'] = [
        gulp_conv3(i) for i in spec_df['reaxff1_radii3']
    ]

    attr_dicts = _create_attr_dicts(angle_df, bond_df, hbond_df, od_df,
                                    spec_df, torsion_df)
    angle_df, bond_df, hbond_df, od_df, spec_df, torsion_df = attr_dicts  # pylint: disable=unbalanced-tuple-unpacking

    outstr = ""

    write_data = _get_write_data_func(id_sym_dict, species_filter)

    outstr = _write_header(descript, outstr, reaxff_par, species_filter,
                           tol_par)

    outstr += "#  Species independent parameters \n"
    outstr += "#\n"
    outstr += ("reaxff0_bond     {:12.6f} {:12.6f}\n".format(
        reaxff_par['Overcoordination 1'], reaxff_par['Overcoordination 2']))
    outstr += (
        "reaxff0_over     {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n".
        format(reaxff_par['Valency overcoordination 2'],
               reaxff_par['Valency overcoordination 1'],
               reaxff_par['Undercoordination 1'],
               reaxff_par['Undercoordination 2'],
               reaxff_par['Undercoordination 3']))
    outstr += ("reaxff0_valence  {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n".format(
        reaxff_par['Valency undercoordination'],
        reaxff_par['Valency/lone pair'], reaxff_par['Valency angle 1'],
        reaxff_par['Valency angle 2']))
    outstr += ("reaxff0_penalty  {:12.6f} {:12.6f} {:12.6f}\n".format(
        reaxff_par['Double bond/angle'],
        reaxff_par['Double bond/angle: overcoord 1'],
        reaxff_par['Double bond/angle: overcoord 2']))
    outstr += ("reaxff0_torsion  {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n".format(
        reaxff_par['Torsion/BO'], reaxff_par['Torsion overcoordination 1'],
        reaxff_par['Torsion overcoordination 2'], reaxff_par['Conjugation']))
    outstr += "reaxff0_vdw      {:12.6f}\n".format(
        reaxff_par['vdWaals shielding'])
    outstr += "reaxff0_lonepair {:12.6f}\n".format(
        reaxff_par['Valency angle/lone pair'])

    outstr += "#\n"
    outstr += "#  Species parameters \n"
    outstr += "#\n"
    outstr += write_data(
        'reaxff1_radii', spec_df,
        ['reaxff1_radii1', 'reaxff1_radii2', 'reaxff1_radii3'])

    outstr += write_data('reaxff1_valence', spec_df, [
        'reaxff1_valence1', 'reaxff1_valence2', 'reaxff1_valence3',
        'reaxff1_valence4'
    ])
    outstr += write_data(
        'reaxff1_over', spec_df,
        ['reaxff1_over1', 'reaxff1_over2', 'reaxff1_over3', 'reaxff1_over4'])
    outstr += write_data('reaxff1_under kcal', spec_df, ['reaxff1_under'])
    outstr += write_data('reaxff1_lonepair kcal', spec_df,
                         ['reaxff1_lonepair1', 'reaxff1_lonepair2'])
    outstr += write_data('reaxff1_angle', spec_df,
                         ['reaxff1_angle1', 'reaxff1_angle2'])
    outstr += write_data('reaxff1_morse kcal', spec_df, [
        'reaxff1_morse1', 'reaxff1_morse2', 'reaxff1_morse3', 'reaxff1_morse4'
    ])

    outstr += "#\n"
    outstr += "# Element parameters \n"
    outstr += "#\n"
    outstr += write_data('reaxff_chi', spec_df, ['reaxff_chi'])
    outstr += write_data('reaxff_mu', spec_df, ['reaxff_mu'])
    outstr += write_data('reaxff_gamma', spec_df, ['reaxff_gamma'])

    outstr += "#\n"
    outstr += "# Bond parameters \n"
    outstr += "#\n"
    outstr += write_data('reaxff2_bo over bo13', bond_df, [
        'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo3', 'reaxff2_bo4',
        'reaxff2_bo5', 'reaxff2_bo6'
    ], ['s.reaxff2_bo7>0.001', 's.reaxff2_bo8>0.001'])
    outstr += write_data('reaxff2_bo bo13', bond_df, [
        'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo3', 'reaxff2_bo4',
        'reaxff2_bo5', 'reaxff2_bo6'
    ], ['s.reaxff2_bo7>0.001', 's.reaxff2_bo8<=0.001'])
    outstr += write_data('reaxff2_bo over', bond_df, [
        'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo3', 'reaxff2_bo4',
        'reaxff2_bo5', 'reaxff2_bo6'
    ], ['s.reaxff2_bo7<=0.001', 's.reaxff2_bo8>0.001'])
    outstr += write_data('reaxff2_bo', bond_df, [
        'reaxff2_bo1', 'reaxff2_bo2', 'reaxff2_bo3', 'reaxff2_bo4',
        'reaxff2_bo5', 'reaxff2_bo6'
    ], ['s.reaxff2_bo7<=0.001', 's.reaxff2_bo8<=0.001'])
    outstr += write_data('reaxff2_bond kcal', bond_df, [
        'reaxff2_bond1', 'reaxff2_bond2', 'reaxff2_bond3', 'reaxff2_bond4',
        'reaxff2_bond5'
    ])
    outstr += write_data('reaxff2_over', bond_df, ['reaxff2_over'])
    outstr += write_data(
        'reaxff2_pen kcal',
        bond_df, ['reaxff2_bo9'], ['s.reaxff2_bo9>0.0'],
        extra_data=[reaxff_par['Not used 1'], 1.0])
    outstr += write_data('reaxff2_morse kcal', od_df, [
        'reaxff2_morse1', 'reaxff2_morse2', 'reaxff2_morse3', 'reaxff2_morse4',
        'reaxff2_morse5', 'reaxff2_morse6'
    ])

    outstr += "#\n"
    outstr += "# Angle parameters \n"
    outstr += "#\n"
    outstr += write_data('reaxff3_angle kcal', angle_df, [
        'reaxff3_angle1', 'reaxff3_angle2', 'reaxff3_angle3', 'reaxff3_angle4',
        'reaxff3_angle5'
    ], ['s.reaxff3_angle2>0.0'])
    outstr += write_data('reaxff3_penalty kcal', angle_df, ['reaxff3_penalty'])
    outstr += write_data('reaxff3_conjugation kcal', angle_df,
                         ['reaxff3_conj'], ['abs(s.reaxff3_conj)>1.0E-4'], [
                             reaxff_par['Valency angle conjugation 1'],
                             reaxff_par['Valency angle conjugation 3'],
                             reaxff_par['Valency angle conjugation 2']
                         ])

    outstr += "#\n"
    outstr += "# Hydrogen bond parameters \n"
    outstr += "#\n"
    outstr += write_data('reaxff3_hbond kcal', hbond_df, [
        'reaxff3_hbond1', 'reaxff3_hbond2', 'reaxff3_hbond3', 'reaxff3_hbond4'
    ])

    outstr += "#\n"
    outstr += "# Torsion parameters \n"
    outstr += "#\n"
    outstr += write_data('reaxff4_torsion kcal', torsion_df, [
        'reaxff4_torsion1', 'reaxff4_torsion2', 'reaxff4_torsion3',
        'reaxff4_torsion4', 'reaxff4_torsion5'
    ])

    return outstr


def _write_header(descript, outstr, reaxff_par, species_filter, tol_par):
    outstr += "#\n"
    outstr += "#  ReaxFF force field\n"
    outstr += "#\n"
    outstr += "#  Original paper:\n"
    outstr += "#\n"
    outstr += "#  A.C.T. van Duin, S. Dasgupta, F. Lorant and W.A. Goddard III,\n"
    outstr += "#  J. Phys. Chem. A, 105, 9396-9409 (2001)\n"
    outstr += "#\n"
    outstr += "#  Parameters description:\n"
    outstr += "#\n"
    outstr += "#  {}".format(descript)
    if species_filter:
        outstr += "#  (Filtered by: {})\n".format(species_filter)
    outstr += "#\n"
    outstr += "#  Cutoffs for VDW & Coulomb terms\n"
    outstr += "#\n"
    outstr += ("reaxFFvdwcutoff {:12.4f}\n".format(
        reaxff_par['Upper Taper-radius']))
    outstr += ("reaxFFqcutoff   {:12.4f}\n".format(
        reaxff_par['Upper Taper-radius']))
    outstr += "#\n"
    outstr += "#  Bond order threshold - check anglemin as this is cutof2 given in control file\n"
    outstr += "#\n"
    outstr += (
        "reaxFFtol  {:12.10f} {:12.10f} {:12.10f} {:12.10f} {:12.10f} {:12.10f}\n"
        .format(
            0.01 * reaxff_par['bond order cutoff'], *[
                tol_par[s] for s in
                "anglemin angleprod hbondmin hbonddist torsionprod".split()
            ]))
    outstr += "#\n"
    return outstr


def _create_attr_dicts(*dicts):
    outdicts = []
    for dct in dicts:
        outdicts.append([
            AttrDict({k: v[i]
                      for k, v in dct.items()})
            for i in range(len(list(dct['idxs'])))
        ])
    return outdicts


def _get_write_data_func(id_sym_dict, species_filter):
    def write_data(name, df, wdata, conditions=None, extra_data=None):
        if extra_data is None:
            extra_data = []
        if conditions is None:
            conditions = []
        datastr = ""
        lines = []
        for s in df:

            evals = [True for c in conditions if eval(
                c)]  # pylint: disable=eval-used
            if not len(evals) == len(conditions):
                continue

            try:
                symbols = [id_sym_dict[idx] for idx in s.idxs]
            except KeyError:
                raise Exception(
                    'ERROR: Species number out of bounds when getting {}'.
                    format(name))

            if species_filter and not set(symbols).issubset(species_filter):
                continue

            regex = "{:2s} core " * len(s.idxs)
            regex += "{:8.4f} " * (len(wdata) + len(extra_data)) + "\n"
            lines.append(
                regex.format(*symbols + [s[w] for w in wdata] + extra_data))

        if lines:
            datastr += "{0} \n".format(name)
            for line in lines:
                datastr += line

        return datastr

    return write_data
