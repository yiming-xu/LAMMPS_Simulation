##################################################################
# The MIT License (MIT)                                          #
#                                                                #
# Copyright (c) 2018 RWTH Aachen University, Malte Doentgen,     #
#                    Felix Schmalz                               #
#                                                                #
# Permission is hereby granted, free of charge, to any person    #
# obtaining a copy of this software and associated documentation #
# files (the "Software"), to deal in the Software without        #
# restriction, including without limitation the rights to use,   #
# copy, modify, merge, publish, distribute, sublicense, and/or   #
# sell copies of the Software, and to permit persons to whom the #
# Software is furnished to do so, subject to the following       #
# conditions:                                                    #
#                                                                #
# The above copyright notice and this permission notice shall be #
# included in all copies or substantial portions of the Software.#
#                                                                #
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,#
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES#
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND       #
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT    #
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,   #
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING   #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR  #
# OTHER DEALINGS IN THE SOFTWARE.                                #
##################################################################
#
## @file	harvesting.py
## @author	Malte Doentgen, Felix Schmalz
## @date	2018/07/13
## @brief	class to retrieve data from quantum mechanics
#		calculations and calculate partition function
#

import sys
import time
import os
import numpy
import openbabel
import traceback
import scipy.optimize
import scipy.constants
import subprocess

import dbhandler
import log as Log

class Harvesting:
	## @brief	Harvesting
	## @version	2016/02/11
	#
	## @version	2016/02/29:
	#		dHf,0_CaHbOcNd = dH + a*dHf,0_CH4 + c*dHf,0_H2O + 0.5*d*dHf,0_N2 + (0.5*b -2*a -c)*dHf,0_H2
	#
	def __init__(self, Database=None, DBhandle=None):
		self.h = scipy.constants.h	# J*s
		self.R = scipy.constants.R	# J/mol*K
		self.NA = scipy.constants.N_A	# molecules/mol
		self.kB = scipy.constants.k	# J/molecules*K
		self.kBw = 0.695	# cm^-1 /K

		# . element symbols
		self.symbol = {1: 'H', 2: 'He', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 16: 'S', 17: 'Cl', 18: 'Ar'}
		self.element = {}
		for sym in self.symbol:
			self.element[self.symbol[sym]] = sym

		if DBhandle == None: self.db = dbhandler.Database(Name=Database)
		else: self.db = DBhandle
		self.log = Log.Log(Width=70)

		# . thermochemistry / J/mol
		self.Hf0 = {	'[C]':	716.68*1E3,		# M.W. Chase Jr., J.Phys.Chem.Ref.Data, Monograph 9 (1998), 1-1951
				'[O]':	249.18*1E3,		# J.D. Cox, D.D. Wagman, V.A. Madvedev, CODATA Key Values for Thermodynamics,
								# Hemisphere Publishing Corp., New York, 1984, 1
				'[N]':	472.68*1E3,		# M.W. Chase Jr., J.Phys.Chem.Ref.Data, Monograph 9 (1998), 1-1951
				'[H]':	218.00*1E3,		# M.W. Chase Jr., J.Phys.Chem.Ref.Data, Monograph 9 (1998), 1-1951
				'O=O':	0.0,
				'O[O]':	12.26*1E3,		# ATcT
				'OO':	-135.457*1E3,		# AtcT
				}
		self.Ha0 = {}
		self.Q = {}
		self.NASA = {}
		self.Arr = {}

		# . openbabel
		self.conv = openbabel.OBConversion()
		self.conv.SetInAndOutFormats('mdl', 'can')

	## @brief	harvest data from a g09 log-file
	## @param	Filename	name of g09 log-file
	## @return	molecular mass, energetics, sym.rot. temperatures, harmonic
	#		frequencies, internal sym.rot. tempertures, and opt. geometry
	## @version	2016/01/06:
	#		Read thermodynamic data from g09 log-file. This comprises
	#		translational, rotational, harmonic, and anharmonic
	#		contributions to the partition function. Further, the
	#		zero-point corrected single-point energy is extracted.
	#		In case of CBS-QB3 computations the CBS-QB3 single-point
	#		energy is extracted aswell.
	#
	#		Modify this function according to the output produced by
	#		the method you use for computing the SPE (e.g. G4, CCSD(T)).
	#
	## @version	2016/02/24:
	#		optimized charge-centered geoemtry is extracted.
	#		extract spin-multiplicity
	#		tr is [0,0,0] at default
	#		check on normal termination
	#
	## @version	2016/03/18:
	#		encapsulated in try-except-clause
	#		for unexpected behavior.
	#		test on empty line to avoid endless loop.
	#
	## @version	2016/03/29:
	#		Tr and symmetry number are not merged, since
	#		calculated symmetries can vary due to computational
	#		errors. merging is done later, after re-check
	#		of DB for biggest symmetry number found.
	#
	## @version	2016/03/29:
	#		The internal rotor information outputted by g09
	#		is checked against internal rules to avoid wrong
	#		interpretation of rotation axes which are inside
	#		rings.
	#
	## @version	2016/03/31:
	#		harvesting of IRC information added. Maximum
	#		deflection from TS structure is used to check
	#		reactants/products of reaction.
	#
	## @version	2016/06/22:
	#		'confirmed' keyword used for species as well (if they break apart).
	#
	## @version	2016/07/01:
	#		IRC check forward and backward.
	#
	## @version	2016/07/02:
	#		Using 'F' instead of 'H' due to the larger covalent radius
	#		of 'F'. This is required to correctly identify [H][H].
	#		For the SMILES generation, the atomic number is set to
	#		hydrogen.
	#
	## @version	2016/12/09:
	#		tolerance of 1 cm^-1 when gathering HR frequencies
	#
	def harvestGaussian(self, Filename):
		molinfo = {
			# . crucial info
			'smi': '',	# SMILES representation of the molecule
			'M': 0.0,	# molecular mass
			'cSPE': 0.0,	# zero-point corrected single-point energy
			'Tr': [0,0,0],	# rotational temperatures
			'v0': [],	# harmonic frequencies
			'eff': 0.0,	# ReaxFF reference energy
			'Ip': [],	# ReaxFF reference inertia
			'Htherm': 0.0,	# thermal enthalpy at 298.15K
			'spin': 0.0,	# spin multiplicity
			# . Hindered Rotor info
			'rotor': {},	# internal rotation data
			# . additional info
			'sym': 1,	# rotational symmetry number
			'geo': {},	# charge-centered coordinates {id:[type,[x,y,z]]}
			'cmd': [],	# g09 command line parameters used for all calculations
			'confirmed': True,	# IRC TS confirmation
			'normalTerm': False,	# normal termination flag
			'errorLink': 0	# g09 part with error
		}
		rotor = {}
		geo = {}
		ivib = []	#   subdata: harmonic frequency
		iper = []	#   subdata: periodicity
		isym = []	#   subdata: symmetry number
		imom = []	#   subdata: reduced moment of inertia
		bond = []	#   subdata: connection between frequency and rotor
		bond2 = {}	#   subdata: connection between frequency and rotor
		ax = []		#   subdata: rotor axes
		irot = 0	#   subdata: number of internal rot deg of freedom
		irc = [[], []]

		# . parse Gaussian log file
		print('file:', Filename)
		done = False
		reaxFFdone = False
		reader = open(Filename, 'r')
		line = reader.readline()
		try:
			while not done:
				if 'Gaussian 09:' in line:
					line = reader.readline(); line = reader.readline()

					# . harvest g09 command line
					while not line.startswith(' # ') and line != '':
						line = reader.readline()
					tmp = line[3:-1]
					line = reader.readline()
					while '----------' not in line and line != '':
						tmp += line[1:-1]
						line = reader.readline()
					cmd = tmp.split()
					cmd = ' '.join([c for c in tmp.split() if '=' not in c and 'opt' not in c and 'freq' not in c and 'int' not in c and 'scf' not in c and 'maxdisk' not in c.lower() and 'symmetry' not in c])
					molinfo['cmd'].append(cmd)

				# . harvest ReaxFF information
				#   stands at start of every job
				#   if no 'normal termination' follows, the job was aborted
				elif line.startswith(' REAXFF:'):
					if not reaxFFdone:
						words = line.strip().split()
						molinfo['smi'] = words[1]
						molinfo['eff'] = float(words[2])
						molinfo['Ip'] = [float(w) for w in words[3:]]
						reaxFFdone = True
					molinfo['normalTerm'] = False

				# . harvest spin multiplicity
				elif 'Symbolic Z-matrix:' in line:
					line = reader.readline()
					words = line.strip().split()
					molinfo['spin'] = float(words[5])

				# . harvest internal rotation data	(anharmonicity)
				elif 'Hindered Internal Rotation Analysis' in line and not ivib:
					while '- Thermochemistry For Hindered Internal Rotation -' not in line and 'No hindered rotor corrections are necessary' not in line and 'This molecule contains only rings' not in line and line != '':
						line = reader.readline()
						words = line.strip().split()
						if line.startswith(' Number of internal rotation degrees of freedom'):
							irot = int(words[-1])
						#elif line.startswith(' Normal Mode Analysis for Internal Rotation'): # or line.startswith(' Redo normal mode analysis with added constraints')
						#	ivib = []
						elif 'Frequencies ---' in line and len(ivib) < irot:
							for v in [float(w) for w in words[2:]]:
								ivib.append(v)
						elif len(words) == 6 and words[0][-1] == ')':
							iper.append(int(words[3])) # periodicity
							isym.append(int(words[4])) # symmetry number
						elif 'Reduced Moments ---' in line:
							for m in [float(w) for w in words[3:]]:
								if m not in imom: imom.append(m)
						elif line.strip() == 'Bond':
							line = reader.readline()
							words = line.strip().split()
							bidx = 0
							while len(words) > 3 and words[1] == '-':
								ax.append(sorted([int(words[0]), int(words[2])])) # bond partners of axis
								if bidx+1 > len(bond): bond.append([])
								bond[bidx] += [float(w) for w in words[3:]] # axis share
								bidx += 1
								line = reader.readline()
								words = line.strip().split()

				# . harvest harmonic frequencies	(vibration)
				elif 'Harmonic frequencies (cm**-1)' in line:
					if molinfo['v0']: molinfo['v0'] = []
					while '- Thermochemistry -' not in line and line != '':
						line = reader.readline()
						if 'Frequencies --' in line:
							words = line.strip().split()
							molinfo['v0'] += [float(w) for w in words[2:]]

				# . harvest molecular mass		(translation)
				elif 'Molecular mass:' in line:
					words = line.strip().split()
					molinfo['M'] = float(words[2])

				# . harvest rotational temperatures	(rotation)
				elif 'Rotational symmetry number' in line:
					words = line.strip().split()
					molinfo['sym'] = int(words[3][:-1])
				elif 'Rotational temperatures (Kelvin)' in line or 'Rotational temperature (Kelvin)' in line:
					words = line.strip().split()
					molinfo['Tr'] = [float(w) for w in words[3:] if '***' not in w]

				# . harvest energetics
				elif 'Sum of electronic and zero-point Energies=' in line:
					words = line.strip().split()
					molinfo['cSPE'] = float(words[6])
				elif 'Sum of electronic and thermal Enthalpies=' in line:
					words = line.strip().split()
					molinfo['Htherm'] = float(words[6])
				elif 'CBS-QB3 (0 K)=' in line:
					words = line.strip().split()
					molinfo['cSPE'] = float(words[3])
				elif 'CBS-QB3 Enthalpy=' in line:
					words = line.strip().split()
					molinfo['Htherm'] = float(words[2])
				elif 'G3MP2(0 K)=' in line:
					words = line.strip().split()
					molinfo['cSPE'] = float(words[2])
				elif 'G3MP2 Enthalpy=' in line:
					words = line.strip().split()
					molinfo['Htherm'] = float(words[2])

				# . harvest geometry
				elif 'Standard orientation' in line:
					for i in range(5): line = reader.readline()
					while '-----------------------------------' not in line and line != '':
						words = line.strip().split()
						geo[int(words[0])] = [int(words[1]),[float(words[3]),float(words[4]),float(words[5])]]
						line = reader.readline()
					molinfo['geo'] = geo

				# . harvest IRC information
				elif 'Input orientation:' in line:
					tmp = []
					line = reader.readline()
					line = reader.readline()
					line = reader.readline()
					line = reader.readline()
					line = reader.readline()
					while '---' not in line:
						words = line.split()
						tmp.append([int(words[1]), [float(words[3]), float(words[4]), float(words[5])]])
						line = reader.readline()
				elif 'Calculation of FORWARD path complete.' in line: irc[0] = tmp
				elif 'Calculation of REVERSE path complete.' in line: irc[1] = tmp

				# . normal termination?
				elif 'Normal termination of Gaussian' in line:
					molinfo['normalTerm'] = True
				elif 'Error termination ' in line:
					molinfo['normalTerm'] = False
					words = line.strip().split()
					if len(words) > 7: molinfo['errorLink'] = words[5].split('/')[-1]

				line = reader.readline()
				if line == '': done = True
			reader.close()
		except:
			# . return here if exception
			self.log.printIssue(Text='Exception while executing\n'+traceback.format_exc(), Fatal=False)
			molinfo['normalTerm'] = False
			molinfo['M'] = 0.0
			return molinfo

		# . return here if crucial data misses
		#   eff=0 and Tr=[0,0,0] allowed ([H])
		if molinfo['cSPE'] == 0.0 or molinfo['M'] == 0.0 or molinfo['smi'] == '' or molinfo['Ip'] == [] or molinfo['Htherm'] == 0.0:
			molinfo['M'] = 0.0
			molinfo['normalTerm'] = False
			return molinfo
		
		# . delete frequencies too close to each other
		keep = [not any(abs(ivib[i]-ivib[j])<1 for i in range(j+1,irot)) for j in range(irot)]
		if not all(keep):
			ivib = [ivib[i] for i in range(irot) if keep[i]]
			iper = [iper[i] for i in range(irot) if keep[i]]		
			isym = [isym[i] for i in range(irot) if keep[i]]
			imom = [imom[i] for i in range(irot) if keep[i]]
			bond = [bond[i] for i in range(irot) if keep[i]]
			ax   = [ax[i]   for i in range(irot) if keep[i]]
		nfreqs = len(ivib)

		# . merge internal rotation data (4.649783E-48 (kg/amu)*(m/bohr)**2
		#   don't use a frequency twice
		unused = [True]*nfreqs
		for i in range(nfreqs):
			idx = bond[i].index(max([bond[i][k] for k in range(nfreqs) if unused[k]]))
			unused[idx] = False
			if imom[idx] != 0.0:
				rotor[ivib[idx]] = isym[i]**2 *self.h**2 /(8 *numpy.math.pi**2 *self.kB *imom[idx] *4.649783E-48 *iper[i]**2)

		# . remove internal rotation axes which are part of ring
		#   atom.thisown=False should solve instabilities
		mol = openbabel.OBMol()
		for a in geo:
			atom = openbabel.OBAtom()
			atom.thisown = False
			atom.SetAtomicNum(geo[a][0])
			atom.SetVector(geo[a][1][0], geo[a][1][1], geo[a][1][2])
			mol.AddAtom(atom)
		mol.ConnectTheDots()
		mol.PerceiveBondOrders()
		tmp_rotor = self.findInternalRotors(Mol=mol)

		# . delete false rotors, which are part of a ring
		#  ax: axes found by g09
		#  tmp_rotor: axes found by local method, no rings
		#  rotor: mapping of rot. freq. <-> rot. temp. (g09 data)
		for idx in range(nfreqs):
			if ax[idx] not in tmp_rotor and ivib[idx] in rotor: del rotor[ivib[idx]]
		molinfo['rotor'] = rotor

		# . check number of molecules (==1)
		#   if species, but nmol > 1, dont use qm-data
		#   return information for species
		if molinfo['smi'] == '[H]' or molinfo['smi'] == '[H][H]': nmol = 1
		else: nmol = len(self.conv.WriteString(mol).strip().split('.'))
		if ':' not in molinfo['smi']:
			molinfo['confirmed'] = (nmol == 1)
			return molinfo

		# . IRC check, only for TS
		#   build reactants
		mol = openbabel.OBMol()
		for a in irc[0]:
			atom = openbabel.OBAtom()
			atom.thisown = False
			if a[0] == 1: atom.SetAtomicNum(9); atom.SetType('H')
			else: atom.SetAtomicNum(a[0])
			atom.SetVector(a[1][0], a[1][1], a[1][2])
			mol.AddAtom(atom)
		mol.ConnectTheDots()
		for atom in openbabel.OBMolAtomIter(mol):
			if atom.GetType() == 'H': atom.SetAtomicNum(1)
		mol.AssignSpinMultiplicity(True)
		reactants = []
		for frag in mol.Separate():
			if frag.NumHvyAtoms() == 0:
				if frag.NumAtoms() == 1: reactants.append('[H]')
				elif frag.NumAtoms() == 2: reactants.append('[H][H]')
				continue
			self.conv.SetInAndOutFormats('mdl','inchi')
			inchi = self.conv.WriteString(frag)
			self.conv.SetInAndOutFormats('inchi','mdl')
			self.conv.ReadString(frag, inchi)
			self.conv.SetInAndOutFormats('mdl','can')
			reactants.append(self.conv.WriteString(frag).strip())

		# . build products
		mol = openbabel.OBMol()
		for a in irc[1]:
			atom = openbabel.OBAtom()
			atom.thisown = False
			if a[0] == 1: atom.SetAtomicNum(9); atom.SetType('H')
			else: atom.SetAtomicNum(a[0])
			atom.SetVector(a[1][0], a[1][1], a[1][2])
			mol.AddAtom(atom)
		mol.ConnectTheDots()
		for atom in openbabel.OBMolAtomIter(mol):
			if atom.GetType() == 'H': atom.SetAtomicNum(1)
		mol.AssignSpinMultiplicity(True)
		products = []
		for frag in mol.Separate():
			if frag.NumHvyAtoms() == 0:
				if frag.NumAtoms() == 1: products.append('[H]')
				elif frag.NumAtoms() == 2: products.append('[H][H]')
				continue
			self.conv.SetInAndOutFormats('mdl','inchi')
			inchi = self.conv.WriteString(frag)
			self.conv.SetInAndOutFormats('inchi','mdl')
			self.conv.ReadString(frag, inchi)
			self.conv.SetInAndOutFormats('mdl','can')
			products.append(self.conv.WriteString(frag).strip())

		# . check
		reacIRC = sorted( [sorted(reactants), sorted(products)] )
		reacFF  = sorted( [sorted(molinfo['smi'].split(':')[0].split(',')), sorted(molinfo['smi'].split(':')[1].split(','))] )
		molinfo['confirmed'] = (reacIRC == reacFF)

		# . return information (for TS)
		return molinfo

	## @brief	find internal rotation axes
	## @param	Mol	OBMol object
	## @version	2016/03/29:
	#		single bonds between non-ring non-hydrogen atoms
	#		with more than one bond are used as rotors.
	#
	def findInternalRotors(self, Mol):
		rotor = []
		for atom in openbabel.OBMolAtomIter(Mol):
			if not atom.IsHydrogen() and not atom.IsInRing():
				for bond in openbabel.OBAtomBondIter(atom):
					ptr = bond.GetNbrAtom(atom)
					if ptr.GetId() < atom.GetId(): continue
					elif bond.GetBO() == 1 and atom.BOSum() > 1 and ptr.BOSum() > 1 and not ptr.IsHydrogen():
						idx = [int(atom.GetId())+1, int(ptr.GetId())+1]
						rotor.append(idx)
		return rotor

	## @brief	harvest a folder of finished g09 jobs
	## @param	Folder	Name of the folder without '/'
	## @param	Files	g09 files start with this string
	## @param	Type	g09 files end with this string
	## @param	Fail	behavior, if harvesting fails
	## @param	Done	behavior, if harvesting succeeds
	## @param	Quiet	don't print a summary per file
	## @return	done	no. of files transferred to DB
	## @return	fail	no. of files that failed somehow
	## @return	total	no. of all files in folder
	## @return	barrierless	dict of barrierless reactions with molmass and geometry
	## @version	2016/02/24
	#
	## @version	2016/06/15:
	#		geometry for barrierless reactions is not stored anymore
	#
	## @version	2016/12/01:
	#		on failure of Gaussian, write new input file from last coordinates ('retry_*')
	#
	def harvestFolder(self, Folder, Files='tmp_', Type='log', Fail='keep', Done='delete', Quiet=False):
		done = 0
		fail = 0
		total = 0
		barrierless = {}

		# . create list of files
		datafiles = []
		if not os.path.exists(Folder):
			self.log.printIssue('No folder named %s.' %(Folder))
		else:
			for file in os.listdir(Folder):
				if file.startswith(Files) and file.endswith(Type):
					datafiles.append(file)
			if datafiles == []:
				self.log.printIssue('No files found in "%s" matching "%s*%s"' %(Folder,Files,Type))

		# . loop list of files
		failfile = {}
		retryfile = {}
		for file in sorted(datafiles):
			total += 1
			failure = True
			methodok = True
			filename = Folder+'/'+file
			outstring = '%s: ' %filename
			molinfo = self.harvestGaussian(filename)
			smi   = molinfo['smi']
			cmd   = molinfo['cmd']
			M     = molinfo['M']
			Ip    = molinfo['Ip']
			cSPE  = molinfo['cSPE']
			v0    = molinfo['v0']
			Tr    = molinfo['Tr']
			rotor = molinfo['rotor']
			confirmed  = molinfo['confirmed'] # default=True if species, reaction default=False
			normalTerm = molinfo['normalTerm']
			errorLink = molinfo['errorLink']
			geo = molinfo['geo']

			if ':' in smi: revsmi = smi.split(':')[0]+':'+smi.split(':')[1]
			else: revsmi = ''
			if smi == '':
				fail += 1
				outstring += 'erroneous data: SMILES not found,'
			elif M == 0.0 or Ip == [] or cSPE == 0.0:
				fail += 1
				moltype = 'reaction' if ':' in smi else 'species'
				outstring += '%s %s, ' %(moltype,smi)
				outstring += 'erroneous data (%s): Molecular information missing,' %errorLink
				if smi in self.db.getSpecies() or smi in self.db.getTSReactions() or revsmi in self.db.getTSReactions(): outstring += ' (already in DB)'
				else: failfile[file] = [smi, 'Molecular information missing']
			else:
				moltype = 'reaction' if ':' in smi else 'species'
				outstring += '%s %s, ' %(moltype,smi)

				if moltype == 'reaction':
					# . reaction
					# . verify proper DB
					methodok = self.db.verifyMethod(cmd[0],TS=True)
					if not normalTerm:
						fail += 1
						outstring += 'erroneous data (%s): Abnormal termination,' %errorLink
						if smi in self.db.getTSReactions() or revsmi in self.db.getTSReactions(): outstring += ' (already in DB)'
						else: failfile[file] = [smi, 'Abnormal termination']
					elif not methodok:
						fail += 1
						outstring += 'method does not match DB,'
					elif not confirmed:
						fail += 1
						outstring += 'IRC check did not confirm TS,'
						if smi in self.db.getTSReactions() or revsmi in self.db.getTSReactions(): outstring += ' (already in DB)'
						else: failfile[file] = [smi, 'IRC check did not confirm TS']
					else:
						if all([v > 0 for v in v0]):
							# . barrierless reaction
							outstring += 'barrierless TS ignored,'
							barrierless[smi] = M
						else:
							# . normal reaction
							added = self.db.addReaction(Molinfo=molinfo)
							if added: outstring += 'new TS, added to DB,'
							else: outstring += 'updated entry,'
							done += 1
						failure = False
				else:
					# . species
					# . verify proper DB
					methodok = self.db.verifyMethod(cmd[0],TS=False)
					if not normalTerm:
						fail += 1
						outstring += 'erroneous data (%s): Abnormal termination,' %errorLink
						if smi in self.db.getSpecies(): outstring += ' (already in DB)'
						else: failfile[file] = [smi, 'Abnormal termination']
					elif not methodok:
						fail += 1
						outstring += 'method does not match DB,'
					elif not confirmed:
						fail += 1
						outstring += 'optimization check did not confirm species,'
						if smi in self.db.getSpecies(): outstring += ' (already in DB)'
						else: failfile[file] = [smi, 'Optimization check did not confirm species']
					else:
						added = self.db.addSpecies(Molinfo=molinfo)
						if added:
							outstring += 'added to DB,'
							done += 1
						else:
							outstring += 'already in DB,'
						failure = False

			outstring += '\n' +(' methodok' if methodok else '') +(' confirmed' if confirmed else '') +(' failure' if failure else '')
				
			# . write new Gaussian input file from last coordinates
			#   use existing com and job files
			if smi and M and methodok and failure and (not normalTerm or confirmed): # and confirmed
				try:
					stopcounter = 0
					retry = True
					buffer = []
					oldgeo = {}
					id = 1
					jobid = file[len(Files):-len(Type)-1]
					oldfilename = Folder+'/'+Files+jobid+'.com'
					oldfile = open(oldfilename, 'r')
					# . read com file
					line = oldfile.readline()
					while line != '':
						if line.startswith('%chk='): line = '%chk=retry_'+jobid+'.chk\n'
						elif line == '\n': stopcounter += 1
						elif len(line.split())==2 and stopcounter==2: 
							# . reached geo part
							buffer += [line] # spin and charge
							w = oldfile.readline().split();
							while len(w) == 4:
								oldgeo[id] = [self.element[w[0]],[float(w[1]),float(w[2]),float(w[3])]]; id += 1
								w = oldfile.readline().split()
							if oldgeo == geo: retry = False; break # geos match, no use retrying
							for atom in list(geo.values()): # insert new geo
								buffer += [' %s\t%f\t%f\t%f\n' %(self.symbol[atom[0]],atom[1][0],atom[1][1],atom[1][2])]
							line = '\n'
						buffer += [line]
						line = oldfile.readline()
					oldfile.close()
					
					if retry:
						# new com file
						newfile = open(Folder+'/'+'retry_'+jobid+'.com', 'w')
						for line in buffer:
							newfile.write(line)
						newfile.close()
						# new job file
						buffer = []
						oldfile = open(Folder+'/'+'job_'+Files+jobid+'.sh', 'r')
						for line in oldfile:
							if line.startswith('#BSUB -J '): line = '#BSUB -J retry_'+jobid+'\n'
							if line.startswith('g09 '): line = 'g09 < retry_'+jobid+'.com > retry_'+jobid+'.log\n'
							buffer += [line]
							#newfile.write(line)
						oldfile.close()
						newfilename = Folder+'/'+'job_retry_'+jobid+'.sh'
						newfile = open(newfilename, 'w')
						for line in buffer:
							newfile.write(line)
						newfile.close()
						#outstring += ' retrying from last geo,'
						outstring += ' made retry file,'
						# . start the job again
						#subprocess.call('bsub < '+Folder+'/'+'job_retry_'+jobid+'.sh', shell=True)
						retryfile[oldfilename] = [smi, newfilename]
				except:
					outstring += ' failed to retry (%s),' %sys.exc_info()[0]

			# . store failure information
			writer = open(Folder+'/qm.fail', 'w')
			for file in sorted(failfile):
				writer.write('%20s %s %s\n' %(file, failfile[file][0], failfile[file][1]))
			writer.close()
			
			# . store retry information
			writer = open(Folder+'/qm.retry', 'w')
			for file in sorted(retryfile):
				writer.write('bsub < %s # %s\n' %(retryfile[file][1], retryfile[file][0]))
			writer.close()

			# . delete or keep file
			if failure and Fail == 'delete' or not failure and Done == 'delete':
				p = len(Type) + 1
				if os.path.exists(filename):                   os.remove(filename)
				if os.path.exists(filename[:-p] + '.chk'):     os.remove(filename[:-p] + '.chk')
				if os.path.exists(filename[:-p] + '_log.txt'): os.remove(filename[:-p] + '_log.txt')
				outstring += ' file was deleted'
			else:
				#outstring += ' file was kept'
				pass
			if not Quiet: self.log.printComment(outstring, onlyBody=True)

		return done, fail, total, barrierless

	## @brief	get reference enthalpies (Ha0) from DB
	## version	2016/03/03:
	#		Ha0 is neccessary when fitting NASA polynomials
	#
	## version	2016/09/01:
	#		removed warning messages, since dbhandler will give them anyway
	#
	def getHa0(self):
		# . H
		self.Ha0['[H]']  = self.db.getConf(Smi='[H]')[0]['Htherm']
		if self.Ha0['[H]'] == None: self.Ha0['[H]'] = 0.0
		# . C
		self.Ha0['[C]'] = self.db.getConf(Smi='[C]')[0]['Htherm']
		if self.Ha0['[C]'] == None: self.Ha0['[C]'] = 0.0
		# . O
		self.Ha0['[O]'] = self.db.getConf(Smi='[O]')[0]['Htherm']
		if self.Ha0['[O]'] == None: self.Ha0['[O]'] = 0.0
		# . N
		self.Ha0['[N]']  = self.db.getConf(Smi='[N]')[0]['Htherm']
		if self.Ha0['[N]'] == None: self.Ha0['[N]'] = 0.0

		return True

	## @brief	compute partition function for species / TS
	## @param	Smi	SMILES for which the partition function
	#			will be calculated
	## @param	T	list of temperatures for which the
	#			partition function will be calculated
	## @return	Q	dict containing temperatures and the
	#			respective partition function
	## @version	2016/02/16:
	#		compuate partition functions based on database
	#		information.
	#
	## @version	2016/03/29:
	#		workaround for transition state rotors added.
	#		Reactant rotors are used, but with closest
	#		analogs to the TS frequencies (g09 hindered
	#		rotor command line argument fails for some
	#		structures!).
	#
	def calcPartFunc(self, Smi, T=[]):
		# . get properties from database
		#   prop = {M, cSPE, Tr, v0, rotor, spin, form, htherm}
		conf = self.db.getConf(Smi=Smi)
		energy = [prop['cSPE'] for prop in conf]
		emin = min(energy); idx = energy.index(emin)
		rotor = {}

		# . anticipate internal rotation information
		if ':' in Smi:
			# . internal rotors of reactants
			#   take rotor data from minmal energy conformer
			#   and merge to "TS rotor"
			ts_rotor = {}
			for smi in Smi.split(':')[0].split(','):
				tmp = self.db.getConf(Smi=smi)
				en = [prop['cSPE'] for prop in tmp]
				enmin = min(en); tidx = en.index(enmin)
				for v in tmp[tidx]['rotor']: ts_rotor[v] = tmp[tidx]['rotor'][v]
			# . TS harmonic frequencies
			#   match freq from TS (ts_freq) to freq from reactants (ts_rotor)
			ts_freq = conf[idx]['v0']
			for v in ts_rotor:
				diff = [abs(vib-v) for vib in ts_freq]
				midx = diff.index(min(diff))
				rotor[ts_freq[midx]] = ts_rotor[v]
		else: rotor = conf[idx]['rotor']

		# . compute partition function
		Q = {}
		for t in T:
			# . translational partition function 1/m3
			#   g/mol -> kg/molecule
			m = conf[idx]['M']/(self.NA*1E3)
			qtr = (2*numpy.math.pi*m*self.kB*t/self.h**2)**1.5

			# . external rotational partition function
			tr = [tmp for tmp in conf[idx]['Tr'] if tmp != 0.0]
			qro = 1.0/conf[idx]['sym']
			if len(tr) == 3: qro *= numpy.math.pi**0.5 *(t**3 /(tr[0]*tr[1]*tr[2]))**0.5
			elif len(tr) == 2 or len(tr) == 1: qro *= t/tr[0]

			# . internal rotational partition function
			qfr = 1.0
			for v in rotor: qfr *= (t/rotor[v])**0.5

			qho = 0.0; qht = 0.0
			for prop in conf:
				# . Boltzmann factor
				Pb = numpy.exp(-(prop['cSPE']-emin)*627.509/(0.002*t))

				# . RRHO partition function
				tmp = 1.0
				for v in sorted(prop['v0']):
					if v > 0.0: tmp *= 1/(1-numpy.exp(-v/(self.kBw*t)))
				qho += tmp*Pb

				# . high-temperature approx. (class.)
				tmp = 1.0
				for v in prop['rotor']: tmp *= self.kBw*t/v
				qht += tmp*Pb
			if not rotor: qanh = qho
			else: qanh = qho*numpy.tanh(qfr/qht)

			# . electronic partition function
			qel = conf[idx]['spin']

			# . add to dict
			Q[t] = qtr*qro*qanh*qel
		return Q

	## @brief	unitless free enthalpy nasa polynomial
	## @param	T	temperature
	## @param	p1	parameter 1
	## @param	p2	parameter 2
	## @param	p3	parameter 3
	## @param	p4	parameter 4
	## @param	p5	parameter 5
	## @param	p6	parameter 6
	## @param	p7	parameter 7
	## @return	unitless free energy
	## @version	2016/02/16:
	#
	def nasaGRT(self, T, p1, p2, p3, p4, p5, p6, p7):
		GRT = [	p1*(1-numpy.log(T)),
			-p2*numpy.power(T, 1)/2.0,
			-p3*numpy.power(T, 2)/6.0,
			-p4*numpy.power(T, 3)/12.0,
			-p5*numpy.power(T, 4)/20.0,
			+p6*numpy.power(T, -1),
			-p7]
		return sum(GRT)

	## @brief	unitless enthalpy nasa polynomial
	## @param	T	temperature
	## @param	p1	parameter 1
	## @param	p2	parameter 2
	## @param	p3	parameter 3
	## @param	p4	parameter 4
	## @param	p5	parameter 5
	## @param	p6	parameter 6
	## @param	p7	parameter 7
	## @return	unitless enthalpy
	## @version	2016/02/16:
	#
	def nasaHRT(self, T, p1, p2, p3, p4, p5, p6):
		HRT = [	p1,
			p2*numpy.power(T, 1)/2.0,
			p3*numpy.power(T, 2)/3.0,
			p4*numpy.power(T, 3)/4.0,
			p5*numpy.power(T, 4)/5.0,
			p6*numpy.power(T, -1)]
		return sum(HRT)

	## @brief	fit NASA polynomials to thermochemistry
	## @param	Smi	SMILES of species
	## @param	Q	partition function
	## @param	Presure	to compute ideal gas density
	## @param	T1	low-temperature limit
	## @param	T2	transition to high-temperature
	## @param	T3	high-temperature limit
	## @return	parameters and maximum parameter uncertainty
	## @version	2016/02/16:
	#		fits NASA polynomials to free enthalpies
	#		computed from parition functions. Note that a
	#		minimum amount of datapoints is required to
	#		fit the 14 low-/high-temperature parameters.
	#		The fitting procedure is working via overlap
	#		at low-temperature and at the transition
	#		temperature.
	#
	## @version	2016/02/29:
	#		partition function computation moved to fitNASA
	#		and high-temperature limit + temperature increment
	#		added (Q removed from input)
	#
	def fitNASA(self, Smi, Pressure=1E5, T1=300.0, T2=1000.0, T3=3000.0, dT=10.0):	
		# . compute partition function
		if ':' in Smi: self.log.printIssue(Text='harvesting.fitNASA: Reaction SMILES detected, skipping ...', Fatal=False); return False
		if T1 > T3: self.log.printIssue(Text='harvesting.fitNASA: You cannot use lowest temperature larger than highest temperature !!!', Fatal=True)
		if T1 > T2: self.log.printIssue(Text='harvesting.fitNASA: You cannot use an initial temperature larger than the transition temperature !!!', Fatal=True)
		if Smi in self.Q: Q = self.Q[Smi]
		else:
			T = sorted([298.15] + [T1+i*dT for i in range(int((T3-T1)/dT)+1)])
			Q = self.calcPartFunc(Smi=Smi, T=T)
			self.Q[Smi] = Q
		if len(Q) < 14: self.log.printIssue(Text='harvesting.fitNASA: insufficient number of partition function data points. Increase number of temperatures', Fatal=True)

		# . find low-/high-temperature reference temperatures
		idx = [T.index(T1), 0]
		tmp = [abs(t-T2) for t in T]
		idx[1] = tmp.index(min(tmp))

		# . overlap range
		n = int(200.0/dT)
		if idx[1]+n > len(T) or idx[1]-n < 0: self.log.printIssue(Text='harvesting.fitNASA: Temperature range for partition function too small !!!', Fatal=True)
		Tlo = T[idx[0]:idx[1]+n]
		Thi = T[idx[1]-n:]

		# . standard enthalpy of formation
		conf = self.db.getConf(Smi=Smi)
		energy = [prop['cSPE'] for prop in conf]
		idx = energy.index(min(energy))
		element = {1: 0, 6: 0, 7: 0, 8: 0}
		for atomid in conf[0]['geo']: # geo={id:[type,[x,y,z]]}
			atype = conf[0]['geo'][atomid][0]
			if atype not in element: element[atype] = 1
			else: element[atype] += 1
		if Smi in self.Hf0: Hstd = self.Hf0[Smi]
		else:
			dH = conf[idx]['Htherm'] -sum([element[6]*self.Ha0['[C]'],
				element[8]*self.Ha0['[O]'],
				element[7]*self.Ha0['[N]'],
				element[1]*self.Ha0['[H]']])
			dH *= 2625.49962*1E3	# J/mol
			Hstd = dH +sum([element[6]*self.Hf0['[C]'],
				element[8]*self.Hf0['[O]'],
				element[7]*self.Hf0['[N]'],
				element[1]*self.Hf0['[H]']])

		# . initial fit for computing the enthalpy correction
		grt = [-numpy.log(Q[t]*self.R*t/(self.NA*Pressure)) for t in Tlo]
		
		try:
			if len(grt) < 7: self.log.printIssue(Text='harvesting.fitNASA: insufficient number of partition function data points for low-temperature. Increase number of temperatures', Fatal=True)
			p0 = [1E+0, 1E-2, -1E-5, 1E-8, -1E-11, -1E3, 1E0]
			popt, pcov = scipy.optimize.curve_fit(f=self.nasaGRT, xdata=Tlo, ydata=grt, p0=p0)
			
			hstd = self.nasaHRT(T=298.15, p1=popt[0], p2=popt[1], p3=popt[2], p4=popt[3], p5=popt[4], p6=popt[5])
			hcorr = Hstd/(self.R*298.15) -hstd

			# . low-temperature fit
			grt = [hcorr*298.15/t -numpy.log(Q[t]*self.R*t/(self.NA*Pressure)) for t in Tlo]
			p0 = popt
			aopt,acov = scipy.optimize.curve_fit(f=self.nasaGRT, xdata=Tlo, ydata=grt, p0=p0)

			# . high-temperature fit
			grt = [hcorr*298.15/t -numpy.log(Q[t]*self.R*t/(self.NA*Pressure)) for t in Thi]
			if len(grt) < 7: self.log.printIssue(Text='harvesting.fitNASA: insufficient number of partition function data points for high-temperature. Increase number of temperatures', Fatal=True)
			p0 = popt
			bopt, bcov = scipy.optimize.curve_fit(f=self.nasaGRT, xdata=Thi, ydata=grt, p0=p0)
		except RuntimeError as exc:
			p, merr = ([],[])
			self.log.printIssue(Text='NASA Fit was not successful. Terminated with message:\n%s' %exc.args[0], Fatal=False)
		else:
			# . final parameters
			p = numpy.hstack([bopt, aopt])
			merr = max(max(numpy.sqrt(numpy.diag(acov))), max(numpy.sqrt(numpy.diag(bcov))))
			self.NASA[Smi] = [p, element, (T1, T2, T3), merr]

		return p, merr

	## @brief	modified Arrhenius equation
	## @param	T	list of temperatures
	## @param	A	pre-exponential factor / reaction
	## @param	n	exponent / -
	## @param	Ea	activation energy / K
	## @return	lnkarr	log-rate constant
	## @version	2016/02/16:
	#		T0 = 1K
	#
	## @version	2016/06/22:
	#		ln(A) -> lnA
	#
	def lnkArr(self, T, lnA, n, Ea):
		lnkarr = lnA +n*numpy.log(T) -Ea/T
		return lnkarr

	## @brief	compute and fit rate constants via mod.Arr.
	## @version	2016/02/29:
	#
	## @version	2016/07/05:
	#		molecules -> mol conversion
	#
	def fitArr(self, Reac, T1=300.0, T3=3000.0, dT=10.0):
		# . calculate partition functions
		T = [T1+i*dT for i in range(int((T3-T1)/dT)+1)]
		ER = 0.0
		rsmi = Reac.split(':')[0].split(',')
		for smi in rsmi:
			conf = self.db.getConf(Smi=smi)
			emin = min([prop['cSPE'] for prop in conf])
			ER += emin
			if smi not in self.Q:
				self.Q[smi] = self.calcPartFunc(Smi=smi, T=T)
		QTS = self.calcPartFunc(Smi=Reac, T=T)

		# . get activation energy
		conf = self.db.getConf(Smi=Reac)	# check for reactions
		ETS = min([prop['cSPE'] for prop in conf])
		ER = 0.0
		for spec in Reac.split(':')[0].split(','):
			tmp = self.db.getConf(Smi=spec)
			ER += min([prop['cSPE'] for prop in tmp])
		Ea = (ETS-ER)*2625.4995	# kJ/mol

		# . calculate rates
		#   (1/s; cm3/mol*s)
		k = []
		for t in T:
			QR = 1.0
			for smi in rsmi:
				if t not in self.Q[smi]: self.log.printIssue(Text='harvesting.fitArr: Use same T1, T3 in fitNASA and fitArr to allow reusing partition function computations. Computation will be skipped !!!', Fatal=False)
				else: QR *= self.Q[smi][t]
			k.append((self.NA*1E6)**(len(rsmi)-1) *self.kB*t*QTS[t]*numpy.exp(-Ea*1000.0/(self.R*t))/(self.h*QR))

		# . fit data
		n0 = numpy.log(abs(k[0]/k[-1] *numpy.exp(1000.0*Ea/self.R *(1/T[0] -1/T[-1]))))/numpy.log(T[0]/T[-1])
		lnA0 = numpy.log(abs(k[0]/(T[0]**n0 *numpy.exp(-1000.0*Ea/(self.R*T[0])))))
		try:
			popt, pcov = scipy.optimize.curve_fit(f=self.lnkArr, xdata=T, ydata=numpy.log(k), p0=[lnA0, n0, Ea*1000.0/self.R])
		except RuntimeError as exc:
			popt, merr = ([],[])
			self.log.printIssue(Text='Arrhenius Fit was not successful. Terminated with message:\n%s' %exc.args[0], Fatal=False)
		else:
			merr = max(numpy.sqrt(numpy.diag(pcov)))
			self.Arr[Reac] = [popt, merr]

		return popt, merr

	## @brief	fit rate constants of barrierless reactions via mod.Arr.
	## @version	2016/03/08:
	#		if a single temperature value is available:
	#			A = k1
	#		if two temperature values are avaibalbe:
	#			n = ln(k2/k1)/ln(T2/T1)
	#			A = k1/T1^n
	#		if three or more temperature values are available:
	#			use scipy.optimize.curve_fit()
	#
	def fitBarrierless(self, Reac):
		T, k, klo, kup, n = self.db.getBarrierlessData(Reac=Reac)
		if len(T) == 1:
			popt = numpy.array([numpy.log(k[0]), 0.0, 0.0]); merr = 0.0
		elif len(T) == 2:
			n = numpy.log(k[1]/k[0])/numpy.log(T[1]/T[0])
			lnA = n*numpy.log(k[0]/T[0])
			popt = numpy.array([lnA, n, 0.0]); merr = 0.0
		else:
			n0 = numpy.log(k[1]/k[0])/numpy.log(T[1]/T[0])
			lnA0 = n0*numpy.log(k[0]/T[0])
			lnk = [numpy.log(k[i]) for i in range(len(k))]
			try:
				popt, pcov = scipy.optimize.curve_fit(f=self.lnkArr, xdata=T, ydata=lnk, p0=[lnA0, n0, 0.0])
			except RuntimeError as exc:
				popt, merr = (numpy.array([]),[])
				self.log.printIssue(Text='Arrhenius Fit was not successful. Terminated with message:\n%s' %exc.args[0], Fatal=False)
			else:
				try: merr = max(numpy.sqrt(numpy.diag(pcov)))
				except ValueError: merr = 1E6
		self.Arr[Reac] = [popt, merr]

		return popt, merr


	## @brief	write NASA thermo-file
	## @param	Filename	name of thermo-file
	## @return	True
	## @version	2016/02/29:
	#		write thermo-file in NASA format to enable thermochemistry
	#		for later numerical simulations. Species are identified via
	#		their stoichiometric formula and an unique ID. The respective
	#		SMILES is supplied at the end of the very first line.
	#
	def writeNASA(self, Filename='Default.therm'):
		writer = open(Filename, 'w')
		writer.write('!version: ChemTraYzer-Mechanism %s\n' %(time.asctime(time.gmtime(time.time()))))
		writer.write('!website: https://sourceforge.net/projects/chemtrayzer/\n')
		writer.write('\n')
		writer.write('thermo\n')
		writer.write('300.00 3000.00 1000.00\n')
		sidx = 0
		for smi in sorted(self.NASA):
			sidx += 1
			param = self.NASA[smi][0]
			element = self.NASA[smi][1]
			nc = element[6]; nh = element[1]; no = element[8]; nn = element[7]
			t1, t2, t3 = self.NASA[smi][2]
			merr = self.NASA[smi][3]
			if nc > 99 or nh > 99 or no > 99 or nn > 99: self.log.printIssue(Text='harvesting.writeNASA: NASA format does not allow for molecule identifiers with more than 18 symbols. At least one molecule is too large to match this condition.', Fatal=True)
			if sidx > 99999: self.log.printIssue(Text='harvesting.writeNASA: NASA format dose not allow for molecule identifiers with more than 18 symbols. Tne number of molecules exceeds the representation threshold of 99999 different species.', Fatal=True)
			spec = 'C%dH%dO%dN%d-%05d' %(nc, nh, no, nn, sidx)
			writer.write('%-18s CTA16c%4dh%4do%4dn%4dg   %7.02f  %7.02f  %7.02f      1 ! %.02f %% max.err. / %s\n' %(spec, nc, nh, no, nn, t1, t3, t2, merr, smi))
			tmp = []
			for p in param[0:5]:
				if p < 0.0: tmp.append('%.08e' %(p))
				else: tmp.append(' %.08e' %(p))
			writer.write(''.join(tmp)+'    2\n')
			tmp = []
			for p in param[5:10]:
				if p < 0.0: tmp.append('%.08e' %(p))
				else: tmp.append(' %.08e' %(p))
			writer.write(''.join(tmp)+'    3\n')
			tmp = []
			for p in param[10:]:
				if p < 0.0: tmp.append('%.08e' %(p))
				else: tmp.append(' %.08e' %(p))
			writer.write(''.join(tmp)+'                   4\n')
		writer.write('end\n')
		writer.close()

		return True

	## @brief	write chemistry file
	## @param	Filename	name of chem-file
	## @return	True
	## @version	2016/02/09:
	#		write chemistry file in ChemKin format to enable
	#		rate constant computation for later numerical
	#		simulations. Species are identified according to
	#		the nomeclature used in the NASA thermo-file.
	#
	def writeArr(self, Filename='Default.chem'):
		writer = open(Filename, 'w')
		writer.write('\n')
		writer.write('----------------------------------------\n')
		writer.write('elements\n')
		writer.write('c\n')
		writer.write('h\n')
		writer.write('o\n')
		writer.write('n\n')
		writer.write('end\n')
		writer.write('species\n')
		sidx = 0; tmp = {}
		for smi in sorted(self.NASA):
			sidx += 1
			element = self.NASA[smi][1]
			nc = element[6]; nh = element[1]; no = element[8]; nn = element[7]
			spec = 'C%dH%dO%dN%d-%05d' %(nc, nh, no, nn, sidx)
			tmp[smi] = spec
			writer.write('%-18s\t! %s\n' %(spec, smi))
		writer.write('end\n')
		writer.write('reactions\n')
		for reac in sorted(self.Arr):
			popt = self.Arr[reac][0]
			merr = self.Arr[reac][1]
			# . check if species is available
			try:
				out = [	'+'.join(tmp[smi] for smi in reac.split(':')[0].split(',')),
					'+'.join(tmp[smi] for smi in reac.split(':')[1].split(','))]
				writer.write('! %s\n' %(reac))
				writer.write('<=>'.join(out)+' %.03e %.04f %.02f ! %.02f %% max.err.\n' %(numpy.exp(popt[0]), popt[1], popt[2]*1.987, merr))	# 1/s / cm3/mol*s, -, cal/mol
			except KeyError:
				self.log.printIssue(Text='Reaction %s was Arrhenius fitted but a reactant is missing in in NASA polynomials.' %(reac), Fatal=False)
		writer.write('end\n')
		writer.close()
		return  True

# Validate partition function


if __name__ == '__main__':
	import sys
	import log as Log

	log = Log.Log(Width=70)

	###	HEADER
	#
	#
	text = ['[options] <source folder> <database> [-files <filename>] [-type <file extension>] [-fail <behavior on failure>] [-norm <behavior on success>]',
		'options:',
		'-files: name of g09 output files. Filenames not starting with this string will be ignored. (default="tmp_")',
		'-type: extension of g09 output files. Files with other extensions will be ignored. (default="log")',
		'-fail: action taken, if harvesting of a file fails. Possible options are: keep, delete. (default="keep")',
		'-norm: action taken, if harvesting of a file succeeds. Possible options are: keep, delete. (default="keep")',
		'For multiple folders, the input line can be repeated. Specify options for all folders first.']
	log.printHead(Title='ChemTraYzer - Gaussian Data Harvesting', Version='2016-02-22', Author='Malte Doentgen, LTT RWTH Aachen University', Email='chemtrayzer@ltt.rwth-aachen.de', Text='\n\n'.join(text))

	###	INPUT
	#
	# . defaults
	folder     = []
	database   = {}
	files	= {0: 'tmp_'}
	type	= {0: 'log'}
	fail	= {0: 'keep'}
	norm	= {0: 'keep'}

	if len(sys.argv) > 2: argv = sys.argv
	else:
		log.printComment(Text=' Use keyboard to type in species and parameters. The <return> button will not cause leaving the input section. Use <strg>+<c> or write "done" to proceed.', onlyBody=False)
		argv = ['input']; d = False
		while not d:
			try: tmp = input('')
			except: d = True
			if tmp.lower() == 'done': d = True
			else:
				if '!' in tmp: argv += tmp[:tmp.index('!')].split()
				else: argv += tmp.split()

	###	INTERPRET INPUT
	#
	i = 1
	name = 0
	while i < len(argv):
		if argv[i].startswith('-'):
			# . option
			try: arg = argv[i+1]
			except IndexError:
				log.printIssue('argument missing for option "%s". Will be ignored.' %argv[i])
			else:
				if arg.startswith('-'):
					log.printIssue('argument missing for option "%s". Will be ignored.' %argv[i])
				elif argv[i].lower() == '-files':
					files[name] = arg
					i += 1
				elif argv[i].lower() == '-type':
					type[name] = arg
					i += 1
				elif argv[i].lower() == '-fail':
					fail[name] = arg
					i += 1
				elif argv[i].lower() == '-norm':
					norm[name] = arg
					i += 1
				else:
					log.printIssue('unknown option "%s". Will be ignored.' %argv[i])
					i += 1
		else:
			# . QM folder name
			name = argv[i]
			# . DB name
			try: db = argv[i+1]
			except IndexError:
				log.printIssue('database missing for folder "%s".' %name, Fatal=True)
			else:
				if db.startswith('-'):
					log.printIssue('database missing for folder "%s".' %name, Fatal=True)
				elif not os.access(name, os.R_OK):
					log.printIssue('No access to "%s" (read).' %(name), Fatal=True)
				else:
					folder.append(name)
					database[name] = db
					i += 1
					if not os.access(db, os.W_OK):
						log.printIssue('No access to "%s" (write). Will try to create.' %(db), Fatal=False)
		i += 1

	for f in folder:
		if f not in files: files[f] = files[0]
		if f not in type: type[f] = type[0]
		if f not in fail: fail[f] = fail[0]
		if f not in norm: norm[f] = norm[0]

	### FEEDBACK INPUT
	#
	text = ['You specified the following:']
	for f in folder:
		text += ['harvest from '+f+'/'+files[f]+'*'+type[f]]
		text += [' use '+database[f]+' as database']
		text += [' '+fail[f]+' on failure']
		text += [' '+norm[f]+' on success']
	log.printComment('\n'.join(text))

	###	PROCESS FILES
	#
	dbcounter = {}
	filecounter = 0
	donecounter = 0
	failcounter = 0
	barrierless = {}
	for f in folder:
		# . print folder name
		log.printComment('Folder: %s' %f, onlyBody=True)
		# . get harvesting instance
		harv = Harvesting(Database=database[f])
		# . harvest folder
		done,failed,total,tmp = harv.harvestFolder(Folder=f,Files=files[f],Type=type[f],Fail=fail[f],Done=norm[f])
		donecounter += done
		failcounter += failed
		filecounter += total
		if done > 0: dbcounter[database[f]] = 0
		for smile in tmp:
			barrierless[smile] = tmp[smile]
		# . test DB on reference species (they may not be there)
		#   it is user responsability, if they are missing (produces wrong NASA polynomials)
		harv.getHa0()

	log.printComment('%d of %d results written to %d database(s), %d failures' %(donecounter,filecounter,len(dbcounter),failcounter))
	log.printComment('The following barrierless reactions were ignored:\n'+'\n'.join('%s' %(smile) for smile in barrierless))

