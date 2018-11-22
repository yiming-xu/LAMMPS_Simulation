##################################################################
# The MIT License (MIT)                                          #
#                                                                #
# Copyright (c) 2018 RWTH Aachen University, Malte Doentgen,     #
#                    Felix Schmalz, Leif Kroeger                 #
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
## @file	simulation.py
## @author	Malte Doentgen, Felix Schmalz, Leif Kroeger
## @date	2018/07/13
## @brief	methods for controlling LAMMPS/ReaxFF simulations
#		in python and connecting it with processing
#		routines
#
##################################################################
#	!!!	ATTENTION	!!!				 #
##################################################################
#								 #
#	THE FUNCTION 'submitQM' MUST BE ADJUSTED TO YOUR	 #
#	SYSTEM (go to line 1359)				 #
#								 #
##################################################################

import sys
import os
import subprocess
import random
import numpy
from collections import deque

import openbabel
import lammps

import log as Log
import dbhandler
import processing

## @class	Simulation
## @brief	contains functions for setting up simulation input
#		and interacting with the simulation software (LAMMPS)
## @version	2015/07/24:
#		Methods for initializing LAMMPS simulations, including
#		generation of molecule xyz files, LAMMPS data files
#		and a ready-to-run LAMMPS object
#
## @version	2015/11/17:
#		generates folders 'spec.dat' and 'reac.dat' in which
#		preoptimized geometries and their potential energies
#		are stored. For later use, species and transition states
#		are stored as spec/ts_%06d.xyz. Lists for identifying
#		which ID stands for which spec/ts are given (spec.list
#		and reac.list)
#
## @version	2016/09/01:
#		small overhaul, see individual version infos
#
class Simulation:
	## @brief	constructor
	## @param	lmp	LAMMPS object
	## @param	Mode	version of the LAMMPS code
	## @version	2015/07/24:
	#
	## @version	2016/01/21:
	#		added sqlite database
	#
	## @version	2016/09/01:
	#		made lammps logs accessible
	#
	## @version	2018/03/01:
	#		number of atoms N is saved
	#
	def __init__(self, lmp=False, Mode='mpi', QMfolder='QM', DBname='chemtrayzer.sqlite', Quiet=False, Submit=True):
		# . static objects
		self.log = Log.Log(Width=70)
		self.Val = {1: 1, 2: 0, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0, 16: 6, 17: 1, 18: 0}
		self.m   = {1: 1.0090, 2: 4.0026, 6: 12.0110, 7: 14.0067, 8: 15.9999, 9: 18.9984, 10: 20.1797, 16: 32.06, 17: 35.45, 18: 39.948}
		self.mid = dict((v,k) for k,v in self.m.items()) # reverse mass-element mapping
		self.symbol = {1: 'H', 2: 'He', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 16: 'S', 17: 'Cl', 18: 'Ar'}
		self.convert = {'density': 6.022E-7,	# mol/m^3 -> molecules/A^3
				'Eff': 1E7}		# g/mol *A^2 /fs^2 -> J/mol
		self.spin = {	'O=O': 3,
				'[C]': 3,
				'[O]': 3,
				'[H]': 2,
				'[N]': 4,
				}

		# . openbabel objects
		self.conv = openbabel.OBConversion()
		self.conv.SetInAndOutFormats('smi', 'mdl')
		self.build = openbabel.OBBuilder()

		# . LAMMPS objects
		self.quiet = Quiet
		self.lammpslogfile = 'lammps.log'
		self.preoptlogfile = 'preopt.log'
		self.dummylogfile  = 'lammps.dummy.log' # overwritten by "log" commands
		self.cmdargs = ['-log',self.dummylogfile]
		if self.quiet: self.cmdargs += ['-screen','none']
		if lmp: self.lmp = lmp
		else: self.lmp = lammps.lammps(name=Mode,cmdargs=[arg for arg in self.cmdargs])
		self.mode = Mode
		self.datafile = ''
		self.identify = {}
		self.element = {}
		self.masses = {}
		self.dt = 0.1
		self.velocityflag = True

		# . processing dummy
		self.proc = False
		self.reaction = {}

		# . output management
		self.idx = {'Spec': 0, 'TS': 0}
		self.reg = {'Spec': {}, 'TS': {}}

		# . system properties
		self.N = 0
		self.v = 0.0
		self.T = 0.1

		# . list of transitions states
		self.ts = []

		# . QM
		self.methods = ['g3mp2', 'g3mp2', 'mp2/6-31G(d)']
		self.command = ['opt=(calcall,tight) freq=(HinderedRotor) int=ultrafine scf=xqc '+self.methods[0]+' symmetry=loose',
				'opt=(calcall,tight,ts,noeigentest,nofreeze) freq int=ultrafine scf=xqc '+self.methods[1]+' symmetry=loose',
				'irc=(calcfc,stepsize=10,maxpoints=20,maxcycle=80) '+self.methods[2]+' geom=check guess=read scf=xqc']
		self.nproc = 1
		self.mem = 10000 # in MB; 1GB == 128MW in G09
		self.optsteps = 10000 # in MB; 1GB == 128MW in G09
		self.QMcount = 0
		self.qmjobs = []
		self.qmfolder = QMfolder
		self.submit = Submit

		# . database
		#cmd = [' '.join([c for c in tmp.split() if 'opt' not in c and 'freq' not in c and 'int' not in c and 'scf' not in c and 'maxdisk' not in c and 'symmetry' not in c]) for tmp in self.command]
		self.db = dbhandler.Database(Name=DBname,QMmethods=self.methods)

	## @brief	generates 3D information
	## @param	Spec	SMILES for which a geometry will
	#			be generated
	## @return	list of atomic numbers and positions
	## @version	2015/07/24:
	#		To overcome lacking database information
	#		the openbabel 3D generation algorithms
	#		are used for creating xyz data.
	#
	def genStruct(self, Spec):
		mol = openbabel.OBMol()
		if not self.conv.ReadString(mol, Spec):
			self.log.printIssue('The species "%s" could not be recognized by openBabel.' %Spec, Fatal=True)
		mol.AddHydrogens()
		self.build.Build(mol)
		struct = []
		for atom in openbabel.OBMolAtomIter(mol):
			struct.append([atom.GetAtomicNum(), atom.x(), atom.y(), atom.z()])
		return struct

	## @brief	read atom info from input LAMMPS-datafile
	#		sets: self.a self.v self.masses self.identify self.element
	## @param	Datafile	path to LAMMPS datafile
	## @return	True
	## @version	2018/02/27:
	#		introduced to start CTY from existing geometry (instead of generating one)
	#
	def readDataFile(self, Datafile):
		reader = open(Datafile, 'r')
		line = reader.readline()
		natoms = int(reader.readline().strip().split()[0])
		line = reader.readline()
		ntypes = int(reader.readline().strip().split()[0])
		line = reader.readline()
		a = float(reader.readline().strip().split()[1])
		line = reader.readline()
		line = reader.readline()
		line = reader.readline()
		line = reader.readline()
		line = reader.readline()
		identify = {}
		element = {}
		masses = {}
		# . loop atom types in 'Masses' section
		#   match atom masses from data file with those of self.m
		for i in range(ntypes):
			tmp = reader.readline().strip().split() # e.g. '1 1.009' -> Hydrogen
			#print tmp
			tidx = int(tmp[0]); data_mass = float(tmp[1])
			#print tidx,data_mass
			for z,m in self.m.items():
				if abs(data_mass-m)/m < 0.05: # within 5% interval
					identify[tidx] = z
					element[z] = tidx
					masses[tidx] = m
					break
			# . a type in the data file (tidx,data_mass) could not be found in self.m
			if tidx not in identify:
				self.log.printIssue(Text='Simulation.readDataFile: The atom type %d with mass %f is not implemented. Cannot continue.' %(tidx,data_mass), Fatal=True)
		# . check if velocities were specified in data file
		#   then set velocity flag (to false) -> no random velocities
		line = reader.readline()
		line = reader.readline() # 'Atoms'
		line = reader.readline()
		for i in range(natoms):
			line = reader.readline()
		line = reader.readline()
		if reader.readline().strip() == 'Velocities':
			self.velocityflag = False

		self.masses = masses
		self.identify = identify
		self.element = element
		self.a = a
		self.v = a**3
		self.N = natoms

		return True


	## @brief	generates simulation input either from a
	#		database or via 3D generation
	## @param	Species		initial chemical composition
	#				given as SMILES and mole
	#				fractions: {<spec>: <x>}
	#				in Kelvin
	## @param	Density		initial density of the system
	#				in mol/m^3
	## @param	N		number of molecules
	## @param	Database	path to database, if False
	#				use openbabel 3D generator
	## @param	Packmol		path to packmol program
	## @param	Datafile	filename for data-file, used as
	#				input for LAMMPS
	## @return	True
	## @version	2018/03/15:
	#		pack in smaller box, so that tolerance is met with PBC
	## @version	2018/03/01:
	#		start CTY optionally via a predefined input box
	#		with positions and velocities
	## @version	2015/07/24:
	#		Uses packmol by Martiinez et al., J. Comput.
	#		Chem. 30(2009), 30, 2157-2164 to generate an
	#		initial molecule distribution based on the
	#		xyz information stored in 'Database'. If the
	#		latter parameter is False, openbabel will be
	#		used for 3D generation.
	#
	def genDat(self, Species=None, Density=0, N=0, Packmol='', Datafile='data.inp', Seed=random.randint(1,99999)):
		self.datafile = Datafile
		# . read box from input
		if (not Species or Density == 0 or N == 0 or Packmol == ''):
			if not os.path.exists(Datafile):
				self.log.printIssue(Text='Simulation.genDat: Neither a datafile specified nor species, density and N. Cannot build Box.', Fatal=True)
			else:
				self.readDataFile(Datafile)
		# . generate data file with packmol
		else:
			# . check whether packmol is available
			if not Packmol or Packmol.split('/')[-1] != 'packmol' or not os.path.isfile(Packmol):
				self.log.printIssue(Text='Simulation.genDat: The packmol software package was not found at the specified location:\n\t%s' %Packmol, Fatal=True)

			# . read database and check species
			#   there will always be a sql database
			struct = {}
			tmp = []
			for spec in Species:
				geometry = self.db.getGeometry(Smile=spec)
				if geometry == []:
					geometry = self.genStruct(Spec=spec)
					tmp.append(spec)
				struct[spec] = geometry
			if tmp: self.log.printIssue(Text='Simulation.genDat: No database entry for the following species:\n\t%s\n... generating molecular geometries from connectivity information instead.' %('\n\t'.join(tmp)), Fatal=False)

			# . write xyz files for species
			idx = 1
			for spec in sorted(Species):
				writer = open('S%d.xyz' %idx, 'w')
				writer.write('%d\n' %len(struct[spec]))
				writer.write('%s\n' %spec)
				for atom in struct[spec]:
					writer.write('%d %.04f %.04f %.04f\n' %(atom[0], atom[1], atom[2], atom[3]))
				writer.close()
				idx += 1

			# . normalize chemical composition
			total = 0.0
			for spec in Species:
				total += Species[spec]
			for spec in Species:
				Species[spec] /= total

			# . microscopic concentrations
			n = dict((spec, int(round(Species[spec]*N))) for spec in Species)
			self.v = sum(n[spec] for spec in Species)/(Density*self.convert['density'])
			a = self.v**(1.0/3.0)

			# . generate packmol input
			packfile = 'pack'
			writer = open(packfile+'.inp', 'w')
			writer.write('seed %d\n' %Seed)
			writer.write('tolerance 2.0\n')
			writer.write('filetype xyz\n')
			writer.write('output %s.out\n' %packfile)
			writer.write('\n')
			idx = 1
			for spec in sorted(Species):
				writer.write('structure S%d.xyz\n' %(idx))
				writer.write('  number %d\n' %n[spec])
				writer.write('  inside box %.02f %.02f %.02f %.02f %.02f %.02f\n' %(-a/2.0+1, -a/2.0+1, -a/2.0+1, a/2.0-1, a/2.0-1, a/2.0-1))
				writer.write('end structure\n')
				writer.write('\n')
				idx += 1
			writer.close()

			# . execute packmol
			inp = open(packfile+'.inp', 'r')
			out = open(packfile+'.log', 'w')
			subprocess.call([Packmol], stdin=inp, stdout=out)

			# . convert to LAMMPS data file
			idx = 1; tidx = 1
			data = []
			reader = open(packfile+'.out', 'r')
			reader.readline(); reader.readline()
			for line in reader:
				words = line.split('\n')[0].split()
				if int(words[0]) not in self.element:
					self.element[int(words[0])] = tidx
					self.masses[tidx] = self.m[int(words[0])]
					self.identify[tidx] = int(words[0])
					tidx += 1
				atmidx = self.element[int(words[0])]
				data.append('%d %d 0 %.06f %.06f %.06f' %(idx, atmidx, float(words[1]), float(words[2]), float(words[3])))
				idx += 1
			reader.close()
			self.N = idx-1

			# . write data file
			writer = open(self.datafile, 'w')
			writer.write('\n')
			writer.write(' %d atoms\n' %len(data))
			writer.write('\n')
			writer.write(' %d atom types\n' %len(self.element))
			writer.write('\n')
			writer.write(' %.02f %.02f xlo xhi\n' %(-a/2.0, a/2.0))
			writer.write(' %.02f %.02f ylo yhi\n' %(-a/2.0, a/2.0))
			writer.write(' %.02f %.02f zlo zhi\n' %(-a/2.0, a/2.0))
			writer.write('\n')
			writer.write(' Masses\n')
			writer.write('\n')
			for key in sorted(self.masses):
				writer.write(' %d %.04f\n' %(key, self.masses[key]))
			writer.write('\n')
			writer.write(' Atoms\n')
			writer.write('\n')
			for line in data:
				writer.write(line+'\n')
			writer.write('\n')
			writer.close()

			# . cleanup
			for i in range(1,len(Species)+1): os.remove('S%d.xyz' %i)
			if os.path.isfile(packfile+'.inp'): os.remove(packfile+'.inp')
			if os.path.isfile(packfile+'.out'): os.remove(packfile+'.out')
			if os.path.isfile(packfile+'.log'): os.remove(packfile+'.log')

	## @brief	initializes LAMMPS simulation
	## @param	Temperature	NVT system temperature
	## @param	Damping		damping constant for Nose-Hoover
	#				thermostat
	## @param	dt		timestep for integration using
	#				the velocity-verlet algorithm
	## @param	Seed		random seed for velocity
	#				distribution
	## @param	FField		force field file used for
	#				simulation
	## @param	Memory		Used to request define amount of
	#				memory
	## @param	Safezone	Used to set max.limit for
	#				exceeding the memory
	## @return	LAMMPS object ready for batched run
	## @version	2015/07/25:
	#		initialize the LAMMPS simulation by setting
	#		atom style, force field style and parameters,
	#		minimization and random velocity distribution.
	#
	## @version	2015/11/13:
	#		preoptimizer is initialized here too
	#
	## @version	2016/03/10:
	#		pe/atom compute added
	#
	## @version	2016/09/01:
	#		LAMMPS main logfile made available
	#
	## @version	2018/03/01:
	#		introduced to start CTY with existing geometry and velocity
	#
	def initLAMMPS(self, Temperature, Damping=100.0, dt=0.1, Seed=random.randint(1,99999), FField='ffield.reax', Memory=False, Safezone=False):
		# . check for ffield file
		if not FField or not os.path.isfile(FField):
			self.log.printIssue(Text='Simulation.initLAMMPS: Cannot find force field file. Please supply correct path to file.', Fatal=True)
		self.ffield = FField

		# . init system
		self.log.printComment(Text='initializing simulation box', onlyBody=False)
		self.lmp.command('log %s' %self.lammpslogfile)
		self.lmp.command('units real')
		self.lmp.command('atom_style charge')
		self.lmp.command('atom_modify map hash')
		self.lmp.command('read_data %s' %self.datafile)

		# . init force field
		ff = 'pair_style reax/c NULL'
		if Memory: ff += ' mincap %d' %Memory
		if Safezone: ff += ' safezone %.02f' %Safezone
		self.lmp.command(ff)
		tmp = dict((self.element[key],key) for key in sorted(self.element))
		self.lmp.command('pair_coeff * * %s %s' %(FField, ' '.join(self.symbol[tmp[key]] for key in sorted(tmp))))

		# . setup parameters
		self.dt = dt
		self.lmp.command('fix qeq all qeq/reax 1 0.0 10.0 1e-6 reax/c')
		self.lmp.command('neighbor 2 bin')
		self.lmp.command('neigh_modify every 10 delay 0 check no')
		self.lmp.command('timestep %.02f' %self.dt)

		# . setup ouput: each 1ps for default timestep of 0.1fs
		self.lmp.command('thermo_style custom step temp press etotal pe ke')
		self.lmp.command('thermo 10000')

		# . minimization
		#self.lmp.command('minimize 1.0e-20 1.0e-20 10000 100000')

		# . velocities
		self.T = Temperature
		if self.velocityflag:
			self.lmp.command('velocity all create %.01f %d mom yes rot yes' %(self.T, Seed))
			print('set random velocities')

		# . setup
		self.lmp.command('reset_timestep 0')
		self.lmp.command('fix 2 all nvt temp %.01f %.01f %.01f' %(self.T, self.T, Damping))
		self.lmp.command('compute ape all pe/atom')

		# . return LAMMPS object
		return self.lmp

	## @brief	on-the-fly processing of a batched trajectory
	## @param	BatchSteps	number of simulation steps to
	#				be performed in this chunk
	## @param	DumpFile	filename for writting dump
	## @param	BondFile	filename for writting
	#				connectivity
	## @param	Keys		indicators for dump data to be
	#				written to 'DumpFile'
	## @param	Freq		frequency of writting ouput
	## @return	filenames of dump and bond file and list of
	#		dump keys
	## @version	2015/07/25:
	#		run a LAMMPS simulation for 'BatchSteps' steps
	#		as a single process and write bond orders as
	#		well as other information specified in 'Keys'
	#		to the temporary files 'BondFile' and 'DumpFile'
	#		respectively.
	#
	def runBatch(self, BatchSteps, DumpFile='dump.tmp', BondFile='bond.tmp', Keys=['id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz'], Freq=200):
		self.lmp.command('dump dmp all custom %d %s %s' %(Freq, DumpFile, ' '.join(Keys)))
		self.lmp.command('fix bnd all reax/c/bonds %d %s' %(Freq, BondFile))
		self.lmp.command('run %d' %BatchSteps)
		self.lmp.command('undump dmp')
		self.lmp.command('unfix bnd')
		return DumpFile, BondFile, Keys

	## @brief	pre-optimizes molecular structures at the ReaxFF
	#		level of theory
	## @param	Geometry	list of atoms
	## @param	Mode		1: uncontraint minimization,
	#				2: freeze transition atoms (TS)
	## @param	Active		list of atom ids of active atoms
	#				(not used for Mode == 1)
	## @return	optimized geometry, energy, principal moments of
	#		inertia
	## @version	2015/11/13:
	#		preoptimizer (self.preopt) is filled with atoms
	#		of the molecule (Mode=1) or TS (Mode=2) to be
	#		optimized. TS optimizing uses a simple freeze
	#		of the <Active> atoms.
	#
	#		format of <Geometry>: [[atomic number, [x, y, z]
	#					[...], ...]
	#
	## @version	2015/12/11:
	#		boundary is periodic, atoms are centered inside the
	#		box after minimization.
	#
	## @version	2016/01/23:
	#		Atoms are created already with displacement, so that
	#		inertia calculation will succeed. Assuming the energy
	#		minimization not to move atoms outside the box.
	#
	## @version	2016/06/25:
	#		create_atoms etc is replaced by read_data to avoid
	#		occasional 'nan' values in forces
	#
	## @version	2016/09/01:
	#		quiet flag disables all pre-optimizer screen output
	#		moved atom displacement at front
	#		last pre-optimize will have a log file
	#		removed check on pe==None (minimize error)
	#
	def preOptimize(self, Geometry, Mode=1, Active=False):
		if not Geometry: return 0.0, [], [0.0,0.0,0.0]
		if not self.quiet: self.log.printComment(Text='PREOPTIMIZER -- START: Mode = %d' %(Mode), onlyBody=False)

		a = self.v**(1.0/3.0)
		natoms = len(Geometry)

		# . write LAMMPS datafile
		datafile = open('preopt.data', 'w')
		datafile.write('\n')
		datafile.write(' %d atoms\n' %natoms)
		datafile.write('\n')
		datafile.write(' %d atom types\n' %(len(self.element)))
		datafile.write('\n')
		datafile.write(' %.2f %.2f xlo xhi\n' %(-a/2.0,a/2.0))
		datafile.write(' %.2f %.2f ylo yhi\n' %(-a/2.0,a/2.0))
		datafile.write(' %.2f %.2f zlo zhi\n' %(-a/2.0,a/2.0))
		datafile.write('\n')
		datafile.write(' Masses\n')
		datafile.write('\n')
		for key in sorted(self.element):
			datafile.write(' %d %f\n' %(self.element[key], self.masses[self.element[key]]))
		datafile.write('\n')
		datafile.write(' Atoms\n')
		datafile.write('\n')
		atomid = 0
		for atom in Geometry:
			atomid += 1
			datafile.write('%d %d 0 %f %f %f\n' %(atomid,self.element[atom[0]],atom[1][0],atom[1][1],atom[1][2]) )
		datafile.write('\n')
		datafile.close()

		# . initialize preoptimizer
		self.preopt = lammps.lammps(name=self.mode,cmdargs=[arg for arg in self.cmdargs])
		self.preopt.command('log %s' %self.preoptlogfile)
		self.preopt.command('units real')
		self.preopt.command('atom_style charge')
		self.preopt.command('read_data preopt.data')
		# . ReaxFF
		self.preopt.command('pair_style reax/c NULL mincap 2000 safezone 1.5')
		tmp = dict((self.element[key],key) for key in sorted(self.element))
		self.preopt.command('pair_coeff * * %s %s' %(self.ffield, ' '.join(self.symbol[tmp[key]] for key in sorted(tmp))))
		self.preopt.command('fix qeq all qeq/reax 1 0.0 10.0 1e-6 reax/c')
		# . parameters
		self.preopt.command('neighbor 2 bin')
		self.preopt.command('neigh_modify every 10 delay 0 check no')
		# . thermo output
		self.preopt.command('thermo_style custom step temp press etotal pe ke')
		self.preopt.command('thermo 100')
		self.preopt.command('compute pe all pe')

		# . freeze TS + minimize
		if Mode == 2:
			if not Active:
				self.log.printIssue(Text='Simlation.preOptimize: Mode=2 requested in preoptimizer (TS optimization), but no active atoms supplied. Performing unconstraint minimization (no saddle point will be found !!!)', Fatal=False)
			else:
				self.preopt.command('group active id %s' %(' '.join('%d' %atom for atom in Active)))
				self.preopt.command('fix freeze active setforce 0.0 0.0 0.0')
		self.preopt.command('minimize 1.0e-20 1.0e-20 10000 100000')

		# . minimized geometry
		coords = self.preopt.gather_atoms('x', 1, 3)
		optGeometry = [[Geometry[i][0], numpy.array(coords[3*i:3*i+3])] for i in range(natoms)]

		# . center by geometry
		#   find biggest gap between atoms for each coordinate
		#   and 'move' boundary to middle of the gap.
		#   displacements ensure that all atoms have the maximum
		#   possible distance to the walls, while maintaining the
		#   closest distance to each other. This makes a com
		#   calculation possible. The wall is expected at +-A/2.
		displace = numpy.array([0.0, 0.0, 0.0])
		for i in range(3):
			x = sorted([atom[1][i] for atom in optGeometry])
			dx = [ x[j+1] - x[j] for j in range(natoms-1) ] + [ x[0]+a - x[-1] ]
			displace[i] = a/2.0 - (x[dx.index(max(dx))] + max(dx)/2.0)

		# . center of mass
		mass = 0.
		com = numpy.array([0.0, 0.0, 0.0])
		for atom in optGeometry:
			m = self.m[atom[0]]
			atom[1] += displace
			atom[1][atom[1]>a/2.0] -= a #remap
			mass += m
			com += atom[1]*m
		com /= mass

		# . center by com and calc inertia tensor
		Ixx, Iyy, Izz, Ixy, Iyz, Ixz = 0.,0.,0.,0.,0.,0.
		for atom in optGeometry:
			atom[1] -= com
			m, [x, y, z] = self.m[atom[0]], atom[1]
			Ixx += m*(y**2+z**2)
			Iyy += m*(x**2+z**2)
			Izz += m*(x**2+y**2)
			Ixy -= m*x*y
			Iyz -= m*y*z
			Ixz -= m*x*z
			atom[1] = [x, y, z] # array->list

		# . retrieve energy and compute principal moments of
		#   inertia
		pe = round(self.preopt.extract_compute('pe', 0, 0), 2)
		I = numpy.array([[Ixx, Ixy, Ixz],[Ixy, Iyy, Iyz],[Ixz, Iyz, Izz]])
		Ip, v = numpy.linalg.eig(I)

		# . return potential energy and optimized geometry
		if not self.quiet: self.log.printComment(Text='PREOPTIMIZER -- DONE', onlyBody=False)
		self.preopt.close()

		return pe, optGeometry, sorted(Ip)

	## @brief	extract transition state strcuture and preopt
	## @param	Timestep	timestep of the reaction event
	## @param	Reaction	array containing the atom ids
	#				of the reacting molecules
	## @param	Buffer		contains the dumped data of
	#				previous timesteps
	## @param	Active		atom ids of atoms actively
	#				involved in the reaction
	## @param	Write		True: save transition states
	#				False: only potential energy
	## @version	2015/11/13:
	#		extract molecular structures of all reacting
	#		molecules at the exact reaction event timestep.
	#
	## @version	2016/01/11:
	#		QM boolean added
	#
	def extractTransitionState(self, Timestep, Reaction, Buffer, Active, Write=False, QM=False):
		if Timestep != Buffer[0]:
			self.log.printIssue(Text='Simulation.extractTransitionState: Transition state cannot be extracted from Buffer, since timestep was not found. Skipping ...', Fatal=False)
			return False

		# . extract geometry and find active atoms
		molmass = 0.0
		geometry = []
		active = []
		idx = 0
		for mol in Reaction[0]:
			for atom in mol[0]:
				idx += 1
				type = self.identify[Buffer[1][atom][0]]
				geometry.append([type, Buffer[1][atom][1]])
				if atom in Active: active.append(idx)
				molmass += self.m[type]

		# . preoptimize
		pe, opt, ip = self.preOptimize(Geometry=geometry, Mode=2, Active=active)
		a = ','.join(sorted(mol[1] for mol in Reaction[0]))
		b = ','.join(sorted(mol[1] for mol in Reaction[1]))
		tmp = [a+':'+b, b+':'+a]
		if tmp[0] in self.ts: reac = tmp[0]
		elif tmp[1] in self.ts: reac = tmp[1]
		else: reac = tmp[0]; self.ts.append(reac)

		# . write
		if Write:
			if reac not in self.reg['TS']:
				self.idx['TS'] += 1
				self.reg['TS'][reac] = self.idx['TS']
				tsfile = open('reac.dat/ts_%06d.xyz' %(self.reg['TS'][reac]), 'w')
				tsfile.write('')
				tsfile.close()
				reaclist = open('reac.dat/reac.list', 'a')
				reaclist.write('ts_%06d.xyz: %s\n' %(self.reg['TS'][reac], reac))
				reaclist.close()
			tmp = repr(len(opt))+'\n'
			tmp += '%s: %.02f %s\n' %(reac, pe, ' '.join(['%.02f' %(i) for i in ip]))
			tmp += '\n'.join('%3s %10.06f %10.06f %10.06f' %(atom[0], atom[1][0], atom[1][1], atom[1][2]) for atom in opt)+'\n'
			tsfile = open('reac.dat/ts_%06d.xyz' %(self.reg['TS'][reac]), 'a')
			tsfile.write(tmp)
			tsfile.close()

		# . submit a QM job if TS reaction
		if QM:
			# . retrieve reactants and products via unconstraint minimization
			self.fullMinimization(Timestep=Timestep, Buffer=Buffer, Molecules=Reaction[0]+Reaction[1], Write=Write, QM=QM)

			if self.isBarrierless(reac):
				# . barrierless reaction (B)
				self.log.printComment('Reaction %s is barrierless.' %reac, onlyBody=True)
				brlesslist = open('reac.dat/reac.list.b', 'a')
				brlesslist.write('%s %f\n' %(reac,molmass))
				brlesslist.close()
			else:
				# . normal reaction (TS)
				if not self.db.barrierReactionInDB(Smile=reac,Mode='deep',FFenergy=pe,FFinertia=ip):
					self.submitQM(Smi=reac, Geometry=opt, FFenergy=pe, Ip=ip, Active=active)
					self.log.printComment('Reaction %s is new to the DB. Started job No. %d' %(reac,self.QMcount), onlyBody=True)
					self.qmjobs.append(('%s/job_tmp_%d.sh' %(self.qmfolder,self.QMcount), reac, Timestep))

		return True

	## @brief	extract and preopt the molecular structures
	#		of all molecules
	## @param	Timestep	timestep of the reaction event
	## @param	Buffer		contains the dumped data of
	#				previous timesteps
	## @param	Molecules	list of molecules, which are
	#				lists of atoms in turn
	## @param	Write		True: save transition states
	#				False: only potential energy
	## @version	2015/11/13:
	#		in order to find all/most conformers, all
	#		molecular structures have to be extracted and
	#		energy-minimized periodically.
	#
	## @version	2016/01/11:
	#		QM boolean added
	#
	## @version	2016/09/01.
	#		removed check on pe==None (minimize error)
	#		(cf. preoptimizer function)
	#
	def fullMinimization(self, Timestep, Buffer, Molecules, Write=False, QM=False):
		for mol in Molecules:
			# . preoptimize
			geometry = []
			for atom in mol[0]:
				geometry.append([self.identify[Buffer[1][atom][0]], Buffer[1][atom][1]])
			pe, opt, ip = self.preOptimize(Geometry=geometry, Mode=1, Active=False)
			spec = mol[1]

			# . check against database (with FF values)
			inDB = self.db.speciesInDB(spec,pe,ip)

			# . write
			if not inDB and Write:
				if spec not in self.reg['Spec']:
					self.idx['Spec'] += 1
					self.reg['Spec'][spec] = self.idx['Spec']
					specfile = open('spec.dat/spec_%06d.xyz' %(self.reg['Spec'][spec]), 'w')
					specfile.write('')
					specfile.close()
					speclist = open('spec.dat/spec.list', 'a')
					speclist.write('spec_%06d.xyz: %s\n' %(self.reg['Spec'][spec], spec))
					speclist.close()
				tmp = repr(len(opt))+'\n'
				tmp += '%s: %.02f %s\n' %(spec, pe, ' '.join(['%.02f' %(i)for i in ip]))
				tmp += '\n'.join('%3s %10.06f %10.06f %10.06f' %(atom[0], atom[1][0], atom[1][1], atom[1][2]) for atom in opt)+'\n'
				specfile = open('spec.dat/spec_%06d.xyz' %(self.reg['Spec'][spec]), 'a')
				specfile.write(tmp)
				specfile.close()

			if not inDB and QM:
				# . submit QM
				self.submitQM(Smi=mol[1], Geometry=opt, FFenergy=pe, Ip=ip, Active=False)
				self.log.printComment('Species %s is new to the DB. Started job No. %d' %(mol[1],self.QMcount), onlyBody=True)
				self.qmjobs.append(('%s/job_tmp_%d.sh' %(self.qmfolder,self.QMcount), mol[1], Timestep))

		return True

	## @brief	check if reaction is barrierless
	## @param	Reac	reaction SMILES
	## @return	true if barrierless, false if not
	## @version	2016/02/04:
	#		TS optimizations are useless for barrieless
	#		reactions. The present function checks if a
	#		reaction is barrierless (true) or not (false).
	#
	#		The check is very weak to avoid treating
	#		reactions as barrierless which are not.
	#
	## @version	2016/02/14:
	#		existing data in DB tables is used to
	#		make the check stronger
	#
	def isBarrierless(self, Reac):
		# . first look up existing DB tables,
		#   since their data is backed with QM
		if self.db.barrierlessReactionInDB(Smile=Reac):
			return True
		if self.db.barrierReactionInDB(Smile=Reac,Mode='quick'):
			return False

		# . check spin multiplicity
		reac = Reac.split(':')[0].split(',')
		prod = Reac.split(':')[1].split(',')
		spin = [[], []]
		for r in reac:
			if r in self.spin: s = self.spin[r]
			else:
				mol = openbabel.OBMol()
				self.conv.ReadString(mol, r)
				s = mol.GetTotalSpinMultiplicity()
			spin[0].append(s)
		for p in prod:
			if p in self.spin: s = self.spin[p]
			else:
				mol = openbabel.OBMol()
				self.conv.ReadString(mol, p)
				s = mol.GetTotalSpinMultiplicity()
			spin[1].append(s)

		nr = len(spin[0]); np = len(spin[1])
		if nr != np:
			if (sorted(spin[0]) == [2, 2] or sorted(spin[1]) == [2, 2]) and (sorted(spin[1]) == [1] or sorted(spin[0]) == [1]):
				return True
			elif (sorted(spin[0]) == [2, 3] or sorted(spin[1]) == [2, 3]) and (sorted(spin[1]) == [2] or sorted(spin[0]) == [2]):
				return True
			elif (sorted(spin[0]) == [3, 3] or sorted(spin[1]) == [3, 3]) and (sorted(spin[1]) == [3] or sorted(spin[0]) == [3]):
				return True
			elif (sorted(spin[0]) == [2, 2, 3] or sorted(spin[1]) == [2, 2, 3]) and (sorted(spin[1]) == [1] or sorted(spin[0]) == [1]):
				return True
		return False

	## @brief	compute molecular energies
	## @param	Data		contains the dumped data of
	#				the a single timestep
	## @param	Molecules	list of molecules, which are
	#				lists of atoms in turn
	## @return	energy		list kinetic and potential
	#				energies
	## @version	2016/03/10:
	#		use position, velocity, and potential energy
	#		information to compute the total energy of
	#		the current chemical composition. The kinetic
	#		energy is used rather the LAMMPS ke/atom since
	#		the translational, rotational, and vibrational
	#		contributions are separately computed.
	#
	## @version	2018/03/01:
	#		print out number of rotational degrees of freedom
	#		to allow equilibrium temperature calculation
	## @version	2018/07/10:
	#		correct energies for linear molecules
	def compMolEn(self, Data, Molecules):
		energy = []
		for mol in Molecules:
			# . receive data for molecule
			geometry = []; velocity = []; pe = []; m = []
			eki = 0.0
			for atom in mol[0]:
				geometry.append(numpy.array(Data[atom][1]))
				velocity.append(numpy.array(Data[atom][2]))
				pe.append(Data[atom][3])
				m.append(self.m[self.identify[Data[atom][0]]])

			# . center atoms and remap
			natoms = len(mol[0])
			displace = numpy.array([0.0, 0.0, 0.0])
			a = self.v**(1.0/3.0)
			# . find biggest gap between atoms for each
			#   coordinate and move boundary to middle of
			#   the gap
			for i in range(3):
				x = sorted([atom[i] for atom in geometry])
				dx = [ x[j+1] - x[j] for j in range(natoms-1) ]
				dx += [ x[0]+a - x[-1] ]
				displace[i] = a/2.0 - (x[dx.index(max(dx))] + max(dx)/2.0)

			# . center-of-mass properties
			com = numpy.array([0.0, 0.0, 0.0])
			vcom = numpy.array([0.0, 0.0, 0.0])
			for i in range(natoms):
				eki += m[i]*numpy.dot(velocity[i],velocity[i])
				geometry[i] += displace
				geometry[i][geometry[i]>a/2.0] -= a #remap
				com += geometry[i]*m[i]
				vcom += velocity[i]*m[i]
			M = sum(m)
			com /= M; vcom /= M

			# . angular momentum
			ang_mom = numpy.array([0.0, 0.0, 0.0])
			for i in range(natoms): # L = sum r x p 
				tmp = numpy.cross(geometry[i]-com, velocity[i])
				ang_mom += m[i]*tmp

			ero = 0.0
			# . moment of inertia
			if natoms == 1:
				frot = 0
			else:
				frot = 3
				mom_inert = numpy.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
				for i in range(natoms):
					pos = geometry[i]-com
					mom_inert[0,0] += m[i]*(pos[1]**2 +(pos[2])**2)
					mom_inert[1,1] += m[i]*(pos[0]**2 +(pos[2])**2)
					mom_inert[2,2] += m[i]*(pos[0]**2 +(pos[1])**2)
					mom_inert[0,1] -= m[i]*pos[0]*pos[1]
					mom_inert[0,2] -= m[i]*pos[0]*pos[2]
					mom_inert[1,2] -= m[i]*pos[1]*pos[2]
				# I is symmetric!
				mom_inert[1,0] = mom_inert[0,1]
				mom_inert[2,0] = mom_inert[0,2]
				mom_inert[2,1] = mom_inert[1,2]
				# not stable:
				#ang_vel = numpy.linalg.solve(mom_inert, ang_mom)
				# stable:
				# Due to symmetry of I we can use eigh
				w, v = numpy.linalg.eigh(mom_inert)
				# project angular momentum:
				LP = numpy.dot(v,ang_mom) # T*L
				W = w
				# Search for small eigenvalues: 
				for i in range(0,3):
					if w[i] < 1E-8:
						LP = numpy.delete(LP,i) # delete element
						W  = numpy.delete(w, i)
						frot-=1
				I_inv = numpy.diag(numpy.reciprocal(W))
				ero = 0.5*numpy.dot(numpy.transpose(LP),numpy.dot(I_inv,LP))*self.convert['Eff']

			# . kinetic energies
			#   note: g/mol *A^2 /fs^2 -> J/mol
			eki = 0.5*eki*self.convert['Eff']
			etr = 0.5*M*numpy.linalg.norm(vcom)**2 *self.convert['Eff']
			evi = eki-etr-ero

			# . store energies
			epo = sum(pe)*4184	# kcal/mol -> J/mol
			energy.append([etr, ero, evi, epo, frot])

		return energy

	## @brief	perform integrated simulation
	## @param	Time		simulation time in ns
	## @param	WorkFile	reaction event storage
	## @param	dt		timestep
	## @param	BathSteps	number of simulation setps per
	#				batch (i.e. between two data
	#				processings)
	## @param	Freq		bond and dump frequency
	## @param	Storage		memory for buffered reading in
	#				bytes
	## @param	Static		static bond order cutoff
	## @param	Recrossing	active (False) / deactive (True)
	#				recrossing filter (cf. -norec)
	## @param	RecSteps	number of steps to be considered
	#				in the recrossing filter
	## @param	Skip		number of steps to be skipped
	#				between two steps
	## @param	Periodic	step-frequency for periodic full
	#				minimization of all molecules
	#				(for conformer search)
	## @param	Write		True: save transition states
	#				False: only potential energy
	## @param	Event		reaction event (given as SMILES)
	#				for exiting simulation
	## @param	QM		generate and submit g09 jobs
	## @param	Close		if True, close lammps object
	## @return	reaction-dict
	## @version	2015/11/13:
	#		simulate a batched ReaxFF simulation (requires
	#		an initialized LAMMPS object: self.lmp).
	#		Evaluate changes in chemical composition on-the-
	#		fly and extract geometries for further usage.
	#
	## @version	2015/11/17:
	#		Reaction event based exit condition added. Searchs for <Event> in reactions,
	#		thus can also take 'A,B:' as argument and aborts if any reaction of 'A' and
	#		'B' are detected.
	#
	## @version	2016/01/11:
	#		QM boolean added
	#
	## @version	2016/01/21:
	#		sqlite database
	#
	## @version	2016/02/24:
	#		Temperature in Workfile
	#
	## @version	2016/02/27:
	#		moved proc to self.proc and added self.reaction
	#		data. self.reaction is returned by this function
	#		Close bool added to allow for keeping lmp open
	#		for subsequent use.
	#
	## @version	2016/03/10:
	#		compute pe/atom added
	#
	## @version	2016/03/18:
	#		work file is flushed every line (every reaction)
	#		so analyzing is possible while simulating
	#
	## @version	2016/04/05:
	#		importance of parameters: Freq, BatchSteps, Time
	#		-> correct BatchSteps to allow for integer-division via Freq
	#		-> correct Time to allow for integer-division via BatchSteps
	#
	## @version	2016/05/24:
	#		optional dump of molecular energies (from compMolEn)
	#
	## @version	2016/09/01:
	#		log -> self.log typo
	#		integrated reference species job here, since it uses the QM switch
	#		removed timestep check for dumpbuffer
	#		added experimental output, which batch is running
	#		made barrierless reaction list conditional to Write flag
	#		remove lammps log remains
	#		added check if write frequency greater than batch steps
	#		passing recrossing time instead of steps
	#		added info about started qm jobs
	#
	## @version	2016/12/13:
	#		replaced dump buffer dictionary by a queue to save time
	#
	def integSim(self, Time, WorkFile=None, dt=0.1, BatchSteps=10000, Freq=200, Storage=10000, Static=0.5, Recrossing=False, RecSteps=None, RecTime=2.0, Skip=1, Periodic=5000, Write=False, Event=False, QM=False, Close=True):
		# . init
		self.log.printComment(Text='starting integrated simulation', onlyBody=False)

		# . event based abort (prep)
		if Event: reacEvent = ','.join(sorted(Event.split(':')[0].split(',')))+':'+','.join(sorted(Event.split(':')[1].split(',')))

		# . generate temporary filenames
		dumpFile = 'dump.tmp'
		bondFile = 'bond.tmp'

		# . create workfile
		#   important: temperature, volume and timestep are stored here
		if WorkFile is None: WorkFile = 'default.reac'
		worker = open(WorkFile, 'w', 1)
		worker.write('ON THE FLY ... %.1f %.1f %.2f\n' %(self.T,self.v,self.dt))

		# . if not existing create species and reaction storage, else remove existing data
		if Write:
			# . folders
			if not os.path.exists('spec.dat'): os.makedirs('spec.dat')
			else:
				self.log.printIssue(Text='Simulation.integSim: removing files in spec.dat ...' , Fatal=False)
				for f in os.listdir('spec.dat/.'):
					os.remove('spec.dat/'+f)
			if not os.path.exists('reac.dat'): os.makedirs('reac.dat')
			else:
				self.log.printIssue(Text='Simulation.integSim: removing files in reac.dat ...' , Fatal=False)
				for f in os.listdir('reac.dat/.'):
					os.remove('reac.dat/'+f)
			# . files
			open('spec.dat/spec.list', 'w').close()
			open('reac.dat/reac.list', 'w').close()
			# . molecule dump and their energy
			molfile = open('spec.dat/moldump.out', 'w').close()
			molfile2 = open('spec.dat/energy.out', 'w')
			molfile2.write('timestep temperature trans rot vib pot ftrans frot fvib\n')
			molfile2.close()
			# . file to store barrierless reactions
			open('reac.dat/reac.list.b', 'w').close()

		# . if not existing create QM storage, else remove existing data
		if QM:
			if not os.path.exists(self.qmfolder): os.makedirs(self.qmfolder)
			else:
				self.log.printIssue(Text='Simulation.integSim: removing files in %s ...' %self.qmfolder, Fatal=False)
				for f in os.listdir('%s/.' %self.qmfolder):
					os.remove(self.qmfolder+'/'+f)

		# . set simulation parameters
		self.lmp.command('reset_timestep 0')
		dumpKeys = ['id', 'type', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'c_ape']
		# . [Time] = ns , [dt] = fs
		nsteps = int(Time*1E6/dt)
		nchunks = nsteps/BatchSteps

		# . correct for too large write frequency
		if Freq > BatchSteps:
			Freq = BatchSteps
			self.log.printComment(Text='Write frequency reduced to number of batch steps to ensure at least two written steps per batch.', onlyBody=False)

		# . correct for 'uneven' batch length
		diff = BatchSteps%Freq
		if diff:
			BatchSteps -= diff
			log.printComment(Text='Number of steps per batch reduced from %d to %d to allow for integer division by the write frequency.' %(BatchSteps+diff, BatchSteps), onlyBody=False)

		# . correct for 'uneven' simulation time
		if nsteps%BatchSteps > 0:
			nchunks += 1
			log.printComment(Text='Number of simulation timesteps cannot be expressed as an integer multiple of the batch size. Thus, the total simulation time was extended to %.04f ns by adding a single batch' %(nchunks*BatchSteps*dt/1E6), onlyBody=False)

		# . set recrossing steps
		if RecSteps: recSteps = RecSteps
		else: recSteps = int(RecTime*1E3/(dt*Freq))
		if recSteps < 1:
			self.log.printIssue(Text='Simulation.integSim: Recrossing steps less than 1, setting has no effect.', Fatal=False)
			recSteps = 1

		# . coordinates of last double recrossing period
		dumpBuffer = deque([[-1]],2*recSteps+1)

		# . batched simulations start here:
		i = 0; mininthisbatch = False
		while i < nchunks:
			# . run batch
			self.log.printComment(Text='Running batch %d of %d (%d steps) ...' %(i+1,nchunks,BatchSteps), onlyBody=True)
			self.runBatch(BatchSteps=BatchSteps, DumpFile=dumpFile, BondFile=bondFile, Keys=dumpKeys, Freq=Freq)

			# . process output
			bondReader = open(bondFile, 'r')
			dumpReader = open(dumpFile, 'r')
			if self.proc: self.proc = processing.Processing(Processing=self.proc, Reader=bondReader, Start=0)
			else: self.proc = processing.Processing(Reader=bondReader, Identify=self.identify, Start=0, Storage=Storage, Static=Static)
			# . generate dump step data
			dumpSteps = self.proc.dumpGenerator(Reader=dumpReader, Read=Storage)
			# . discard first entry
			dumpStep = next(dumpSteps); dumpStep = next(dumpSteps)
			# . build index for dumped atom attributes
			head = dumpStep[7].split()[2:]
			idx = [head.index(k) for k in dumpKeys]

			# . processing batch
			done = False
			while not done:
				# . process connectivity data
				timestep, tmp = self.proc.processStep(Recrossing=Recrossing, Steps=recSteps, Skip=Skip, Close=False)
				endofbatch = (timestep < 0)
				timestep = abs(timestep)
				if endofbatch: done = True

				# . process dumped data
				#   only if new
				if timestep != dumpBuffer[-1][0]:
					geo = {}
					for line in dumpStep[8:-1]:
						words = line.split()
						geo[int(words[idx[0]])] = [int(words[idx[1]]), [float(words[idx[2]]), float(words[idx[3]]), float(words[idx[4]])], [float(words[idx[5]]), float(words[idx[6]]), float(words[idx[7]])], float(words[idx[8]])]
					dumpBuffer.append([timestep,geo])

				# . process reactions
				#   negative timesteps contain reactions not yet to process
				#   keep them until next batch, or flush them at last batch
				if not endofbatch or i == nchunks-1:
					# . process reactions
					for t in sorted(tmp):
						if t not in self.reaction: self.reaction[t] = []
						for r in tmp[t]:
							reac = ','.join(mol[1] for mol in r[0])+':'+','.join(mol[1] for mol in r[1])
							self.reaction[t].append(r)
							self.log.printBody(Text=repr(t)+':'+reac, Indent=2)
							worker.write(repr(t)+':'+reac+'\n')
							# . extract TS, start QM Job, only if real reaction
							if (Write or QM) and t in self.proc.active:
								pos = (timestep-t)/int(Freq)+1
								self.extractTransitionState(Timestep=t, Reaction=r, Buffer=dumpBuffer[-pos], Active=self.proc.active[t], Write=Write, QM=QM)
							# . check for reaction event (also parts of it)
							if Event and reacEvent in reac and t in self.proc.active:
								self.log.printComment(Text='%s: reaction event occured, aborting ...' %reacEvent, onlyBody=False)
								i = nchunks
								done = True
				# . since reactions of previous batch
				#   are copied to new 'self.proc' object
				#   the recrossing filter includes
				#   all previous reactions
				else: pass

				# . periodic full-min, spec files, start QM Job
				#   save cpu time and file quota by restricting to end of batch
				if (timestep%Periodic == 0):
					mininthisbatch = True
				if mininthisbatch and endofbatch:
					mininthisbatch = False
					self.log.printComment(Text='Timestep %d: starting full min' %timestep, onlyBody=True)
					molecules = [mol for mol in self.proc.molecule if mol[0]]
					if Write or QM:	self.fullMinimization(Timestep=timestep, Buffer=dumpBuffer[-1], Molecules=molecules, Write=Write, QM=QM)

				# . prepare next dump step, same routine as in processStep()
				try:
					for s in range(Skip): dumpStep = next(dumpSteps)
				except StopIteration: pass

			if Write:
				# . compute molecular energies; this is done
				#   once per batch to avoid the huge
				#   computational effort which comes along
				#   energy : list, 4-tuple for each molecule
				molecules = [mol for mol in self.proc.molecule if mol[0]]
				nmol = len(molecules)
				energies = self.compMolEn(Data=dumpBuffer[-1][1], Molecules=molecules)

				# . dump energies per molecule (id smiles etr ero evi epo natoms atoms)
				molid = 0
				molfile = open('spec.dat/moldump.out', 'a')
				molfile.write('TIMESTEP: %d\n' %timestep)
				for m in range(min(len(energies),len(molecules))):
					mol = molecules[m]
					eng = energies[m]
					molfile.write('%d %s %g %g %g %g %d %s \n' %(molid,mol[1],eng[0],eng[1],eng[2],eng[3],len(mol[0]),'/'.join([str(a) for a in mol[0]])))
					molid += 1
				molfile.close()
				# . dump energy of all molecules (timestep temperature etr ero evi epo)
				temperature = self.lmp.extract_compute("thermo_temp",0,0)
				eng0 = eng1 = eng2 = eng3 = 0.0; frot = 0
				for m in range(len(energies)):
					eng0 += energies[m][0]
					eng1 += energies[m][1]
					eng2 += energies[m][2]
					eng3 += energies[m][3]
					frot += energies[m][4]
				molfile2 = open('spec.dat/energy.out', 'a')
				molfile2.write('%d %g %g %g %g %g %d %d %d\n' %(timestep, temperature, eng0, eng1, eng2, eng3, 3*(nmol-1), frot, 3*self.N-3*nmol-frot))
				#Ttrans = eng0 /           (3*(nmol-1)) * 2 / 8.3144598
				#Trot   = eng1 /                   frot * 2 / 8.3144598
				#Tvib   = eng2 / (3*self.N-3*nmol-frot) * 2 / 8.3144598
				molfile2.close()

			bondReader.close()
			dumpReader.close()
			i += 1
		if Close:
			self.lmp.close()
			self.proc.closeSystem()
			maxtime = max(self.proc.reaction)
			for reac in self.proc.reaction[maxtime]:
				tmp = ','.join(r[1] for r in reac[0])+':'+','.join(p[1] for p in reac[1])
				worker.write(repr(maxtime)+':'+tmp+'\n')
		worker.close()

		# . sumbmit QM jobs for reference species (H2,H2O,CH4,N2)
		#   pre-optimize the four molecules and check whether they are already existent
		if QM: self.submitJobsForReferenceSpecies()

		# . a little clean up
		if os.path.isfile(self.dummylogfile): os.remove(self.dummylogfile)

		# . show started jobs
		if QM:
			text = 'jobfiles:\n' + '\n'.join('%s %s %s' %job for job in self.qmjobs)
			self.log.printComment(Text=text)
			qmjobinfo = open('qmjobinfo.out', 'w')
			qmjobinfo.write(text)
			qmjobinfo.close()

		return self.reaction

	## @brief	create and submit g09 input file
	## @param	Smi		SMILES of species or reaction
	## @param	Geometry	xyz format
	## @param	FFenergy	force field energy for ref.
	## @param	Ip		force field principial moments
	#				of intertia for ref.
	## @return	True
	## @version	2016/01/06:
	#		submits a quantum mechanical job. Needs to be
	#		defined by user
	#
	## @version	2016/04/07:
	#		added changable qm folder
	#
	def submitQM(self, Smi, Geometry, FFenergy, Ip, Active):
		# . get charge and spin
		mol = openbabel.OBMol()
		if ':' in Smi: tmp = Smi.split(':')[0].replace(',', '.')
		else: tmp = Smi
		self.conv.ReadString(mol, tmp)
		c = mol.GetTotalCharge()
		if tmp in self.spin: s = self.spin[tmp]
		else: s = mol.GetTotalSpinMultiplicity()

		# . write g09 input file
		#   try to find best memory size and core numbers:
		#   memory increases over self.mem above 10 atoms
		#   no of procs ranges from self.proc over 4 and 8 to 12 (with more than 12, 24 or 36 atoms resp.).
		#   maxdisk is always 750 GB
		#   optimization steps are minimum 60, or else 8 times no of atoms
		natoms = len(Geometry)
		memory = self.mem * max(1, natoms/10.0)
		nproc = max( self.nproc, min((natoms/3)/4*4, 12) ) # 1,4,8,12
		maxdisk = 750
		optsteps = max(60, 8*natoms)
		optcart = 'opt=cartesian' if any(i < 0.4 for i in Ip) else ''

		self.QMcount += 1
		writer = open('%s/tmp_%d.com' %(self.qmfolder,self.QMcount), 'w')

		if ':' in Smi:
			# pre opt
			writer.write('%%nprocshared=%d\n' %(nproc))
			writer.write('%%mem=%dMB\n' %(memory*0.9))
			writer.write('%%chk=tmp_%d.chk\n' %(self.QMcount))
			writer.write('# opt=modredundant hf/tzvp scf=xqc\n')
			writer.write('\n')
			writer.write('REAXFF: %s %.02f %.04f %.04f %.04f\n' %(Smi, FFenergy, Ip[0], Ip[1], Ip[2]))
			writer.write('\n')
			writer.write('%d %d\n' %(c, s))
			for atom in Geometry:
				writer.write(' %s\t%.08f\t%.08f\t%.08f\n' %(self.symbol[atom[0]], atom[1][0], atom[1][1], atom[1][2]))
			writer.write('\n')
			# (add and) freeze coordinates for HF pre-opt
			for atomnr in list(set(Active)):
				writer.write('%d F\n' %(atomnr))
			# main TS opt, Freq and Energy
			writer.write('--Link1--\n')
			writer.write('%%nprocshared=%d\n' %(nproc))
			writer.write('%%mem=%dMB\n' %(memory*0.9))
			writer.write('%%chk=tmp_%d.chk\n' %(self.QMcount))
			writer.write('# geom=check maxdisk=%dGB %s opt=(maxcycles=%d) opt=modredundant %s\n' %(maxdisk, self.command[1], optsteps, optcart))
			writer.write('\n')
			writer.write('REAXFF: %s %.02f %.04f %.04f %.04f\n' %(Smi, FFenergy, Ip[0], Ip[1], Ip[2]))
			writer.write('\n')
			writer.write('%d %d\n' %(c, s))
			writer.write('\n')
			# remove added (and freezed) coordinates, needs opt=modredundant
			for atomnr in list(set(Active)):
				writer.write('%d R\n' %(atomnr))
			# IRC
			writer.write('--Link1--\n')
			writer.write('%%nprocshared=%d\n' %(nproc))
			writer.write('%%mem=%dMB\n' %(memory*0.9))
			writer.write('%%chk=tmp_%d.chk\n' %(self.QMcount))
			writer.write('# %s\n' %(self.command[2]))
			writer.write('\n')
			writer.write('REAXFF: %s %.02f %.04f %.04f %.04f\n' %(Smi, FFenergy, Ip[0], Ip[1], Ip[2]))
			writer.write('\n')
			writer.write('%d %d\n' %(c, s))
			writer.write('\n')
		else:
			writer.write('%%nprocshared=%d\n' %(nproc))
			writer.write('%%mem=%dMB\n' %(memory*0.9))
			writer.write('%%chk=tmp_%d.chk\n' %(self.QMcount))
			writer.write('# maxdisk=%dGB %s opt=(maxcycles=%d) %s\n' %(maxdisk, self.command[0], optsteps, optcart))
			writer.write('\n')
			writer.write('REAXFF: %s %.02f %.04f %.04f %.04f\n' %(Smi, FFenergy, Ip[0], Ip[1], Ip[2]))
			writer.write('\n')
			writer.write('%d %d\n' %(c, s))
			for atom in Geometry:
				writer.write(' %s\t%.08f\t%.08f\t%.08f\n' %(self.symbol[atom[0]], atom[1][0], atom[1][1], atom[1][2]))
			writer.write('\n')
		writer.close()

		# . submit g09 job on your desired system
		#   DEFAULT: submit on local system
		#   NOTE: This part has to be adjusted to your system (e.g. batch submission system
		#         on a supercomputing cluster).
		subprocess.call('g09 < tmp_%d.com > tmp_%d.log &' %(self.QMcount, self.QMcount))

		return self.QMcount

	## @brief	submit QM jobs for H, O, C or N if they are missing in DB
	## @version	2016/09/01:
	#		sumbmit QM jobs for reference species (H,O,C,N)
	#		pre-optimize the four molecules and check whether they are already existing
	#		always send qm jobs for bath gas N2, Ar and He
	#		removed trial to write hardcode values
	#
	#		NOTE: He N2 Ar disabled, due to ffield
	#
	def submitJobsForReferenceSpecies(self):
		self.log.printComment(Text='Creating jobs for reference species ...', onlyBody=False)
		refSpecGeo = {}
		# . C, H and O only if needed
		if 1 in self.element: refSpecGeo['[H]']      = [ [1, [0.0, 0.0, 0.0]] ]
		if 6 in self.element: refSpecGeo['[C]']      = [ [6, [0.0, 0.0, 0.0]] ]
		if 8 in self.element: refSpecGeo['[O]']      = [ [8, [0.0, 0.0, 0.0]] ]
		# . N2, Ar and He always (common bath gas molecules)
		#   introduce new elements if needed
		#refSpecGeo['He']  = [ [2,  [0.0, 0.0, 0.0]] ]
		#refSpecGeo['N#N'] = [ [7,  [0.0, 0.0, 0.372087]], [7, [0.0, 0.0, -0.372087]] ]
		#refSpecGeo['Ar']  = [ [18, [0.0, 0.0, 0.0]] ]
		#for z in [2, 7, 18]:
		#	if z not in self.element:
		#		self.element[z] = len(self.element)+1 # new tidx
		#	self.masses[self.element[z]] = self.m[z]
		# . check against DB
		for smile in refSpecGeo:
			eFF, optGeo, iFF = self.preOptimize(Geometry=refSpecGeo[smile], Mode=1, Active=False)
			if not self.db.speciesInDB(Smile=smile,Energy=eFF,Ip=iFF):
				self.log.printComment('Reference Species %s is missing in DB. Submitting a QM job.' %(smile))
				self.submitQM(Smi=smile,Geometry=optGeo,FFenergy=eFF,Ip=iFF,Active=False)
				self.qmjobs.append(('%s/job_tmp_%d.sh' %(self.qmfolder,self.QMcount), smile, 0))

		return True



#######################################
if __name__ == '__main__':
	log = Log.Log(Width=70)

	###	HEADER
	#
	text = ['<concentration1> <species1> ... -n <amount of molecules> -d <molecule density> -t <time interval> -temp <temperature> -packmol <packmol executable> [-nbatch <number of simulation batches>] [-static <static bond order cutoff>] [-periodic <minimization interval in steps>] [-recrossing <norec>] [-rectime <rectime>] [-storage <read buffer>] [-skip <skip steps>] [-event <abort event>] [-dbase <database>] [-qm <QM boolean>] [-frequency <dump frequency>] [-qmfolder <QM folder>] [-datafile <temporary datafile>] [-ffield <force field file>] [-damping <temperature damping>] [-dt <timestep>] [-memory <memory>] [-safezone <memory safezone>] [-mode <LAMMPS mode>] [-quiet <screen output>] [-seed <random number seed>]',
		'supply concentration followed by SMILES code for each species',
		'required options:',
		'-t: time interval in nanoseconds, also E notation',
		'-temp: temperature in K',
		'required either:',
		'-n: amount of molecules',
		'-d: density in mol/m^3',
		'-packmol: location of packmol executable',
		'or as alternative:',
		'-datafile: path to LAMMPS data-file (default=data.inp)',
		'',
		'simulation options:',
		'-nbatch: length of simulation batch in steps (default=10000)',
		'-static: static bond order cutoff (default=0.5)',
		'-periodic: periodic minimization interval in steps (default=5000)',
		'-recrossing: allow reaction recrossing (default=0)',
		'-rectime: recrossing interval in picoseconds (default=2)',
		'-storage: buffer when reading files in byte (default=10000)',
		'-skip: process every nth step (default=1)',
		'-event: abort if reaction or part of it is detected (default=0)',
		'-dbase: location of the database (default=chemtrayzer.sqlite)',
		'-qm: toggle quantum mechanical optimization on/off (default=1)',
		'-frequency: write frequency of dumpfiles in steps (default=200)',
		'-qmfolder: folder in which QM files are stored (default=QM)',
		'-submit: submit QM jobs after creation (default=1)',
		'',
		'LAMMPS options:',
		'-ffield: ReaxFF parameter-file with the elements C, H, N, O, He and Ar. (default=ffield.reax)',
		'-damping: NVT thermostat damping-constant in steps (default=100)',
		'-dt: integration timestep in femtoseconds (default=0.1)',
		'-memory: memory for reax/c in MB (default=1000)',
		'-safezone: safety factor for reax/c (default=1.2)',
		'-mode: LAMMPS mode (default=serial)',
		'-quiet: switch LAMMPS screen output on/off (default=0)',
		'-seed: random number used for velocity initialization in LAMMPS and molecule distribution in packmol']
	log.printHead(Title='ChemTraYzer - Batched ReaxFF Simulation', Version='2016-03-11', Author='Malte Doentgen, LTT RWTH Aachen University', Email='chemtrayzer@ltt.rwth-aachen.de', Text='\n\n'.join(text))

	###	INPUT
	#
	if len(sys.argv) > 1: argv = sys.argv
	else:
		log.printComment(Text=' Use keyboard to type in species and parameters. The <return> button will not cause leaving the input section. Use <strg>+<c> or write "done" to proceed.', onlyBody=False)
		argv = ['input']; done = False
		while not done:
			try: tmp = input('')
			except: done = True
			if tmp.lower() == 'done': done = True
			else:
				if '!' in tmp: argv += tmp[:tmp.index('!')].split()
				else: argv += tmp.split()

	###	INTERPRET INPUT
	#
	options = {}
	species = {}
	i = 1
	while i < len(argv)-1:
		arg = argv[i]
		opt = argv[i+1]
		if arg.startswith('-'):
			# . option
			if opt.startswith('-'): log.printIssue(Text='Missing value for option "%s". Will be ignored.' %arg, Fatal=False)
			else: options[arg[1:].lower()] = opt; i += 1
		else:
			# . species
			try: species[opt] = float(arg) ; i += 1
			except: log.printIssue(Text='Expected a number for species fraction, got "%s". Will be ignored.' %opt, Fatal=False)
		i += 1

	#if not species:
	#	log.printIssue(Text='no species specified. Exit.', Fatal=True)

	### DEFAULTS
	#
	try: time = float(options['t'])
	except: log.printIssue(Text='t: option expected float. Exit.', Fatal=True)
	try: T = float(options['temp'])
	except: log.printIssue(Text='temp: option expected float. Exit.', Fatal=True)
	try: N = int(options['n'])
	except: N = 0
	try: density = float(options['d'])
	except: density = 0
	try: packmol = options['packmol']
	except: packmol = ''
	try: dfile = options['datafile']
	except: dfile = 'data.inp'
	try: nbatch = int(float(options['nbatch']))
	except: nbatch = 10000
	try: static = float(options['static'])
	except: static = 0.5
	try: periodic = float(options['periodic'])
	except: periodic = 5000
	try: storage = int(float(options['storage']))
	except: storage = 10000
	try: qmfolder = options['qmfolder']
	except: qmfolder = 'QM'
	try: dbase = options['dbase']
	except: dbase = 'chemtrayzer.sqlite'
	try: qm = bool(int(options['qm']))
	except: qm = True
	try: ffield = options['ffield']
	except: ffield = 'ffield.reax'
	try: damping = float(options['damping'])
	except: damping = 100.0
	try: dt = float(options['dt'])
	except: dt = 0.1
	try: recrossing = bool(int(options['recrossing']))
	except: recrossing = False
	try: recTime = float(options['rectime'])
	except: recTime = 2
	try: freq = int(float(options['frequency']))
	except: freq = 200
	try: skip = int(float(options['skip']))
	except: skip = 1
	try: memory = int(float(options['memory']))
	except: memory = 1000
	try: safezone = float(options['safezone'])
	except: safezone = 1.2
	try: mode = options['mode']
	except: mode = 'serial'
	try: quiet = bool(int(options['quiet']))
	except: quiet = False
	try: submit = bool(int(options['submit']))
	except: submit = True
	try: seed = int(options['seed'])
	except: seed = random.randint(1,99999)
	try: event = options['event']
	except: event = False

	# . set up simulation
	log.printComment(Text='SEED: %d' %(seed))
	sim = Simulation(Mode=mode, DBname=dbase, QMfolder=qmfolder, Quiet=quiet, Submit=submit)
	sim.genDat(Species=species, Density=density, N=N, Packmol=packmol, Datafile=dfile, Seed=seed)
	sim.initLAMMPS(Temperature=T, Damping=damping, dt=dt, Seed=seed, FField=ffield, Memory=memory, Safezone=safezone)
	sim.integSim(Time=time, WorkFile=None, dt=dt, BatchSteps=nbatch, Freq=freq, Storage=storage, Static=static, Recrossing=recrossing, RecTime=recTime, Skip=skip, Periodic=periodic, Write=True, Event=event, QM=qm)

