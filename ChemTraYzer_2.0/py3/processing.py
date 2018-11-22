##################################################################
# The MIT License (MIT)                                          #
#                                                                #
# Copyright (c) 2017 RWTH Aachen University, Malte Doentgen,     #
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
## @file	processing.py
## @author	Malte Doentgen, Felix Schmalz
## @date	2017/05/19
## @brief	This file contains all methods required to process
#		data read from ReaxFF Bond Order files.
#
#  Changelog: 2015/05/05 -> 2017/05/19:
#  - Class 'Log' moved to separte file, since it is imported by
#    multiple classes in the ChemTraYzer package
#  - 'self.active' added to keep trace of the atoms actively
#    involved in reactions. This is used for transition state
#    pre-optimization
#  - Initialization updated to account for single-atom molecules
#    and to obtain the correct time-resolution of the bond order
#    file (new function for extracting time-resolution)
#    (BUGFIX: 2015-07-08; 2016-01-29)
#  - First molecular structure initialization no longer via the
#    'chkReactions' function (which was too time-consuming).
#    Initial reactions have no reactants anymore
#  - 'extractBonds' function updated to accelerate processing
#  - 'splitBonds' checks for empty bond array to avoid abort
#    (BUGFIX: 2015-09-29)
#  - 'canonical' gets H and H2 correct now
#    (BUGFIX: 2015-08-31)
#  - 'Filter' parameter of 'filerRecrossing' was not effective
#    inside the function, but outside (in 'processStep'). Now
#    it can be used inside 'filterRecrossing' aswell (was only
#    a problem if 'filterRecrossing' was not used in combination
#    with 'processStep')
#  - 'closeSystem' function added to have the latest timestep in
#    the reaction dict and to be able to remove all molecules when
#    analyzing the results.
#  - When supplying multiple bond order files, the trajectories
#    are processed sequentially, instead of parallel. Latter would
#    assume that all molecules are available for reaction all the
#    time (but they are in different simulations, thus cannot react)
#  - Default number of processing steps changed from 1000 to 0
#    (i.e. process full trajectory)
#  - Round half-step bond orders for valency check to avoid artifical
#    oscillatory bond cleavage and formation (issue shifted to 0.25
#    before and after integer bond orders, presumably less important)
#    (BUGFIX: 2016-01-29)
#  - In addition to half-step rounded bond orders, the coordination
#    number of elements is added to avoid overcoordinated atoms, e.g.,
#    during reactions.
#  - Check connectivity after InChI-based bond order perception to
#    avoid wrong SMILES (separate bond order perception function
#    added in case of failure of default approach).
#

import sys
import array
import openbabel
import log as Log

## @class	Node
## @brief	object to store basic atom information and bond /
#		molecule information. Contains an openbabel.OBAtom
#		object to create openbabel.OBMol
## @version	2014/04/29:
#		In order to overcome some shortages of the openbabel
#		control methods when creating / comparing / processing
#		system of molecules this class enables ID based
#		processing of atoms instead of molecule bound index
#		based.
#		-> if possible, find elegant and efficient way to remove
#		this class, since it was created as a workaround
#
## @version	2017/04/07:
#		coordination number and charge are added to the object.
#
class Node:
	def __init__(self, Id, Type, Val, Cor, Q, Bonds, Mol):
		self.id = Id
		self.type = Type
		self.val = Val
		self.cor = Cor
		self.q = Q
		self.bonds = Bonds
		self.mol = Mol

		self.atom = openbabel.OBAtom()
		self.atom.SetId(Id)
		self.atom.SetAtomicNum(Type)

## @class	Processing
## @brief	compiles all methods used to process the simulation
#		results of ReaxFF
## @version	2014/05/05:
#		This class provides a library of processing functions
#		which can be applied on ReaxFF simulation results. The
#		internal dependencies of the functions is very small
#		(only one function is called internal). Thus,
#		initializing the 'Processing' object yields an
#		'operator' rather than an 'object'.
#
## @version	2014/05/07:
#		Change to 'real' processing unit - process data direct
#		after initialization. Extended __init__ which takes
#		'Filename', 'Work', 'Start' and 'N'. If a Processing
#		object is initialized with its default arguments it can
#		be used as an 'operator' again.
#
class Processing:
	## @brief	Constructor
	## @param	Processing
	## @param	Reader
	## @param	Identify
	## @param	Start
	## @param	Storage
	## @param	Static
	## @return	'Processing' object
	## @version	2015/11/12:
	#		Global molecule and bond structures are now initialized
	#		by initSystem. Extraction of time resolution has its own
	#		routine now.
	#
	## @version	2014/05/31:
	#		initialization of parameters and variables
	#
	def __init__(self, Processing=None, Reader=None, Identify={}, Start=0, Storage=0, Static=0):
		openbabel.obErrorLog.SetOutputLevel(0)
		self.conv = openbabel.OBConversion()
		if Processing is None:
			self.log = Log.Log(Width=70)
			self.Val = {1: 1, 2: 0, 6: 4, 7: 5, 8: 2, 9: 1, 10: 0, 16: 6, 17: 1, 18: 0}
			self.Cor = {1: 1, 2: 0, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0, 16: 6, 17: 1, 18: 0}
			self.num = Identify
			self.val = {}; self.cor = {}
			try:
				for element in Identify:
					self.val[element] = self.Val[Identify[element]]
					self.cor[element] = self.Cor[Identify[element]]
			except: self.log.printIssue(Text='some elements are not supported by the internal valence list. Please extend the list of supported elements by adding to Processing.Val', Fatal=True)
			self.start = Start
			self.static = Static
			self.storage = Storage


			self.timestep = 0
			self.dt_min = self.extractTimeResolution(Reader)

			self.bond = [[], []]
			self.update = [[], []]
			self.reaction = {}
			self.active = {}
			self.switch = False
		else:
			self.log = Processing.log
			self.Val = Processing.Val
			self.num = Processing.num
			self.val = Processing.val
			self.start = Start
			self.static = Processing.static
			self.storage = Processing.storage

			openbabel.obErrorLog.SetOutputLevel(0)
			self.conv = openbabel.OBConversion()

			self.timestep = 0
			self.dt_min = Processing.dt_min
			self.atom = Processing.atom
			self.molecule = Processing.molecule

			self.bond = Processing.bond
			self.update = Processing.update
			self.reaction = Processing.reaction
			self.active = Processing.active
			self.switch = Processing.switch

		self.steps = self.bondGenerator(Reader=Reader, Read=self.storage)
		self.step = next(self.steps); self.step = next(self.steps)
		fail = False
		while int(self.step[0]) < Start and not fail:
			try: self.step = next(self.steps)
			except: fail = True
		if fail: self.log.printIssue(Text='start time exceeds largest timestep. Nothing will be read from this bond order file.', Fatal=False)

		# . use generated first step to init atoms, bonds, molecules and virtual reactions
		if Processing is None:
			self.reaction, self.atom, self.molecule, self.bond[not self.switch] = self.initSystem()

	## @brief	executes single processing step
	## @param	Recrossing	switches recrossing filter off
	#				(True) or on (False)
	## @param	Steps		maximum number of steps a
	#				reaction is considered for
	#				recrossing (only if
	#				Recrossing=False)
	## @param	Skip		number of connectivity steps to
	#				be skiped
	## @param	Close		close system, i.e. add destructor
	#				reactions
	## @return	current timestep and reactions
	## @version	2014/05/07:
	#		This function processes a single timestep based
	#		on the functions given in the 'Processing' class
	#		in order to update the self.reaction dict.
	#
	## @version	2014/10/16:
	#		returns string formated version of reactions.
	#		Recrossing filter added: comparison between
	#		subsequent reactions. New version prints
	#		reactions of previous step rather than current
	#		to correctly account for recrossing.
	#
	#		data management structure (0: not self.switch,
	#		1: self.switch)
	#		timestep	n-2	n-1	n
	#		self.bond		0	1
	#		self.update	    0	    1
	#		self.molecule	0	1
	#		self.atom	0	1
	#		self.reaction		0	1
	#
	## @version	2014/10/20:
	#		new ouput style of 'chkReactions' incoporated:
	#		molecules as lists of atom IDs rather canonical
	#		SMILES. Line representations are generated in
	#		this function now.
	#
	## @version	2014/11/11:
	#		store reactions in self.reactions until a
	#		backreaction is detected or the reaction is
	#		older than Steps times the sampling frequency.
	#		The algorithm for filtering cyclic recrossing
	#		assumes that intermediates do not react with
	#		other molecules. Otherwise phantom reactions may
	#		be produced which lead to faulty
	#		concentrations in subsequent analysis. This is
	#		avoided by chosen a sufficient large parameter
	#		Steps, which defines the lower freq. border for
	#		recrossing.
	#		Skip parameter introduced for skipping several
	#		steps rather than checking each connectivity
	#		step.
	#
	## @version	2015/03/23:
	#		recorssing filter was concentrated in an extra
	#		function (cf. 'filterRecrossing' for more
	#		details)
	#
	## @version	2015/11/18:
	#		storage for atoms 'active'ly participating at
	#		reactions added
	#
	## @version	2016/12/20:
	#		close system on demand (default=True)
	#
	def processStep(self, Recrossing=True, Steps=1, Skip=1, Close=True):
		self.timestep, self.bond[self.switch] = self.extractBonds(Step=self.step, Bond=self.bond[self.switch], Static=self.static)
		self.bond[self.switch], tsbond = self.splitBonds(Bond=self.bond[self.switch], Atom=self.atom)
		self.update = self.reduceBonds(Old=self.bond[not self.switch], New=self.bond[self.switch])
		if self.update[0] or self.update[1]:
			atom = dict(self.atom)
			molecule = self.molecule
			self.molecule, self.atom = self.updateSystem(molecule, atom, self.update)
			for mol in self.molecule:
				if mol[1] == 'changed': mol[1] = self.canonical(mol[0], self.atom)
			self.reaction[self.timestep] = self.chkReactions([molecule, atom], [self.molecule, self.atom], self.update)
			# . active atoms
			self.active[self.timestep] = []
			for b in self.update[0]:
				self.active[self.timestep].append(b[1][0])
				self.active[self.timestep].append(b[1][1])
			for b in self.update[1]:
				self.active[self.timestep].append(b[1][0])
				self.active[self.timestep].append(b[1][1])
			# . recrossing
			if not Recrossing: self.reaction = self.filterRecrossing(Reaction=self.reaction, Timestep=self.timestep, Filter=Steps*self.dt_min, Debug=False)
			self.switch = not self.switch
		try:
			for i in range(Skip): self.step = next(self.steps)
			tmpReaction = {}
			if not Recrossing:
				for t in self.reaction:
					if self.timestep-t >= Steps*self.dt_min:
						tmpReaction[t] = self.reaction[t]
				for t in tmpReaction:
					self.reaction.pop(t,None)
			else: tmpReaction[self.timestep] = self.reaction.pop(self.timestep, [])
			return self.timestep, tmpReaction
		except:
			if Close: self.closeSystem()
			return -self.timestep, self.reaction


	## @brief	Extracts the time resolution of the simulation.
	#		Calculated as the difference of first and second time
	#		step in the Bond file.
	## @param	Reader		filestream containing the Bond Order file
	#				which was written during the simulation
	## @return	dt		time resolution of Bond file
	## @version	2015/11/12:
	#		This routine was excluded from system initialization
	#		since it needs an untouched Bond file, whereas
	#		initSystem operates on an already processed Bond file
	#		provided by the bondGenerator function.
	#
	def extractTimeResolution(self, Reader):
		line = Reader.readline()
		if 'Timestep' not in line:
			self.log.printIssue(Text='missing keyword "Timestep" at beginning of file '+Reader.name+'. The time resolution could not be retrieved.', Fatal=True)
		words = line.split()
		time0 = int(words[2])

		for i in range(7):
			line = Reader.readline()
		while line[0] != '#':
			line = Reader.readline()
		line = Reader.readline()
		if 'Timestep' not in line:
			self.log.printIssue(Text='missing keyword "Timestep" at second time step of file '+Reader.name+'. The time resolution could not be retrieved.', Fatal=True)
		words = line.split()
		time1 = int(words[2])
		dt = time1 - time0

		# . rewind file to beginning
		Reader.seek(0,0)

		return dt

	## @brief	initialization of the system which extracts all
	#		necessary information about the atoms present in
	#		the system using self.step
	## @return 	atoms		hashable list of Node objects,
	#				representing the atoms of the system
	## @return 	molecules	list of molecules at time step zero and
	#				their SMILES representation
	## @return 	bonds		list of bonds at time step zero, sorted
	#				by bond order, non-repetitive
	## @version	2014/03/19:
	#		The current version of the initializer extracts
	#		the atom types from the Bond Order file
	#		contained in 'Reader'. Based on the atom types
	#		their maximum valence is determined.
	#
	## @version	2014/05/03:
	#		New 'Node' object incorporated. Output
	#		consolidated to single dict
	#
	## @version	2015/11/12:
	#		Global molecule and bond structures are now initialized
	#		as well. They were previously initialized via
	#		chkReactions, which was time consuming in large systems.
	#		Extraction of time resolution has its own routine now.
	#
	## @version	2015/12/08:
	#		Transitional bonds are discarded in the creation of
	#		molecules.
	#
	## @version 2016/03/03:
	#		reenabled virtual initial reactions
	#
	## @version	2017/04/07:
	#		partial charges are extracted for processing of
	#		ionic species.
	#
	def initSystem(self):
		reaction = {}
		atoms = {}
		molecules = []
		bonds = []
		# . set bonds, atoms.ID .Type .Val
		for line in self.step[7:-2]:
			words = line.split()
			ID, TYPE, NB = int(words[0]), int(words[1]), int(words[2])
			BP = [int(words[i]) for i in range(3,3+NB)]
			BO = [float(words[i]) for i in range(4+NB,4+2*NB)]
			for i in range(NB):
				if BO[i] > self.static and ID < BP[i]:
					bonds.append([BO[i], [ID, BP[i]]])
			Q = float(words[6+2*NB])
			molecules.append([[],''])
			try: atoms[ID] = Node(
				Id = ID,
				Type = self.num[TYPE],
				Val = self.val[TYPE],
				Cor = self.cor[TYPE],
				Q = Q,
				Bonds = {},
				Mol = -1)
			except: self.log.printIssue(Text='missing identify for '+words[1]+'. The code does not know how to treat this/these atoms. Please restart the program and supply all required identifiers as follows:\n\n-i <element ID in ReaxFF>:<atomic number of element>', Fatal=True)
		bonds.sort(reverse=True)

		# . strip bonds from transitionals and set atom.Bonds
		bonds, tsbonds = self.splitBonds(bonds, atoms)
		for bond in bonds:
			if bond[1]:
				atoms[bond[1][0]].bonds[bond[1][1]] = bond[0]
				atoms[bond[1][1]].bonds[bond[1][0]] = bond[0]

		# . define recursive method that visits all atoms via bond partners
		def setMolIDRecursive(ID, Mol):
			if atoms[ID].mol < 0:
				atoms[ID].mol = Mol
				molecules[Mol][0].append(ID)
				for partner in atoms[ID].bonds:
					setMolIDRecursive(partner, Mol)
			return

		# . set molecule IDs in atoms.Mol and fill molecule list
		for id in atoms:
			setMolIDRecursive(id,id-1)
		
		# . sort atom ID list in molecules
		for i in range(len(molecules)):
			molecules[i][0].sort()

		# . set smiles
		for mol in molecules:
			if mol[0]:
				mol[1] = self.canonical(mol[0],atoms)

		# . set initial (virtual) reactions
		reaction[int(self.step[0])] = [ [[],[mol]] for mol in molecules if mol[0] ]

		return reaction, atoms, molecules, bonds

	## @brief	generate destructor reactions with final timestep
	## @return	None
	## @version	2016/12/20:
	#		Write destructor reactions for final timestep to delete chemical
	#		composition and report latest time.
	#
	def closeSystem(self):
		self.reaction[self.timestep] = [ [[mol],[]] for mol in self.molecule if mol[0] ]
		return None

	## @brief	create step generator object for ReaxFF bond
	#		order files
	## @param	Reader		filestream object to ReaxFF bond
	#				order file
	## @param	Read		buffer size for reading to char
	#				array
	## @return	generator which yields single steps from a
	#		ReaxFF Bond Order file
	## @version	2014/03/18:
	#		The c-build-in 'array' is used to combine the
	#		high read-speed of c with the easy and fast
	#		string processing of python. This generator
	#		loops over timesteps of a Bond Order file, as
	#		long as it is called. It reads until the
	#		indicator 'Timestep' is found twice at least.
	#		Thus, 'data' will contain one full step at
	#		least. If the end of file is not reached, the
	#		last step in 'data' will be preserved since it
	#		is most likely not read complete.
	#
	## @version	2015/04/19:
	#		name changed to 'bondGenerator' to avoid
	#		confusion with the newly introduced
	#		'dumpGenerator'.
	#
	def bondGenerator(self, Reader, Read=10000):
		data = ''
		done = False
		while not done:
			while  data.count('Timestep') < 2 and not done:
				try: data += reader.read(Read)
				except: done = True
			steps = data.split('# Timestep ')
			
			if not done:
				data  = steps[-1]
				del steps[-1]
			for step in steps: yield step.split('\n')

	## @brief	create step generator object for ReaxFF dump
	#		files
	## @param	Reader		filestream object to ReaxFF dump
	#				file
	## @param	Read		buffer size for reading to char
	#				array
	## @return	generator which yields single steps from a
	#		ReaxFF dump file
	## @version	2015/04/19:
	#		Similar to the 'bondGenerator' function this
	#		method yields information from a ReaxFF output
	#		file, step by step.
	#
	def dumpGenerator(self, Reader, Read=10000):
		data = ''
		done = False
		while not done:
			while data.count('ITEM: TIMESTEP') < 2 and not done:
				try: data = Reader.read(Read)
				except: done = True
			steps = data.split('ITEM: TIMESTEP\n')
			if not done:
				data = steps[-1]
				del steps[-1]
			for step in steps: yield step.split('\n')

	## @brief	extracts bond information given by a timestep
	#		read from a Bond Order file
	## @param	Step		array of strings read from Bond
	#				Order file by 'bondGenerator'
	## @param	Bond		working array which is should be
	#				reused to avoid calling the
	#				'append' function too often
	## @param	Static		static cutoff of the bond order,
	#				which help to avoid faulty
	#				interpretation of very long
	#				range interactions
	## @return	time of the timestep and list of bonds and
	#		corresponding bond order
	## @version	2015/11/12:
	#		replaced bond uniqueness check with faster one.
	#
	## @version	2014/03/18:
	#		Steps read by the 'bondGenerator' are processed
	#		by this method. It divides the bond information,
	#		which is given per atom, into single bonds and
	#		their bond order. This yields a list of bonds
	#		for each timestep and allows to compare these
	#		lists afterwards.
	#
	## @version	2017/04/07:
	#		partial charges are extracted for processing of
	#		ionic species.
	#
	def extractBonds(self, Step, Bond=[], Static=0.5):
		time = int(Step[0])
		lines = Step[7:-2]

		idx = 0
		LB = len(Bond)
		for i in range(LB): Bond[i] = [0, []]

		for line in lines:
			words = line.split()
			ID, TYPE, NB = int(words[0]), int(words[1]), int(words[2])
			BP = [int(words[i]) for i in range(3,3+NB)]
			BO = [float(words[i]) for i in range(4+NB,4+2*NB)]
			step = [[BO[i], [ID, BP[i]]] for i in range(NB) if BO[i] > Static]
			for bond in step:
				if ID < bond[1][1]:
					if idx < LB: Bond[idx] = bond
					else: Bond.append(bond)
					idx += 1
			Q = float(words[6+2*NB])
			self.atom[ID].q = Q
		Bond.sort(reverse=True)
		return time, Bond

	## @brief	splits all bonds of a timestep into "real" bonds
	#		and long-range interactions
	## @param	Bond		array of bonds and their bond
	#				order from 'extractBond'
	## @param	Atom		hashable list of Atom objects
	## @return	list of "real" bonds and list of long-range
	#		interactions, called 'tsBond' since it could be
	#		used for determining transition states
	## @version	2014/03/18:
	#		In order to avoid interpreting long-range
	#		interaction as actual molecules, this method
	#		splits the 'Bond' array into "real" bonds and
	#		long-range interactions. The latter can be used
	#		to determine the transition state, or at least
	#		the inital guess for a transition state
	#		optimization.
	#
	## @version	2016/03/18:
	#		check for empty <Bond> array added
	#
	## @version	2017/03/29:
	#		replaced integer bond order valency check with
	#		half-step valency check:
	#		 Static < BO < 0.75: BO = 0.5
	#		 0.75 < BO < 1.25:   BO = 1
	#		 1.25 < BO < 1.75:   BO = 1.5
	#		etc.
	#
	## @version	2017/04/07:
	#		coordination number check
	#
	def splitBonds(self, Bond, Atom):
		tsBond = []
		if Bond:
			buff = list(zip(*Bond))
			valence = {}; coordination = {}
			for i in Atom: valence[i] = 0; coordination[i] = 0
			for i in range(len(buff[1])):
				b = buff[1][i]
				if not b: continue
				if Atom[b[0]].val > valence[b[0]] and Atom[b[1]].val > valence[b[1]] and Atom[b[0]].cor > coordination[b[0]] and Atom[b[1]].cor > coordination[b[1]]:
					# . Version 2017/03/29: replaced with below line
					#if buff[0][i] < 0.5: BO = 1
					#else: BO = int(round(buff[0][i]))
					BO = round(2.0*buff[0][i])/2.0
					valence[b[0]] += BO; coordination[b[0]] += 1
					valence[b[1]] += BO; coordination[b[1]] += 1
				else: Bond[i] = [0, []]
			tsBond = [[buff[0][i], buff[1][i]] for i in range(len(buff[0])) if Bond[i][0] == 0]
			Bond.sort(reverse=True)
		return Bond, tsBond

	## @brief	reduces list of bonds to important, reactive
	#		bonds
	## @param	Old		previous bond situation to which
	#				the new one is compared
	## @param	New		current bond situation
	## @return	bond depletion and creation as two arrays,
	#		each containing the bonds and their bond order
	## @version	2014/03/18:
	#		In order to check whether a bond is present in
	#		both timesteps, the new and the old, it is
	#		necessary to cheak whether the bond, or its
	#		inverse, is in the list of bonds of the previous
	#		step. If not, the bond either vanished (if the
	#		bond is present in the old but not the new step)
	#		or the bond is created (if the bond is present
	#		in the new but not the old step).
	#
	def reduceBonds(self, Old, New):
		old = []; new = []
		if Old: old = list(zip(*Old))[1]
		if New: new = list(zip(*New))[1]
		depletion = [i for i in Old if i[1] and i[1] not in new and i[1][::-1] not in new]
		creation = [i for i in New if i[1] and i[1] not in old and i[1][::-1] not in old]
		return depletion, creation

	## @brief	increase spin multiplicity of an OBAtom
	## @param	Atom	OBAtom
	## @return	OBAtom with increased spin multiplicity
	## @version	2017/05/19:
	#		Increase spin multiplicity of an OBAtom by one.
	#		For openbabel, singlets have a spin multiplicity
	#		of zero.
	#
	def increaseMultiplicity(self, Atom):
		Atom.SetSpinMultiplicity(Atom.GetSpinMultiplicity() + 1)
		if Atom.GetSpinMultiplicity() == 1:
			Atom.SetSpinMultiplicity(2)
		return Atom

	## @brief	decrease spin multiplicity of an OBAtom
	## @param	Atom	OBAtom
	## @return	OBAtom with decreased spin multiplicity
	## @version	2017/05/19:
	#		Decrease spin multiplicity of an OBAtom by one.
	#		For openbabel, singlets have a spin multiplicity
	#		of zero.
	#
	def decreaseMultiplicity(self, Atom):
		Atom.SetSpinMultiplicity(Atom.GetSpinMultiplicity() - 1)
		if Atom.GetSpinMultiplicity() == 1:
			Atom.SetSpinMultiplicity(0)
		return Atom

	## @brief	perceive bond orders (in case default procedure failed)
	## @param	Molecule	list of atom IDs the Molecule
	#				consists of
	## @param	Atom		list of all atoms, each given as
	#				an 'Atom' object
	## @return	canonical SMILES
	## @version	2017/05/19:
	#		If the InChI-based bond order perception failed, the
	#		ReaxFF Bond Order information will be used to rank
	#		bonds between atoms with free valencies in order
	#		of decreasing bond order. Starting with the strongest
	#		bond, bond orders are increased and the respecive 
	#		ReaxFF bond orders are reduced (stepwise: +1 BO).
	#		If a bond order becomes negative, or no free valencies
	#		are left at either bond partner, that bond remains
	#		with its bond order. This is done until all free
	#		valencies are used up or no ReaxFF bond order is left
	#		above zero (given the -1 corrections for added bonds).
	#
	def perceiveBondOrders(self, Molecule, Atom):
		# . generate OBMol
		frag = openbabel.OBMol()
		atoms = [None] + Molecule
		for i in Molecule:
			frag.AddAtom(Atom[i].atom)
			[frag.AddBond(atoms.index(Atom[i].id), atoms.index(bond), 1) for bond in Atom[i].bonds if Atom[i].id in atoms and bond in atoms]
		frag.AssignSpinMultiplicity(True)
		# . find potential multi-bonds
		multi = [[], []]
		for atom in openbabel.OBMolAtomIter(frag):
			if atom.GetSpinMultiplicity() > 0:
				for bond in openbabel.OBAtomBondIter(atom):
					ptr = bond.GetNbrAtom(atom)
					if ptr.GetSpinMultiplicity() > 0:
						aidx = atom.GetIdx();pidx = ptr.GetIdx()
						bo= Atom[atoms[aidx]].bonds[atoms[pidx]]
						if (pidx, aidx) in multi[1]: continue
						multi[0].append(bo-1.0)
						multi[1].append((aidx, pidx))
		# . sort in order of decreasing bond orders
		multi = sorted(zip(multi[0], multi[1]), reverse=True)
		# . loop over bonds in <multi> and set multi-bonds
		while multi:
			bond = list(multi[0]); del multi[0]
			atom = frag.GetAtom(bond[1][0])
			btom = frag.GetAtom(bond[1][1])
			if atom.GetSpinMultiplicity() > 0 and btom.GetSpinMultiplicity() > 0:
				tmp = frag.GetBond(atom, btom)
				tmp.SetBO(tmp.GetBO() +1)
				atom = self.decreaseMultiplicity(Atom=atom)
				atom.DecrementImplicitValence()
				btom = self.decreaseMultiplicity(Atom=btom)
				btom.DecrementImplicitValence()
				bond[0] -= 1.0
				if bond[0] > 0: multi.append(tuple(bond))
			multi = sorted(multi, reverse=True)
		# . return SMILES
		self.conv.SetInAndOutFormats('mdl', 'can')
		return self.conv.WriteString(frag).strip().replace('@@','@')

	## @brief	creates canonical SMILES from list of atom IDs
	#		and list of all atoms
	## @param	Molecule	list of atom IDs the Molecule
	#				consists of
	## @param	Atom		list of all atoms, each given as
	#				an 'Atom' object
	## @return	canonical SMILES
	## @version	2014/04/29:
	#		Create openbabel.OBMol object from list of atom
	#		IDs based on a list of Atom objects.
	#
	## @version	2014/10/20:
	#		'@@' vs. '@' treatment shifted from
	#		'chkReactions' to this method.
	#
	## @version	2016/03/18:
	#		workaround for H and H2
	#
	## @version	2017/05/19:
	#		for some species the correct connectivity gets lost
	#		when converting it to an InChI (which is required
	#		for bond order perception). Check added.
	#		TODO: deal with failure
	#
	def canonical(self, Molecule, Atom):
		# . generate OBMol
		frag = openbabel.OBMol()
		atoms = [None] + Molecule
		for i in Molecule:
			frag.AddAtom(Atom[i].atom)
			[frag.AddBond(atoms.index(Atom[i].id), atoms.index(bond), 1) for bond in Atom[i].bonds if Atom[i].id in atoms and bond in atoms]
		frag.AssignSpinMultiplicity(True)
		# . generate output
		if frag.NumHvyAtoms() > 0:
			# . generate single-bonded SMILES for checking
			self.conv.SetInAndOutFormats('mdl', 'can')
			pre = self.conv.WriteString(frag).strip().replace('@@','@')
			# . perceive bond orders via conversion to and from InChI
			self.conv.SetInAndOutFormats('mdl','inchi')
			inchi = self.conv.WriteString(frag)
			self.conv.SetInAndOutFormats('inchi','mdl')
			self.conv.ReadString(frag, inchi)
			self.conv.SetInAndOutFormats('mdl','can')
			out = self.conv.WriteString(frag).strip().replace('@@','@')
			frag.AddHydrogens()
			# . check if InChI conversion worked properly
			chk = openbabel.OBMol(frag)
			for bond in openbabel.OBMolBondIter(chk):
				bo = bond.GetBO()
				if bo > 1:
					bond.SetBO(1)
					atom = bond.GetBeginAtom()
					atom.SetImplicitValence(atom.GetImplicitValence() +(bo-1))
					btom = bond.GetEndAtom()
					btom.SetImplicitValence(btom.GetImplicitValence() +(bo-1))
			chk.AssignSpinMultiplicity(True)
			post = self.conv.WriteString(chk).strip().replace('@@','@')
			if pre != post: out = self.perceiveBondOrders(Molecule=Molecule, Atom=Atom)
		elif frag.NumAtoms() == 1: out = '[H]'
		elif frag.NumAtoms() == 2: out = '[H][H]'
		return out

	## @brief	update the bond situation of a system of
	#		'Molecules' / 'Atoms' based on the depletion
	#		and creation of bonds given in 'Bonds'
	## @param	Molecule	list containing the lists of
	#				atom IDs of each molecule
	## @param	Atom		list of Atom objects
	## @param	Bond		double list of bond depletion
	#				and creation
	## @return	New 'Molecules'
	## @version	2014/04/29:
	#		Based on a list of existing molecules, Atom
	#		objects and bond depletion and creation the
	#		Molecule list is updated and returned together
	#		with a list of reaction and the updated list of
	#		Atom objects.
	#
	def updateSystem(self, Molecule, Atom, Bond):
		molecule = [[mol[0][:],mol[1]] for mol in Molecule]
		atom = {}
		for i in Atom:
			atom[i] = Node(Atom[i].id, Atom[i].type, Atom[i].val, Atom[i].cor, Atom[i].q, dict(Atom[i].bonds), Atom[i].mol)
		mols = len(molecule)

		# . remove bonds and split molecules
		for bond in Bond[0]:
			del atom[bond[1][0]].bonds[bond[1][1]]
			del atom[bond[1][1]].bonds[bond[1][0]]
			frag = []; new = [atom[bond[1][1]]]
			for next in new:
				if next.id not in frag:
					frag.append(next.id)
					[new.append(atom[b]) for b in next.bonds]
			frag.sort()
			# . dissociation
			if frag != molecule[atom[bond[1][0]].mol][0]:
				[molecule[atom[bond[1][0]].mol][0].remove(i) for i in frag]
				if frag: molecule[atom[bond[1][0]].mol][1] = 'changed'
				for i in range(mols):
					if not molecule[i][0]:
						molecule[i][0] = frag
						molecule[i][1] = 'changed'
						for j in molecule[i][0]: atom[j].mol = i
						break
			# . isomerization
			else: molecule[atom[bond[1][0]].mol][1] = 'changed'

		# . add bonds and merge molecules
		for bond in Bond[1]:
			atom[bond[1][0]].bonds[bond[1][1]] = bond[0]
			atom[bond[1][1]].bonds[bond[1][0]] = bond[0]
			# . association
			if bond[1][1] not in molecule[atom[bond[1][0]].mol][0]:
				molecule[atom[bond[1][0]].mol][0] += molecule[atom[bond[1][1]].mol][0]
				molecule[atom[bond[1][0]].mol][0].sort()
				molecule[atom[bond[1][0]].mol][1] = 'changed'
				molecule[atom[bond[1][1]].mol][0] = []
				molecule[atom[bond[1][1]].mol][1] = ''
				for j in molecule[atom[bond[1][0]].mol][0]: atom[j].mol = atom[bond[1][0]].mol
			# . isomerization
			else: molecule[atom[bond[1][0]].mol][1] = 'changed'
		return molecule, atom

	## @brief	checks for reactions between the old set of
	#		molecules and the updated/new set of molecules
	## @param	Old		list of old [molecules, atoms]:
	#				previous timestep
	## @param	New		list of new [molecules, atoms]:
	#				current timestep
	## @param	Bond		double list of first, bond
	#				depletion and second, bond
	#				creation
	## @return	list of reactions (all sorted !)
	## @version	2014/05/03:
	#		From the change between the molecules and atoms
	#		of the previous and the current timestep the
	#		reactions are extracted. The reactions are given
	#		as a double list of first, the reactants and
	#		second the products, both in its respective
	#		canonical SMILES representation. Enantiomeres
	#		represented by '@@' and '@' are reduced to the
	#		'@' case due to the lack of coordinate
	#		information.
	#
	## @version	2014/10/20:
	#		Enantiomere treatment moved to 'canonical' and
	#		output changed to list of atom IDs rather than
	#		canonical smiles. This allows for comparing
	#		molcules of subsequent reactions (important for
	#		filtering recrossing).
	#
	def chkReactions(self, Old, New, Bond):
		reaction = []
		Molecule = Old[0]; Atom = Old[1]
		molecule = New[0]; atom = New[1]

		# . connection between previous and current molecules
		#   via broken bonds
		for bond in Bond[0]:
			reac = [[Molecule[Atom[bond[1][0]].mol]], []]
			if atom[bond[1][0]].mol == atom[bond[1][1]].mol:
				reac[1] = [molecule[atom[bond[1][0]].mol]]
			else: reac[1] = [molecule[atom[bond[1][0]].mol], molecule[atom[bond[1][1]].mol]]
			subnet = False
			for r in reaction:
				subnet = any(A in r[0] for A in reac[0]) or any(B in r[1] for B in reac[1])
				if subnet:
					[r[0].append(A) for A in reac[0] if A not in r[0]]
					[r[1].append(B) for B in reac[1] if B not in r[1]]
					break
			if not subnet: reaction.append(reac)

		# . connection between current and previous molecules
		#   via formed bonds
		for bond in Bond[1]:
			reac = [[], [molecule[atom[bond[1][0]].mol]]]
			if bond[1][1] not in Molecule[Atom[bond[1][0]].mol][0]:
				reac[0] = [Molecule[Atom[bond[1][0]].mol], Molecule[Atom[bond[1][1]].mol]]
			else: reac[0] = [Molecule[Atom[bond[1][0]].mol]]
			subnet = False
			for r in reaction:
				subnet = any(A in r[0] for A in reac[0]) or any(B in r[1] for B in reac[1])
				if subnet:
					[r[0].append(A) for A in reac[0] if A not in r[0]]
					[r[1].append(B) for B in reac[1] if B not in r[1]]
					break
			if not subnet: reaction.append(reac)

		# . algorithm of merging reactions requires final check
		for r in reaction:
			for reac in reaction:
				subnet = any(A in r[0] for A in reac[0] if r != reac) or any(B in r[1] for B in reac[1] if r != reac)
				if subnet:
					[r[0].append(A) for A in reac[0] if A not in r[0]]
					[r[1].append(B) for B in reac[1] if B not in r[1]]
					del reaction[reaction.index(reac)]
		return [[sorted(r[0]), sorted(r[1])] for r in reaction]

	## @brief	filter fast back-and-forth reactions
	## @param	Reaction	time-resolved list of reactions
	## @param	Timestep	filter recrossings for reactions
	#				of that timestep
	## @param	Filter		time-period for filtering
	## @param	Debug		activates debugging output
	## @version	2015/10/08:
	#		Timesteps considered for recrossing are now calculated
	#		independently of entries of 'Reaction' dict. 'Filter'
	#		setting can now have an effect. However, making the
	#		'Reaction' dict hold only entries within filter range
	#		is still done by processStep.
	#
	## @version	2015/04/19:
	#		Recrossing filter was formerly included in the
	#		'processStep' function. The new version is based
	#		on a different idea:
	#		Whenever a product is found on the reactant
	#		side in a past reaction (exact same atoms and
	#		connectivity), it is removed. Whenever a product
	#		is found in a past reaction which is involved in
	#		a reaction relevant for recrossing, it is added
	#		to the 'final' products of the recrossing
	#		reactions.
	#
	def filterRecrossing(self, Reaction, Timestep, Filter, Debug=False):
		times = sorted(Reaction)
		if Timestep not in times: return False	# exit point
		# . copy Reaction in a way that changes in 'reaction'
		#   do not affect the corresponding values in 'Reaction'
		#   (deepcopy)
		reaction = {}
		for t in Reaction: reaction[t] = list(Reaction[t])

		idx = times.index(Timestep)
		irec = 0
		for i in range(len(times)):
			#if times[i] <= Timestep-Filter:
			if times[i] >= Timestep-Filter:
				irec = i
				break
		revTime = times[irec:idx][::-1]

		# . loop over all current reactions
		for r in Reaction[Timestep]:
			tmp = {Timestep: [r]}
			prod = []
			for mol in r[1]: prod.append(mol)
			if Debug:
				print()
				print('>>> DEBUG: check recorssing filter')
				print('>>> timestep:', Timestep)
				print('>>> reaction:', r)
				print('>>> initial', prod)
			# . loop backward over time
			i = 0
			while prod and i < len(revTime):
				t = revTime[i]
				for reac in Reaction[t]:
					source = []
					for mol in reac[0]: source += mol[0]
					target = []
					for mol in prod: target += mol[0]
					if any(a in source for a in target):
						# . whenever an atom of the final products participate on
						#   a reaction, it has to be filtered in case of recorssing
						if t in tmp: tmp[t].append(reac)
						else: tmp[t] = [reac]
						# . add products which contain atoms which are not in the
						#   initial list of product's atoms. That means that some
						#   sort of coupling with other molecules is taking place
						for mol in reac[1]:
							if any(a not in target for a in mol[0]):
								prod.append(mol)
								if Debug:
									print('>>>', reac)
						# . remove products which have been reactants of previous
						#   reactions
						for mol in reac[0]:
							if mol in prod:
								prod.remove(mol)
				if Debug:
					print('>>>', prod)
				i += 1
			if not prod:
				for t in tmp:
					for r in tmp[t]:
						try: reaction[t].remove(r)
						except: self.log.printIssue(Text='Removing recorssing reaction failed, since already removed (presumably due to previous filtering).', Fatal=False)

		# . remove self reaction without change in chemical
		#   composition
		for r in Reaction[Timestep]:
			if r in reaction[Timestep] and all(mol in r[1] for mol in r[0]):
				reaction[Timestep].remove(r)

		return reaction


#######################################
if __name__ == '__main__':
	log = Log.Log(Width=70)
	save = 1000

	###	HEADER
	#
	text = ['<source file> [<work file>] [-start <start time>] [-mem <memory in bytes>] [-cut <satic bond order cutoff>] [-n <number of steps>] [-i <element ID in ReaxFF>:<atomic number of element>] [-all] [-norec <recsteps>] [-skip <steps>]',
		'options (for latest <source file>):',
		'-start: timestep to start reading at, except the latest timestep in a <work file> exceeds the <start time> (default=0)',
		'-mem: bytes of memory available for reading from the bond order file (default=10000)',
		'-cut: static bond order cutoff which is used to eliminate long range interactions (default=0.5)',
		'-n: set number of steps for processing; to run until end of file is reached use negative value or zero (default=0)',
		'-i: define element identifier (without identifiers the code can not process data). Note: each element has to be given in a separte -i option (e.g. -i 1:6 -i 2:1, for ID 1 being carbon and ID 2 being hydrogen)',
		'-all: remove <source file> binding of element identifiers. Each identifier given in the command line is copied to all bond order files',
		'-norec: filter-period for fast back- and forth reactions. Note: For very reactive systems, recrossing events and "normal" fast back- and forth reactions can occur on the same timescale!',
		'-skip: analyzse only each <steps>th entry of the connectivity file']
	log.printHead(Title='ChemTraYzer - ReaxFF Data Processing', Version='2017-04-07', Author='Malte Doentgen, LTT RWTH Aachen University', Email='chemtrayzer@ltt.rwth-aachen.de', Text='\n\n'.join(text))

	###	INPUT
	#
	if len(sys.argv) > 1: argv = sys.argv
	else:
		log.printComment(Text=' Use keyboard to type in filenames, identifiers and optional parameters. The <return> button will not cause leaving the input section. Use <strg>+<c> or write "done" to proceed.', onlyBody=False)
		argv = ['input']; done = False
		while not done:
			try: tmp = input('')
			except: done = True
			if 'done' in tmp.lower(): done = True
			else:
				if '!' in tmp: argv += tmp[:tmp.index('!')].split()
				else: argv += tmp.split()

	log.printBody(Text='Reading files ...', Indent=0)

	###	INTERPRET INPUT
	#
	filename = []
	work = {}
	identify = {}
	id_all = False
	start = {}
	storage = {}
	static = {}
	n = {}
	recrossing = True
	recsteps = 0
	skip = 1
	i = 1
	while i < len(argv):
		if argv[i].lower() == '-start' and filename:
			try: start[filename[-1]] = int(argv[i+1]); i += 1
			except: log.printIssue(Text='-start: option expected integer, got string/nothing and will use default=0 for '+filename[-1], Fatal=False)
		elif argv[i].lower() == '-mem' and filename:
			try: storage[filename[-1]] = int(argv[i+1]); i += 1
			except: log.printIssue(Text='-mem: option expected integer, got string/nothing and will use default=10000 for '+filename[-1], Fatal=False)
		elif argv[i].lower() == '-cut' and filename:
			try: static[filename[-1]] = float(argv[i+1]); i += 1
			except: log.printIssue(Text='-cut: cutoff option expected float, got string/nothing and will use default=0.5 for '+filename[-1], Fatal=False)
		elif argv[i].lower() == '-n' and filename:
			try: n[filename[-1]] = int(argv[i+1]); i += 1
			except: log.printIssue(Text='-n: steps option expected integer, got string/nothing and will use default=0 for '+filename[-1], Fatal=False)
		elif argv[i].lower() == '-i' and filename:
			try: tmp = argv[i+1].split(':'); int(tmp[0]); int(tmp[1]); i += 1
			except: log.printIssue(Text='-i: identify option expected additional argument to define an elements identifier. Supply as follows:\n\n-i <element ID in ReaxFF>:<atomic number of element>', Fatal=True)
			if len(tmp) != 2: log.printIssue(Text='-i: identify option expected argument seperated by single ":". Supply as follows:\n\n-i <element ID in ReaxFF>:<atomic number of element>', Fatal=True)
			else:
				if filename[-1] in identify: identify[filename[-1]][int(tmp[0])] = int(tmp[1])
				else: identify[filename[-1]] = {int(tmp[0]): int(tmp[1])}
		elif argv[i].lower() == '-all': id_all = True
		elif argv[i].lower() == '-norec':
			recrossing = False
			try: recsteps = int(argv[i+1]); i += 1
			except: log.pintIssue(Text='-norec: option expected integer, got string/nothing and will use default=0 / no filter', Fatal=False)
		elif argv[i].lower() == '-skip':
			try: skip = int(argv[i+1]); i += 1
			except: log.printIssue(Text='-skip: option expected integer, got string/nothing and will use default=1', Fatal=False)
		else:
			try: line = open(argv[i], 'r').readline()
			except: log.printIssue(Text='attempt to read from '+argv[i]+' failed. Please check whether file is broken or does not exist. File will be ignored ...', Fatal=False); line = ''
			if 'Timestep' in line: filename.append(argv[i])
			elif 'WORK' in line: work[line.split()[1]] = argv[i]
		i += 1
	if len(filename) > len(set(filename)):
		filename = set(filename)
		log.printIssue(Text='redundant filenames found, condensing to unique set of files', Fatal=False)

	if id_all:
		for f in filename:
			if f not in identify: identify[f] = {}
		for f in filename:
			for g in identify:
				for i in identify[g]:
					if i not in identify[f]: identify[f][i] = identify[g][i]
	for f in filename:
		if f not in start: start[f] = 0
		if f not in storage: storage[f] = 10000
		if f not in static: static[f] = 0.5
		if f not in n: n[f] = 0
		if f not in identify: log.printIssue(Text='no identifier(s) for '+f+' found. Please restart the code with all required identifier. Supply as follows:\n\n-i <element ID in ReaxFF>:<atomic number of element>', Fatal=True)

	def readWork(Work):
		reader = open(Work, 'r')
		line = reader.readline(); done = False
		worktime = -1; reaction = {}
		while not done:
			words = reader.readline().split(':')
			if len(words) > 1:
				if int(words[0]) not in reaction:
					reaction[int(words[0])] = [[words[1].split('\n')[0].split(', '), words[2].split('\n')[0].split(', ')]]
				else:
					reaction[int(words[0])] += [words[1].split('\n')[0].split(', '), words[2].split('\n')[0].split(', ')]
			elif words[0]: worktime = int(words[0].split('\n')[0])
			else: done = True
		reader.close()
		if worktime == -1 or worktime < max(reaction): worktime = max(reaction)
		return worktime, reaction

	###	FILE INITIALIZATION
	#
	reaction = {}
	worktime = {}
	if filename:
		if any(f in work for f in filename):
			log.printComment(Text='Bond order files with work data', onlyBody=False)
			log.printBody(Text=', '.join(f for f in filename if f in work), Indent=1)
		if any(f not in work for f in filename):
			log.printComment(Text='Bond order files without work data', onlyBody=False)
			log.printBody(Text=', '.join(f for f in filename if f not in work), Indent=1)
		if any(f not in filename for f in work):
			log.printComment(Text='Additional work files (<work file>: <source file>)', onlyBody=False)
			log.printBody(Text=', '.join(work[f]+': '+f for f in work if f not in filename), Indent=1)
		log.printBody(Text='', Indent=1)
		for f in filename:
			if f in work:
				worktime[f], tmp = readWork(work[f])
				for time in tmp:
					if time in reaction: reaction[time] += tmp[time]
					else: reaction[time] = tmp[time]
				log.printBody(Text='timesteps '+repr(min(tmp))+' to '+repr(worktime[f])+' of "'+f+'" read from "'+work[f]+'"', Indent=1)
				if worktime[f] < start[f]: worktime[f] = start[f]
		for f in work:
			if f not in filename:
				wt, tmp = readWork(work[f])
				for time in tmp:
					if time in reaction: reaction[time] += tmp[time]
					else: reaction[time] = tmp[time]
				log.printBody(Text='timesteps '+repr(min(tmp))+' to '+repr(wt)+' of "'+f+'" read from "'+work[f]+'"', Indent=1)
		if work: log.printBody(Text='', Indent=0)
	elif work:
		log.printComment(Text='Work files (<work file>: <source file>)', onlyBody=False)
		log.printBody(Text=', '.join(work[f]+': '+f for f in work)+'\n', Indent=1)
		for f in work:
			wt, tmp = readWork(work[f])
			for time in tmp:
				if time in reaction: reaction[time] += tmp[time]
				else: reaction[time] = tmp[time]
			log.printBody(Text='timesteps '+repr(min(tmp))+' to '+repr(wt)+' of "'+f+'" read from "'+work[f]+'"', Indent=1)
		log.printBody(Text='', Indent=0)
	else: log.printIssue(Text='No input files found ...', Fatal=True)

	###	PROCESS FILES
	#
	if filename: log.printBody(Text='Processing files ...', Indent=0)
	for f in filename:
		if f not in work: worktime[f] = start[f]
		if n[f] <= 0: run = 'all'
		else: run = repr(n[f])
		log.printBody(Text=f+' reading '+run+' steps, staring at timestep '+repr(worktime[f])+', using '+repr(storage[f])+' bytes memory, a static cutoff of '+repr(static[f])+', considering '+repr(recsteps)+' steps for recrossing and reading each '+repr(skip)+'th step of connectivity.\nIdentifying '+', '.join(repr(key)+': '+repr(identify[f][key]) for key in identify[f]), Indent=1)
		reader = open(f, 'r')
		proc = Processing(Reader=reader, Identify=identify[f], Start=worktime[f], Storage=storage[f], Static=static[f])
		done = False; i = 0
		if f in work: timestep, tmp = proc.processStep(Recrossing=recrossing)
		else:
			work[f] = f+'.reac'
			open(work[f], 'w').write('WORK '+f+' <temperature> <volume> <timestep>\n')
		while not done:
			timestep, tmp = proc.processStep(Recrossing=recrossing, Steps=recsteps, Skip=skip, Close=True)
			if timestep >= 0:
				if timestep not in reaction: reaction[timestep] = []
				for t in tmp:
					if t not in reaction: reaction[t] = []
					reaction[t] += tmp[t]
					for r in tmp[t]:
						reac = ','.join(mol[1] for mol in r[0])+':'+','.join(mol[1] for mol in r[1])
						log.printBody(Text=repr(t)+':'+reac, Indent=2)
						open(work[f], 'a').write(repr(t)+':'+reac+'\n')
				if i%save == 0 and i != 0: reaction[timestep].append('')
				i += 1
				if n[f] > 0 and i > n[f]: done = True
			else:
				done = True
				timestep = abs(timestep)
				for t in tmp:
					if t not in reaction: reaction[t] = []
					reaction[t] += tmp
					for r in tmp[t]:
						reac = ','.join(mol[1] for mol in r[0])+':'+','.join(mol[1] for mol in r[1])
						log.printBody(Text=repr(t)+':'+reac, Indent=2)
						open(work[f], 'a').write(repr(t)+':'+reac+'\n')
		reader.close()

