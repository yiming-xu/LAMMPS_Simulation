##################################################################
# The MIT License (MIT)                                          #
#                                                                #
# Copyright (c) 2016 RWTH Aachen University, Felix Schmalz       #
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
## @file	dbhandler.py
## @author	Felix Schmalz
## @date	2016/09/01
## @brief	methods for controlling the SQLite database

import sqlite3
import time
import log

## @class	Database
## @brief	object that handles database access
## @version	2016/01/23
#
## @version	2016/09/01:
#		QM methods names required
#		hardcoded Ha0 values removed
#
class Database:
	## @brief	Constructor
	## @parameter	Name		name of the DB
	## @parameter	QMmethods	names of used QM
	## @parameter	Timeout		sqlite timeout in s
	## @return	None
	## @version	2016/01/23
	#
	## @version	2016/09/01:
	#		QM methods names required
	#
	def __init__(self, Name='./chemtrayzer.sqlite', QMmethods=['',''], Timeout=5):
		# . log
		self.log = log.Log(Width=70)

		# . name
		self.name = Name
		self.timeout = Timeout

		# . database structure
		self.sql_tables = [
				# . information
				('info','CREATE TABLE info (user text, date text, commandS text, commandTS text, comment text)'),
				# . species data
				('species','CREATE TABLE species (smile text, conformer integer, molmass real, energy real, tr1 real, tr2 real, tr3 real, sym integer, spin real, htherm real, ff_energy real, ff_i1 real, ff_i2 real, ff_i3 real, geometry text)'),
				('hofreq','CREATE TABLE hofreq (smile text, conformer integer, frequency real)'),
				('rottemp','CREATE TABLE rottemp (smile text, conformer integer, frequency real, temperature real)'),
				# . reaction data
				('reactionTS','CREATE TABLE reactionTS (smile text, conformer integer, energy real, molmass real, tr1 real, tr2 real, tr3 real, sym integer, spin real, htherm real, emin real, emax real, i1min real, i1max real, i2min real, i2max real, i3min real, i3max real, geometry text)'),
				('tsfreq','CREATE TABLE tsfreq (smile text, conformer integer, frequency real)'),
				('tstemp','CREATE TABLE tstemp (smile text, conformer integer, frequency real, temperature real)'),
				('reactionB','CREATE TABLE reactionB (smile text, molmass real, temperature real, rate real, klo real, kup real, n real, samplesize integer, geometry text)')
			]

		# . create handle
		self.database, self.cursor = self.initDB()

		# . write! initial contents
		self.writeMethods(QMmethods)

		# . print DB Info
		self.printInfo()

	## @brief	destructor, called when object is no longer needed
	## @version	2016/09/01:
	#		replaces explicit call to close DB
	#
	def __del__(self): self.closeDB()

	## @brief	connects to the database
	## @param	Name	name of the DB file
	## @return	database handle connected to the file
	## @version	2016/01/21:
	#		Exits in case of unknown tables.
	#		Missing tables will be created.
	#
	## @version	2016/09/01:
	#		writing Qm methods in separate function
	#
	def initDB(self):
		self.log.printComment('connecting to: %s' %self.name, onlyBody=True)

		# . connect to db
		connection = sqlite3.connect(database=self.name,timeout=self.timeout)
		# create cursor
		cursor = connection.cursor()

		# . check existence of tables and for conflicts to prevent data corruption
		cursor.execute('SELECT name,sql FROM sqlite_master WHERE type="table"')
		existing_tables = cursor.fetchall()

		for ex_name,ex_create in existing_tables:
			matched = False
			for name,create in self.sql_tables:
				if name == ex_name:
					matched = True
					ex_columns = [c.strip() for c in ex_create.split('(')[1].split(')')[0].split(',')]
					columns    = [c.strip() for c in    create.split('(')[1].split(')')[0].split(',')]
					if sorted(columns) != sorted(ex_columns):
						# . existing table is known, but has different structure
						self.log.printIssue(Text='dbhandler.initDB: The table "%s" is already existing, but has not the desired structure!\nThe existing structure looks like: %s\nYou may be using the wrong database, or you should update the source code.' %(name,ex_create), Fatal=True)
					break
			if not matched:
				# . existing table is unknown
				self.log.printIssue(Text='dbhandler.initDB: The existing table "%s" is unknown! You may be using the wrong database, or you should update the source code.' %(ex_name), Fatal=True)

		# . create missing tables
		for name,create in self.sql_tables:
			matched = False
			for ex_name,ex_create in existing_tables:
				if name == ex_name:
					matched = True
					break
			if not matched:
				self.log.printComment(Text='table "%s" ist not existing. will be created with: %s' %(name,create), onlyBody=False)
				cursor.execute(create)
				connection.commit()

		# . return connection
		return connection, cursor

	## @brief	closes the database
	## @return	None
	## @version	2016/01/21
	#
	def closeDB(self):
		self.commit()
		self.database.close()
		self.log.printComment('closing: %s' %self.name, onlyBody=True)
		return

	## @brief	commits changes in DB
	## @return	bool, success=1, failure=0
	## @version	2016/05/17:
	#		added. necessary because sometimes the DB is
	#		locked after executing data changing command.
	#
	def commit(self,Maxretry=10,Wait=3):
		i = 0
		while i < Maxretry:
			try:
				self.database.commit()
				return True
			except sqlite3.OperationalError:
				i += 1
				self.log.printIssue(Text='Attempt to commit while DB is locked. Retry (%d/%d).' %(i,Maxretry), Fatal=False)
				time.sleep(Wait)
		return False
	
	def write(self, Sqlite, Record=(), Maxretry=10, Wait=3):
		i = 0
		while i < Maxretry:
			try:
				if Record: self.cursor.execute(Sqlite, Record)
				else: self.cursor.execute(Sqlite)
				self.commit()
				return True
			except sqlite3.OperationalError:
				i += 1
				self.log.printIssue(Text='Attempt to write to locked DB. Retry (%d/%d).' %(i,Maxretry), Fatal=False)
				time.sleep(Wait)
		return False
			

	## @brief	write time and QM methods to info table
	## @version	2016/09/01:
	#		moved here from initDB()
	#		only compare and write method names if given command string also contains options.
	#
	def writeMethods(self, QMmethods):
		#c_new = [' '.join([c for c in tmp.split() if 'opt' not in c and 'freq' not in c and 'int' not in c and 'scf' not in c]) for tmp in QMmethods]
		c_new = QMmethods
		self.cursor.execute('SELECT commandS,commandTS FROM info')
		result = self.cursor.fetchall()
		if result != []:
			c_old = [' '.join([c for c in tmp.split() if 'opt' not in c and 'freq' not in c and 'int' not in c and 'scf' not in c and 'maxdisk' not in c and 'symmetry' not in c]) for tmp in result[0]]
			#c_old = result[0]
			if c_old[:2] == ['','']:
				self.write('UPDATE info SET commandS=?,commandTS=?', (c_new[0],c_new[1]))
			elif c_new[:2] == ['','']:
				pass
			elif c_old[:2] != c_new[:2]:
				self.log.printIssue(Text='dbhandler.writeMethods: The QM methods you want to use the DB for ("%s"/"%s") do not match those stored in the DB ("%s"/"%s").' %tuple(c_new[:2]+c_old[:2]), Fatal=True)
		else:
			record = (time.asctime(),c_new[0],c_new[1],'')
			self.write('INSERT INTO info (date,commandS,commandTS,comment) VALUES (?,?,?,?)', record)

	## @brief	puts info about usage and method into DB
	## @param	User		editing user (e.g. login name)
	## @param	Command		argument line used for g09
	## @param	Comment		any appropriate DB-describing text
	## @version	2016/03/01:
	#		sets the date to now automatically.
	#		saves the history of who used the DB.
	#
	def updateInfo(self, User='', CommandS='', CommandTS='', Comment=''):
		query = 'UPDATE info SET date="' + time.asctime() + '"'
		if User != '':
			self.cursor.execute('SELECT user FROM info')
			result = self.cursor.fetchone()
			if result[0] != None: users = result[0].split()
			else: users = []
			if User not in users: users.append(User)
			query += ', user="' + ' '.join(users) + '"'
		if CommandS != '':
			query += ', commandS="' + CommandS + '"'
		if CommandTS != '':
			query += ', commandTS="' + CommandTS + '"'
		if Comment != '':
			query += ', comment="' + Comment + '"'
		self.write(query, ())
		self.commit()

	## @brief	set a new symmetry number
	## @param	Smile		SMILES identifier of species
	## @param	Conf		Conformer Number
	## @param	SymmetryNo	new symmetry number
	## @version 2016/31/03:
	#		was made necessary, because g09 does not always
	#		returns the correct rotational symmetry number
	#
	def updateSymmetryNumber(self, Smile, Conformer, SymmetryNo):
		if ':' in Smile:
			self.write('UPDATE reactionTS SET sym=? WHERE smile=? AND conformer=?', (SymmetryNo,Smile,Conformer))
		else:
			self.write('UPDATE species SET sym=? WHERE smile=? AND conformer=?', (SymmetryNo,Smile,Conformer))

	## @brief	adds a new species to the database
	## @param	Data		data package, consisting of:
	##		Smile		SMILES identifier of species
	##		Molmass		molecular mass
	##		Geometry	type and xyz information on atoms
	##		CSPenergy	corrected single point energy
	##		Tr		rotational temperatures vector from QM
	##		V0		harmonic frequencies
	##		Rotor		internal rotations
	##		FFenergy	potential energy from Force Field
	##		Ip		principle inertia axes vector from MD
	##		Spin		spin multiplicity
	##		Htherm		Ha0
	## @param	Erange		energy interval to match existing [kcal/mol]
	## @param	Irange		inertia interval to match existing [-]
	## @return	conformer number of species (known or new)
	## @version	2016/02/14
	#
	def addSpecies(self, Molinfo, Erange=1, Irange=2.0, RetConf=False):
		inDB = False
		added = False

		# . split up data
		smi       = Molinfo['smi']		# reaction SMILE
		molmass   = Molinfo['M']		# molecular mass
		geo       = Molinfo['geo']		# position data  {id:[type,[x,y,z]]}*
		cSPE      = Molinfo['cSPE']		# corr. single point energy [hartree/particle]
		tr        = Molinfo['Tr']		# external rot. temp.  []*3
		rotsym    = Molinfo['sym']		# number of rotational symmetries
		v0        = Molinfo['v0']		# harmonic frequencies  []*
		rotor     = Molinfo['rotor']		# internal rotations and temperatures  {freq:T}
		ffenergy  = Molinfo['eff']		# force field potential energy
		ffi       = Molinfo['Ip']		# force field principle inertia  []*3
		spin      = Molinfo['spin']		# spin multiplicity
		htherm    = Molinfo['Htherm']		# Ha0
		geostring = ';'.join(['%d,%.8f,%.8f,%.8f' %(geo[id][0],geo[id][1][0],geo[id][1][1],geo[id][1][2]) for id in geo])

		# . hartree <> kcal conversion
		erange = Erange / 627.509

		# . make a preselect by qm energy
		self.cursor.execute('SELECT tr1,tr2,tr3,sym,conformer FROM species WHERE smile=?  AND energy BETWEEN ? AND ?', (smi,cSPE-erange,cSPE+erange))
		species = self.cursor.fetchall()

		# . set up inertia tensor (reciprocal temperature)
		#   Theta_i = h*h / 8*pi*pi*kB*I_i
		if len(tr) == 1:
			tr = [tr[0],tr[0],0.0]
		elif len(tr) == 2:
			tr = [tr[0],tr[1],0.0]
		else:
			tr = [tr[0],tr[1],tr[2]]
		i_species = [0.,0.,0.]
		for i in range(3):
			if tr[i] != 0.0:
				i_species[i] = 1 / tr[i]

		# . in case for many conformers within Irange, find closest
		conf = -1
		mindist = 1000.
		maxsym = rotsym
		for tr1,tr2,tr3,sym,conformer in species:
			# . set up comparison
			i_compare = []
			for t in [tr1,tr2,tr3]:
				if t != 0.0:
					i_compare.append(1/t)
				else:
					i_compare.append(0.0)
			dist = sum( [(i_species[i] - i_compare[i])**2 for i in range(3)] )

			# . found?
			if dist < 3.0*Irange**2:
				inDB = True
				if dist < mindist:
					mindist = dist
					conf = conformer
					maxsym = max(rotsym,sym)

		# . species was not found
		if not inDB:
			self.cursor.execute('SELECT conformer FROM species WHERE smile=?', (smi,))
			result = self.cursor.fetchall()
			if result == []:
				conf = 0
			else:
				conf = max(result)[0] + 1

			# . assemble record. ffi and QMTr are assumed to be lists.
			if len(ffi) == 1:
				ffi = [ffi[0],ffi[0],0.0]
			elif len(ffi) == 2:
				ffi = [ffi[0],ffi[1],0.0]
			record = (smi,conf,molmass,cSPE) + tuple(tr) + (rotsym,spin,htherm,ffenergy) + tuple(ffi) + (geostring,)

			# . write
			self.write('INSERT INTO species (smile,conformer,molmass,energy,tr1,tr2,tr3,sym,spin,htherm,ff_energy,ff_i1,ff_i2,ff_i3,geometry) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', record)
			for v in v0:
				record = (smi,conf,v)
				self.write('INSERT INTO hofreq (smile,conformer,frequency) VALUES (?,?,?)', record)
			for v in rotor:
				record = (smi,conf,v,rotor[v])
				self.write('INSERT INTO rottemp (smile,conformer,frequency,temperature) VALUES (?,?,?,?)', record)
			self.commit()
			
			added = True

		# . species was found
		else:
			# . update rotational symmetry to the max
			self.updateSymmetryNumber(Smile=smi,Conformer=conformer,SymmetryNo=maxsym)

		if RetConf: return conf
		else: return added

	## @brief	insert QM data from a finished TS job
	## @param	Data		QM Data
	## @param	Erange		cSP-Energy range to identify [kcal/mol]
	## @param	Irange		range to identify via rot. temp
	## @return	conformer number of reaction (known or new)
	## @version	2016/02/14:
	#		only barrier reactions
	#
	## @version	2016/03/29:
	#		rotational temp. (Tr) withohut symmetry number
	#		symmetry number is seperate now
	#
	def addReaction(self, Molinfo, Erange=1, Irange=2.0, RetConf=False):
		added = False

		# . hartree to kcal/mol conversion
		erange = Erange / 627.509

		# . split up data
		smi      = Molinfo['smi']	# reaction SMILE
		molmass  = Molinfo['M']		# molecular mass
		geo      = Molinfo['geo']	# position data  {id:[type,[x,y,z]]}*
		cSPE     = Molinfo['cSPE']	# corr. single point energy [hartree/particle]
		tr       = Molinfo['Tr']	# external rot. temp.  []*3
		rotsym   = Molinfo['sym']	# number of rotational symmetries
		v0       = Molinfo['v0']	# harmonic frequencies  []*
		rotor    = Molinfo['rotor']	# internal rotations and temperatures  {freq:T}
		ffenergy = Molinfo['eff']	# force field potential energy
		ffi      = Molinfo['Ip']	# force field principle inertia  []*3
		spin     = Molinfo['spin']	# spin multiplicity
		htherm   = Molinfo['Htherm']	# Ha0
		geostring = ';'.join(['%d,%.8f,%.8f,%.8f' %(geo[id][0],geo[id][1][0],geo[id][1][1],geo[id][1][2]) for id in geo])

		# . correct tr, there must be 3 values in the DB
		# . set up inertia tensor (reciprocal temperature)
		#   Theta_i = h*h / 8*pi*pi*kB*I_i
		if len(tr) == 1:
			i_reaction = [1/tr[0], 1/tr[0], 0.0]
			tr = [tr[0],tr[0],0.0]
		elif len(tr) == 2:
			i_reaction = [1/tr[0], 1/tr[1], 0.0]
			tr = [tr[0],tr[1],0.0]
		else:
			i_reaction = [1/tr[0], 1/tr[1], 1/tr[2]]
			tr = [tr[0],tr[1],tr[2]]

		# . normal TS reaction
		self.cursor.execute('SELECT conformer,emax,emin,i1max,i1min,i2max,i2min,i3max,i3min,tr1,tr2,tr3,sym FROM reactionTS WHERE smile=? AND energy BETWEEN ? AND ?', (smi,cSPE-erange,cSPE+erange))
		result = self.cursor.fetchall()

		# . look for conformer
		match = -1
		for conf,emax,emin,i1max,i1min,i2max,i2min,i3max,i3min,tr1,tr2,tr3,sym in result:

			i_compare = []
			for t in [tr1,tr2,tr3]:
				if t != 0.0:
					i_compare.append(1/t)
				else:
					i_compare.append(0.0)

			dist = sum( [(i_reaction[i] - i_compare[i])**2 for i in range(3)] )

			if dist < 3.0*Irange**2:
				match = conf
				update = (
					max(emax,ffenergy),
					min(emin,ffenergy),
					max(i1max,ffi[0]),
					min(i1min,ffi[0]),
					max(i2max,ffi[1]),
					min(i2min,ffi[1]),
					max(i3max,ffi[2]),
					min(i3min,ffi[2]),
					# . rot. symmetry, take biggest
					max(sym,rotsym)
				)
				# . seek no further
				break

		# . already in DB, update FF energy and inertia range
		if match != -1:
			self.write('UPDATE reactionTS SET emax=?,emin=?,i1max=?,i1min=?,i2max=?,i2min=?,i3max=?,i3min=?,sym=? WHERE smile=? AND conformer=?', update+(smi,match))
			added = True
		# . new TS
		else:
			self.cursor.execute('SELECT conformer FROM reactionTS WHERE smile=?', (smi,))
			result = self.cursor.fetchall()
			if result != []:
				newconf = max(result)[0] + 1
			else:
				newconf = 0

			# . assemble record
			record = (smi,newconf,cSPE,molmass,tr[0],tr[1],tr[2],rotsym,spin,htherm,ffenergy,ffenergy,ffi[0],ffi[0],ffi[1],ffi[1],ffi[2],ffi[2],geostring)
			# . create new entries
			self.write('INSERT INTO reactionTS (smile,conformer,energy,molmass,tr1,tr2,tr3,sym,spin,htherm,emin,emax,i1min,i1max,i2min,i2max,i3min,i3max,geometry) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', record)
			for freq in v0:
				self.write('INSERT INTO tsfreq (smile,conformer,frequency) VALUES (?,?,?)', (smi,newconf,freq))
			for freq in rotor:
				self.write('INSERT INTO tstemp (smile,conformer,frequency,temperature) VALUES (?,?,?,?)', (smi,newconf,freq,rotor[freq]))

			self.commit()
			match = newconf
			added = True

		if RetConf: return match
		else: return added

	## @brief	insert QM data from a finished TS job (barrierless)
	## @param	Smile		SMILES
	## @param	M		molecular mass
	## @param	Geometry	geometry info: 'type,x,y,z;...'
	## @param	FFrate		rate for barrierless
	## @param	Temperature	simulated temperatuer
	## @version	2016/03/17:
	#		only barrierless reactions
	#
	## @version	2016/06/15:
	#		make Geometry optional
	#
	## @version	2016/12/16:
	#		introduced error of rate
	#
	def addReactionB(self, Smile, M, FFrate, Temperature, Klo, Kup, N, Geometry=''):
		# . arithmetic mean for rate and error
		self.cursor.execute('SELECT rate,klo,kup,n,samplesize FROM reactionB WHERE smile=? AND temperature=?', (Smile,Temperature))
		result = self.cursor.fetchone()
		if result:
			oldrate, oldklo, oldkup, n, s = result

			### TODO: verrechnung verschiedener raten aus verschiedenen trajektorien
			### hier: arithmetisches mittel
			newrate = (oldrate * s + FFrate) / float(s+1)
			newklo  = (oldklo  * s + Klo)    / float(s+1)
			newkup  = (oldkup  * s + Kup)    / float(s+1)
			newn    = n+N
			self.write('UPDATE reactionB SET rate=?,klo=?,kup=?,n=?,samplesize=? WHERE smile=? AND temperature=?', (newrate,newklo,newkup,newn,s+1,Smile,Temperature))
		else:
			self.write('INSERT INTO reactionB (smile,molmass,temperature,rate,klo,kup,n,samplesize,geometry) VALUES (?,?,?,?,?,?,?,?,?)', (Smile,M,Temperature,FFrate,Klo,Kup,N,1,Geometry))

		self.commit()

	## @brief	check whether data was created with specific arguments
	## @param	Command:	g09 argument string to check
	## @return	bool (yes/no) whether data in DB was created with these args
	## @version	2016/03/01:
	#		command line arguments for g09 only match, if they
	#		have the same structure within: e.g.
	#		"opt=(calcfc,tight)" doesn't match
	#		"opt=(tight,calcfc)", whereas
	#		"int=ultrafine cbs-qb3" matches
	#		"cbs-qb3 int=ultrafine"
	#
	## @version	2016/09/01:
	#		'getMethodFromCommand' has been removed;
	#		use command as given in 'simulation.py' for
	#		varification
	#
	def verifyMethod(self, Command, TS=True):
		if TS: cmd = 'commandTS'
		else: cmd = 'commandS'
		self.cursor.execute('SELECT %s FROM info' %cmd)
		result = self.cursor.fetchone()
		return result[0] == Command or result[0] == ''

	## @brief	quick lookup by FF energy/inertia whether species in DB
	## @param	Smile	SMILES identifier of species
	## @param	Energy	potential energy to identify conformer
	## @param	Ip	vector of principle axes of inertia
	## @param	Erange	max. difference in potential energy [kcal/mol]
	## @param	Irange	max. difference in rmse(Ip-Ip_i) [-]
	## @return	bool yes/no whether in DB
	## @version	2016/02/14
	#
	def speciesInDB(self, Smile, Energy, Ip, Erange=1, Irange=2.0):
		# . make a preselect by ff energy
		self.cursor.execute('SELECT conformer,ff_energy,ff_i1,ff_i2,ff_i3 FROM species WHERE smile=? AND ff_energy BETWEEN ? AND ?', (Smile,Energy-Erange,Energy+Erange))
		result = self.cursor.fetchall()

		# . check ff inertia ranges
		inDB = False
		for record in result:
			ip = record[2:5]
			if ( (Ip[0]-ip[0])**2 + (Ip[1]-ip[1])**2 + (Ip[2]-ip[2])**2 ) < 3.0*Irange**2:
				inDB = True

		return inDB

	## @brief	quick search in barrierless table for a smile
	## @param	Smile
	## @return	bool, yes/no whether found in DB
	## @version	2016/02/14:
	#		This method is used in sim.py to check whether a
	#		reaction has been previously determined to be
	#		barrierless
	#
	def barrierlessReactionInDB(self, Smile):
		self.cursor.execute('SELECT smile FROM reactionB WHERE smile=?', (Smile,))
		result = self.cursor.fetchall()

		return (result != [])

	## @brief	search normal TS reaction table for smile
	## @param	Mode		use 'quick' to search generally
	##				otherwise search for specific
	##				energy and inertia
	## @param	Smile		Smile
	## @param	FFenergy	Force Field Energy
	## @param	FFIntertia	Force Field Inertia
	## @param	Erange		energy search range [kcal/mol]
	## @param	Irange		inertia comp. search range [?]
	## @return	bool, yes/no whether found in DB
	## @version	2016/02/14:
	#		This method is used in sim.py to check whether a
	#		reaction has been previously determined to be not
	#		barrierless
	#
	def barrierReactionInDB(self, Smile, Mode='quick', FFenergy=0, FFinertia=[0,0,0], Erange=1, Irange=2.0):

		if Mode == 'quick':
			self.cursor.execute('SELECT smile FROM reactionTS WHERE smile=?', (Smile,))
			result = self.cursor.fetchall()

			return (result != [])
		else:
			# . make a preselect by energy
			self.cursor.execute('SELECT i1min,i1max,i2min,i2max,i3min,i3max FROM reactionTS WHERE smile=? AND emin > ? AND emax < ?', (Smile, FFenergy-Erange, FFenergy+Erange) )
			result = self.cursor.fetchall()

			# . look in inertia range for a match,
			#   extend search range by Irange.
			inDB = False
			for i in range(len(result)):
				iplimits = result[i][0:6] # min,max ...
				if (iplimits[0]+Irange > FFinertia[0] and iplimits[1]-Irange < FFinertia[0]
				and iplimits[2]+Irange > FFinertia[1] and iplimits[3]-Irange < FFinertia[1]
				and iplimits[4]+Irange > FFinertia[2] and iplimits[5]-Irange < FFinertia[2] ):
					inDB = True

			return inDB

	## @brief	convert geo info from string to dict
	## @param	Geostring	geometric info of format:
	##				"type1,x,y,z;type2,..."
	## @return	dict of geo, format: {id:[type,[x,y,z]]}
	## @version	2016/03/01:
	#		first id is always 1
	#
	def convertGeometryString(self, Geostring):
		i = 0
		geo = {}
		atoms = Geostring.split(';')
		for atom in atoms:
			if atom != '':
				i += 1
				type,x,y,z = atom.split(',')
				geo[i] = [int(type),[float(x),float(y),float(z)]]

		return geo

	## @brief	returns all data for a TS reaction
	## @param	Smi	reaction smile
	## @return	a list of lists containing the following properties:
	##		M	molecular mass (float)
	##		cSPE	corrected single point energy (float)
	##		Tr	external rotational temperatures (list of 3)
	##		v0	all harmonic frequencies (list)
	##		rotor	internal rotations and temperatures (dict of (freq:temp))
	##		spin	spin multiplicity
	##		form	geometric info
	##		hterm	Ha0 in hartree/particle
	## @version	2016/02/14:
	#		if smile is not found, an empty list is returned
	#		v0 and rotor can be empty
	#
	## @version	2016/02/14:
	#		returns species as well as reactions.
	#
	## @version	2016/04/07:
	#		added symmetry number
	#
	def getConf(self, Smi):
		conf = []
		if ':' in Smi: TS = True
		else: TS = False

		if TS: table = 'reactionTS'
		else: table = 'species'
		self.cursor.execute('SELECT conformer,molmass,energy,tr1,tr2,tr3,sym,spin,geometry,htherm FROM %s WHERE smile=?' %table, (Smi,))
		result = self.cursor.fetchall()

		for record in result:
			coid = record[0]
			mass = record[1]
			cspe = record[2]
			tr = record[3:6]
			sym  = record[6]
			spin = record[7]
			form = self.convertGeometryString(record[8])
			htherm = record[9]

			# . harmonic frequencies
			if TS: table = 'tsfreq'
			else: table = 'hofreq'
			self.cursor.execute('SELECT frequency FROM %s WHERE smile=? AND conformer=?' %table, (Smi,coid))
			r = self.cursor.fetchall()
			v0 = [v[0] for v in r]
			# . internal rotor
			if TS: table = 'tstemp'
			else: table = 'rottemp'
			self.cursor.execute('SELECT frequency,temperature FROM %s WHERE smile=? AND conformer=?' %table, (Smi,coid))
			r = self.cursor.fetchall()
			rotor = dict([(v[0],v[1]) for v in r])

			molinfo = {'M':mass, 'cSPE':cspe, 'Tr':tr, 'sym':sym, 'v0':v0, 'rotor':rotor, 'spin':spin, 'geo':form, 'Htherm':htherm, 'conformer':coid}
			conf.append(molinfo)

		if conf == []:
			self.log.printIssue(Text='No Data in DB for "%s"' %Smi, Fatal=False)
			conf = [{'M':None, 'cSPE':None, 'Tr':None, 'sym':None, 'v0':None, 'rotor':None, 'spin':None, 'geo':None, 'Htherm':None, 'conformer':None}]

		return conf

	## @brief	retrieves molecule geometry from species table
	## @param	Smile		SMILES identifier of species
	## @param	Conf		Conformer Number
	## @return	list of atoms with type and xyz: [[type,x,y,z], ... ]
	## @version	2016/01/23:
	#		also usable for reactions
	#
	def getGeometry(self, Smile, Conf=0):
		geometry = []
		if ':' in Smile:
			# . read reaction from db
			self.cursor.execute('SELECT geometry FROM reactionTS WHERE smile=? AND conformer=?', (Smile,Conf))
			result = self.cursor.fetchone()
			if result == None:
				# . read barrierless from db
				self.cursor.execute('SELECT geometry FROM reactionB WHERE smile=?', (Smile,))
				result = self.cursor.fetchone()
		else:
			# . read species from db
			self.cursor.execute('SELECT geometry FROM species WHERE smile=? AND conformer=?', (Smile,Conf))
			result = self.cursor.fetchone()

		# . convert to [[type,x,y,z], [type,x,y,z] ... ]
		if result != None:
			geostring = result[0]
			exception = False
			for atom in geostring.split(';'):
				fields = atom.split(',')
				try:
					geometry.append([int(fields[0]),float(fields[1]),float(fields[2]),float(fields[3])])
				except:
					exception = True
					geometry = []
					break
			if exception:
				self.log.printIssue(Text='The geometry information for "%s" seems to be corrupt and is skipped' %(Smile), Fatal=False)

		return geometry

	## @brief	return all species smiles from DB
	## @return	list of unique smiles codes in species db
	## @version	2016/03/17
	def getSpecies(self):
		self.cursor.execute('SELECT smile FROM species WHERE conformer=0')
		result = self.cursor.fetchall()
		return [r[0] for r in result]

	## @brief	return all reaction smiles from DB
	## @return	list of unique smiles codes in TS reaction table
	## @version	2016/03/17
	#
	def getReactions(self):
		return self.getBarrierlessReactions() + self.getTSReactions()

	## @brief	return all TS reaction smiles from DB
	## @return	list of unique smiles codes in TS reaction table
	## @version	2016/03/17
	#
	def getTSReactions(self):
		self.cursor.execute('SELECT smile FROM reactionTS WHERE conformer=0')
		result = self.cursor.fetchall()
		ret = [r[0] for r in result]
		return ret

	## @brief	return all barrierless reaction smiles from DB
	## @return	list of unique smiles codes in B reaction table
	## @version	2016/05/22
	#
	def getBarrierlessReactions(self):
		self.cursor.execute('SELECT smile FROM reactionB')
		result = self.cursor.fetchall()
		ret = list(set([r[0] for r in result]))
		return ret


	## @brief	return all data for a barrierless reaction
	## @return	3 lists of temperature, rate konstant and error
	## @version	2016/03/17
	#
	def getBarrierlessData(self, Reac):
		self.cursor.execute('SELECT temperature,rate,klo,kup,n FROM reactionB WHERE smile=?', (Reac,))
		result = self.cursor.fetchall()
		[t,k,klo,kup,n] = list(zip(*result))
		return t,k,klo,kup,n

	## @brief	prints users, date, method and comment of DB
	## @version	2016/03/01
	#
	def printInfo(self):
		self.cursor.execute('SELECT date,commandS,commandTS,comment FROM info')
		info = self.cursor.fetchone()
		if info:
			self.log.printComment('Database info:\n date: %s\n g09 Species: %s\n g09 TS: %s\n%s' %info, onlyBody=True)
		else:
			self.log.printComment('No Database Info available', onlyBody=True)

