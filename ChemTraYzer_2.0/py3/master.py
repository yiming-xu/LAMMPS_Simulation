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
## @file	master.py
## @author	Malte Doentgen, Felix Schmalz
## @date	2018/07/13
## @brief	main program
## @version:
#		program intended to bring together analyzing.py and harvesting.py
#		only section that actually writes to the DB
#		analyzing: reac-files with rates etc
#		harvesting: qm-files in qm-folder with frequencies etc
#		both created previously through simulation.py

import sys
import os
import numpy

import harvesting
import analyzing
import dbhandler
import log as Log


## @brief	read the trajectory file from the MD simulation
## @param	Work		Filename
## @return	worktime	biggest timestep
## @return	temperature	temp. of simulation
## @return	reaction	ocurred reactions
## @version	2016/02/24:
#		copied here from analyzing.py
#
def readWork(Work):
	reader = open(Work, 'r')
	done = False
	worktime = -1
	temperature = -1
	volume = -1
	timestep = -1
	reaction = {}
	while not done:
		line = reader.readline().strip()
		words = line.split(':')
		if 'ON THE FLY' in line:
			try: temperature = float(line.split()[4])
			except: pass
			try: volume = float(line.split()[5])
			except: pass
			try: timestep = float(line.split()[6])
			except: pass
		try: int(words[0])
		except:
			if len(words) == 1 and words[0] == '': done = True
			continue
		if len(words) > 1:
			try:
				int(words[0])
				if int(words[0]) not in reaction:
					reaction[int(words[0])] = [[words[1].split('\n')[0].split(','), words[2].split('\n')[0].split(',')]]
				else:
					reaction[int(words[0])].append([words[1].split('\n')[0].split(','), words[2].split('\n')[0].split(',')])
			except: pass
		else: done = True
	reader.close()
	if reaction and (worktime == -1 or worktime < max(reaction)): worktime = max(reaction)
	return worktime, temperature, volume, timestep, reaction

def receiveInput(argv, log=Log.Log(Width=70)):
	# . interpret input
	options = {}
	options['reac'] = []
	i = 0
	while not argv[i].startswith('-'):
		options['reac'].append(argv[i])
		i += 1
	while i < len(argv)-1:
		arg = argv[i]
		opt = argv[i+1]
		if arg.startswith('-'):
			if opt.startswith('-'): log.printIssue(Text='Missing value for option "%s". Will be ignored.' %arg, Fatal=False)
			else:
				try: tmp = float(opt)
				except: tmp = opt
				options[arg[1:].lower()] = tmp
				i += 1
		i += 1

	# . defaults: analyzing
	if 'main' not in options: options['main'] = 'Default'
	if 'step' not in options: options['step'] = 0.1
	if 'start' not in options: options['start'] = 0
	if 'end' not in options: options['end'] = -1

	# . defaults: harvesting
	if 'source' not in options: options['source'] = 'QM'
	if 'dbase' not in options: options['dbase'] = 'chemtrayzer.sqlite'
	if 'files' not in options: options['files'] = 'tmp_'
	if 'type' not in options: options['type'] = 'log'
	if 'fail' not in options: options['fail'] = 'keep'
	if 'norm' not in options: options['norm'] = 'keep'

	# . defaults: master
	if 'all' not in options: options['all'] = False
	if 'pressure' not in options: options['pressure'] = 1E5

	return options

#######################################
if __name__ == '__main__':
	log = Log.Log(Width=70)

	###	HEADER
	#
	text = ['<.reac-file(s)> [-main <main>] [-step <timestep>] [-start <start>] [-end <end>]',
		'-source <QM folder> -dbase <database> [-files <filename>] [-type <file extension>] [-fail <behavior on failure>] [-norm <behavior on success>]',
		'[-all <kinetic model size>] [-pressure <pressure/Pa>]',
		'',
		'analyzing options:',
		'-reac: name of .reac file produced via processing.py / simulation.py',
		'-main: name for output of analyzing.py (default="Default")',
		'-step: give the timestep of the simultaion in [fs] (default=0.1)',
		'-start: skip timesteps smaller than <start> for rate computation (default=0)',
		'-end: end-flag for rate computation (default=-1)',
		'',
		'harvesting options:',
		'-source: name of folder containing the QM results (default="QM")',
		'-dbase: name of the database for reading and writing (default="chemtrayzer.sqlite")',
		'-files: pre-fix for QM files (default="tmp_")',
		'-type: file-extension for QM files (default="log")',
		'-fail: action taken, if harvesting of a file fails. Possible options are: "keep", "delete". (default="keep")',
		'-norm: action taken, if harvesting of a file succeeds. Possible options are: "keep", "delete". (default="keep")',
		'',
		'master options:',
		'-all: if "1", write all database entries to the kinetic model, elif "0", only for species and reactions observed during trajectory simulation (default="0")',
		'-pressure: specifiy pressure for computing the translational partition function, in [Pa] (default=1E5)',
		'']
	log.printHead(Title='ChemTraYzer - Chemical Trajectory Analyzer', Version='2016-09-01', Author='Malte Doentgen, LTT RWTH Aachen University', Email='chemtrayzer@ltt.rwth-aachen.de', Text='\n\n'.join(text))

	###	INTERPRET INPUT
	#
	if len(sys.argv) > 1: argv = sys.argv[1:]
	else:
		log.printComment(Text=' Use keyboard to type in commands. Use <strg>+<c> or write "done" to proceed.', onlyBody=False)
		argv = []; done = False
		while not done:
			try: tmp = input('')
			except: done = True
			if 'done' in tmp.lower(): done = True
			else:
				if '!' in tmp: argv += tmp[:tmp.index('!')].split()
				else: argv += tmp.split()
	options = receiveInput(argv, log=log)
	if not options['reac']:
		log.printIssue(Text='No .reac file supplied, cannot process trajectory data', Fatal=True)
	elif not options['source']:
		log.printIssue(Text='No source folder supplied, cannot process QM data', Fatal=True)
	elif not options['dbase']:
		log.printIssue(Text='No database supplied, cannot store trajectory or QM data', Fatal=True)
	db = dbhandler.Database(Name=options['dbase'],Timeout=10)

	###	READ TRAJECTORY FILES
	#
	reaction = {}
	log.printComment(Text='The following list of work files will be merged into a single list of reactions. All virtual reactions required for initial molecule creation are stored at time=0 !', onlyBody=False)
	log.printBody(Text=', '.join(f for f in options['reac']), Indent=1)
	log.printBody(Text='', Indent=1)
	tshift = 0.0
	for f in options['reac']:
		wt, T, V, dt, tmp = readWork(f)
		for time in tmp:
			if time+tshift in reaction: reaction[time+tshift] += tmp[time]
			else: reaction[time+tshift] = tmp[time]
		tshift = max(reaction)
		log.printBody(Text='timesteps '+repr(min(tmp))+' to '+repr(wt)+' read from "'+f+'"', Indent=1)
	log.printBody(Text='', Indent=1)

	if T == -1 or V == -1 or dt == -1: log.printIssue(Text='Extracted simulation parameters faulty.', Fatal=True)

	###	REACTION ANALYSIS
	#
	anly = analyzing.Analyzing(Reaction=reaction, Main=options['main'], Start=options['start'], End=options['end'], Vol=V, Timestep=dt)
	anly.writeData()

	###	QUANTUM ANALYSIS
	#
	harv = harvesting.Harvesting(DBhandle=db)
	# . test DB on reference species (they may not be there)
	#   it's user responsability, if they are missing (produces wrong NASA polynomials)
	harv.getHa0()
	done, failed, total, barrierless = harv.harvestFolder(Folder=options['source'], Files=options['files'], Type=options['type'], Fail=options['fail'], Done=options['norm'])
	log.printComment('%d of %d results written to database, %d failures' %(done, total, failed))
	# . set up species list
	species_DB = db.getSpecies()
	if options['all']: species = species_DB
	else: species = [spec for spec in anly.species if spec in species_DB]
	# . fit NASA polynomials for all species
        log.printBody(Text='Fitting NASA parameters ...', Indent=1)
	[harv.fitNASA(Smi=spec, Pressure=options['pressure'], T1=300.0, T2=1000.0, T3=3000.0, dT=10.0) for spec in species]
	harv.writeNASA(Filename=options['main']+'.therm')
        log.printBody(Text='... NASA parameters done', Indent=1)

	###	BARRIERLESS REACTIONS
	#
	# . merge barrierless reactions
	#   ruled out by simulation.py (reac.list.b)
	#   with those ruled out by harvesting.py
	if os.path.exists('reac.dat/reac.list.b'):
		barrierlessfile = open('reac.dat/reac.list.b', 'r')
		for line in barrierlessfile:
			try:
				s = line.strip().split()
				reacA = s[0].split(':')[0].split(',')
				reacB = s[0].split(':')[1].split(',')
				if len(reacA) > len(reacB): reac = s[0]
				else: reac = ','.join(reacB)+':'+','.join(reacA)
				barrierless[reac] = s[1]
			except: log.printIssue(Text='Extracted barrierless data faulty.', Fatal=False)
		barrierlessfile.close()

	# . update DB with barrierless reactions
	#   store added reactions in extra file
	barrierlessdone = {}; barrierlessdonefile = {}
	# . get already added barrierless reactions
	if os.path.exists('reac.dat/reac.list.b.done'): barrierlessdonefile = open('reac.dat/reac.list.b.done', 'r')
	for line in barrierlessdonefile:
		try: smi = line.strip(); barrierlessdone[smi] = True;
		except: pass
	# . update DB
	#   no reactions where reactants=products
	for reac in barrierless:
		reacA = reac.split(':')[0]
		reacB = reac.split(':')[1]
		if reac not in barrierlessdone and sorted(reacA) != sorted(reacB) and reac in anly.rate:
			rate, n  = anly.rate[reac]
			klo, kup = anly.err[reac]
			# . calculate missing association rate constants via equilibrium constant
			if rate == 0.0:	# only dissociation ReaxFF rate constant available
				dG = 0.0
				for smi in reacA.split(','):
					if T > harv.NASA[smi][2][1]: p = harv.NASA[smi][0][0:7]
					else: p = harv.NASA[smi][0][7:14]
					dG -= harv.nasaGRT(T, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
				for smi in reacB.split(','):
					if T > harv.NASA[smi][2][1]: p = harv.NASA[smi][0][0:7]
					else: p = harv.NASA[smi][0][7:14]
					dG += harv.nasaGRT(T, p[0], p[1], p[2], p[3], p[4], p[5], p[6])
				Keq = numpy.exp(-dG) # k(association)/k(dissociation)
				rate = Keq*anly.rate[reacB+':'+reacA][0]
				n = 0
			m = barrierless[reac]
			db.addReactionB(Smile=reac, M=m, FFrate=rate, Temperature=T, Klo=klo, Kup=kup, N=n)
			barrierlessdone[reac] = True
	if os.path.exists('reac.dat/reac.list.b.done'): barrierlessdonefile.close()
	# . write updated list of added barrierless reactions
	barrierlessdonefile = open('reac.dat/reac.list.b.done', 'w')
	for reac in barrierlessdone:
		barrierlessdonefile.write('%s\n' %reac)
	barrierlessdonefile.close()

	###	MECHANISM
	#
	reactions_DB = db.getTSReactions() + db.getBarrierlessReactions()
	if options['all']: species = species_DB; barrierless = db.getBarrierlessReactions(); reactions = reactions_DB
	else:
		species = [spec for spec in anly.species if spec in species_DB]

		reactions = []
		for reac in sorted(anly.reaction):
			reactants = reac.split(':')[0].split(',')
			products  = reac.split(':')[1].split(',')
			allSpeciesInDB = all(reactant in species_DB for reactant in reactants) and  all(product in species_DB for product in products)
			if reac in reactions_DB and allSpeciesInDB: # and reac not in reactions
				reactions.append(reac)

	# . fit Arrhenius rates for all reactions
        log.printBody(Text='Fitting Arrhenius parameters ...', Indent=1)
	for reac in reactions:
		if reac in barrierless: harv.fitBarrierless(Reac=reac)
		else: harv.fitArr(Reac=reac, T1=300.0, T3=3000.0, dT=10.0)
	harv.writeArr(Filename=options['main']+'.chem')
        log.printBody(Text='... Arrhenius parameters done', Indent=1)


