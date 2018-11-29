##################################################################
# The MIT License (MIT)                                          #
#                                                                #
# Copyright (c) 2017 RWTH Aachen University, Malte Doentgen      #
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
# @file	analyzing.py
# @author	Malte Doentgen
# @date	2017/11/06
# @brief	This file contains function required for analyzing
#		reaction lists generated via the 'processing.py'
#		program
#
#  Changelog: 2015/05/05 -> 2017/11/06
#  - trying to import packages 'pygraphviz' and 'matplotlib.pyplot',
#    but passes if not found -> not essenential for basic usage
#  - Class 'Log' moved to separte file, since it is imported by
#    multiple classes in the ChemTraYzer package
#  - 'pygraphviz' and 'matplotlib.pyplot' are optional imports now
#  - 'End', 'Vol', and 'Timestep' have been added to allow for
#    analyzing '.reac' files to a certain timestep ('End') and for
#    computing the actual rate constants with common units
#  - Rate calculation is now limited to storing the latest rate
#    constants (which are most accurate by definition). This helps
#    processing large simulations and avoids running into memory
#    limitations.
#  - rate constant uncertainties are based on a probability analysis.
#    This analysis must be conducted serial, not in parallel
#  - 'dict' objects are replaced by 'list' objects to reduce the
#    memory requirements
#  - '-step' option was expecting integer rather float as input,
#    now changed to float
#

import os
import shutil
import sys
from subprocess import run

import networkx as nx
import numpy
import openbabel
import scipy.optimize
import scipy.special

import log as Log

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('ImportError: matplotlib.pyplot')
except RuntimeError:
    print('RuntimeError: Display available?')


# @class	Analyzing
# @brief	comprises all functions used to represent the simulation
#		results in a consolidated and simple way
# @version	2014/05/27:
#		The reactions extracted by the Processing class are
#		further processed and used to generate 2D
#		representations of the participating species,
#		concentration profiles and to generate the mechanism
#		of reactions.
#


class Analyzing:
    # @brief	constructor for Analyzing class
    # @param	Reaction	dict of reactions with keys
    #				being the timesteps and values
    #				being the reactions taking place
    #				at that time
    # @param	Main		key name of the output files and
    #				depiction folder
    # @param	Start		timestep from which the analysis
    #				should start
    # @param	End		timestep at which analyzing should
    #				stop
    # @param	Vol		simulation volume
    # @param	Timestep	length of single timestep in
    #				[fs]
    # @param	X		confidence interval for uncertainty
    #				estimation (0-1)
    # @param	NoReacSpec	list of species for estimating lower
    #				and upper bounds for not observed
    #				reactions
    # @return	Analysing object
    # @version	2014/06/02:
    #		The constructor requires a list of reactions in
    #		the style <time> : <reactants> : <products>. The
    #		processing tool produces work files in exactly
    #		this format. This list is converted into
    #		concentration over time profiles for the unique
    #		species and unique reactions.
    #
    # @version	2014/09/01:
    #		reactions with exactly the same reactants and
    #		products are neglected.
    #
    # @version	2014/10/01:
    #		time resolved rate constant calculation added
    #
    # @version	2014/12/15:
    #		Start parameter added for skipping the initial
    #		phase of thermal imbalance
    #
    # @version	2015/07/10:
    #		End parameter added for cutting at earlier times.
    #		Default=-1, i.e. run till very end
    #
    # @version	2016/03/08:
    #		Due to memory issues, the rate constant ouput is
    #		reduced to the latest timestep (which has the
    #		best statistics anyway).
    #		The dicts were replaced by lists due to memory
    #		reasons. All functions are updated accordingly.
    #
    # @version	2016/03/15:
    #		volume added as input option. This is used to
    #		compute the bimolecular rate constants.
    #
    # @version	2016/11/17:
    #		rate constant uncertainties based on probability
    #		analysis: X = confidence interval. A list of
    #		species can be supplied for which the lower and
    #		upper bounds for unimolecular and bimolecular
    #		reactions can be estimated. All bimolecular
    #		permutations of the listed species are
    #		considered.
    #
    # @version	2017/01/11:
    #		replaced scipy.optimize.least_squares() by the more
    #		compatible .leastsq()
    #
    # @version	2018/07/10:
    #		better inital guess for k_up to ensure convergence.
    #
    def __init__(self, Reaction={}, Main='Default', Start=0, End=-1, Vol=1.0, Timestep=0.1, X=0.9, NoReacSpec=[]):
        # . initialization of parameters
        #
        self.log = Log.Log(Width=70)
        self.main = Main
        self.conv = openbabel.OBConversion()
        self.Val = {1: 1, 2: 0, 6: 4, 7: 3, 8: 2,
                    9: 1, 10: 0, 16: 6, 17: 1, 18: 0}
        if Reaction:
            tmin = min(Reaction)
            if End < 0:
                tmax = max(Reaction)
            else:
                tmax = End
        self.time = [t for t in sorted(Reaction) if t >= Start and t <= tmax]
        if tmin not in self.time:
            self.time = [tmin] + self.time
        self.time.append(tmax)
        self.ntime = len(self.time)

        # . conversion of reaction "history" to species
        #   concentrations
        #
        self.log.printBody(
            Text='conversion of reaction time history to species concentrations...', Indent=1)
        self.species = {}
        for i in range(self.ntime):
            time = self.time[i]
            print('%.02f%%\r'.rjust(12) % (100*float(i)/self.ntime), end=' ')
            sys.stdout.flush()
            for r in Reaction[time]:
                for reac in r[0]:
                    if not reac:
                        continue
                    if reac not in self.species:
                        self.species[reac] = [0 for j in range(self.ntime)]
                    self.species[reac][i] -= 1
                    for j in range(self.ntime):
                        t = self.time[j]
                        if t > time:
                            self.species[reac][j] = self.species[reac][i]
                for prod in r[1]:
                    if not prod:
                        continue
                    if prod not in self.species:
                        self.species[prod] = [0 for j in range(self.ntime)]
                    self.species[prod][i] += 1
                    for j in range(self.ntime):
                        t = self.time[j]
                        if t > time:
                            self.species[prod][j] = self.species[prod][i]
        print('100.00%'.rjust(12))
        self.log.printBody(Text='... species generated', Indent=5)

        # . correction of virtual reactions
        #
        for spec in self.species:
            iconc = self.species[spec][0]
            if iconc < 0:
                for i in range(self.ntime):
                    self.species[spec][i] += abs(iconc)
        self.log.printBody(Text='... virtual reactions removed', Indent=5)

        # . generation of stoichiometric formulas
        # . counting of atoms per species
        #
        self.formula = {}
        self.natom = {}
        self.conv.SetInAndOutFormats('smi', 'mdl')
        mol = openbabel.OBMol()
        for spec in self.species:
            self.conv.ReadString(mol, spec)
            self.formula[spec] = mol.GetFormula()
            self.natom[spec] = 0
            for atom in openbabel.OBMolAtomIter(mol):
                if atom.GetAtomicNum() != 1:
                    self.natom[spec] += 1 + atom.ImplicitHydrogenCount()
                else:
                    self.natom[spec] += 1
        self.log.printBody(
            Text='... stoichiometric formulas generated and atom number computed', Indent=5)

        # . check of total atom balance per timestep
        #
        total = []
        for i in range(self.ntime):
            total.append(sum(self.natom[spec]*self.species[spec][i]
                             for spec in self.species if self.species[spec][i] > 0))
        if Reaction and min(total) != max(total):
            self.log.printIssue(Text='number of atoms vary between ' +
                                repr(min(total))+' and '+repr(max(total)), Fatal=False)
        self.log.printBody(Text='... balance of elements evaluated', Indent=5)

        self.log.printBody(Text='... species done\n', Indent=1)

        # . conversion of reaction "history" to reaction
        #   concentrations
        # . check of atom balance for single reactions
        #
        self.log.printBody(
            Text='conversion of reaction time history to reaction concentrations...', Indent=1)
        self.reaction = {}
        for i in range(self.ntime):
            time = self.time[i]
            for r in Reaction[time]:
                if not r[0][0] or not r[1][0]:
                    continue
                if sorted(r[0]) != sorted(r[1]):
                    forward = ','.join(sorted(r[0]))+':'+','.join(sorted(r[1]))
                    backward = ','.join(
                        sorted(r[1]))+':'+','.join(sorted(r[0]))
                    if forward in self.reaction:
                        self.reaction[forward][i] += 1
                    elif backward in self.reaction:
                        self.reaction[backward][i] -= 1
                    else:
                        self.reaction[forward] = [0 for j in range(self.ntime)]
                        self.reaction[forward][i] += 1

                    balance = 0
                    for reac in r[0]:
                        balance -= self.natom[reac]
                    for prod in r[1]:
                        balance += self.natom[prod]
                    if balance != 0:
                        self.log.printIssue(
                            Text='Balance of elements failed for '+forward, Fatal=False)
        self.log.printBody(Text='... reactions generated', Indent=5)

        # . inversion of reactions with negative net flux
        #
        for reac in list(self.reaction):
            if sum(self.reaction[reac][i] for i in range(self.ntime)) < 0:
                self.reaction[reac.split(':')[
                    1]+':'+reac.split(':')[0]] = [-self.reaction[reac][i] for i in range(self.ntime)]
                del self.reaction[reac]
        self.log.printBody(
            Text='... negative net flux reactions inverted', Indent=5)
        self.log.printBody(Text='... reactions done\n', Indent=1)

        # . define uncertainty estimation function
        def lmbW(lmb, N, k): return scipy.special.lambertw(
            numpy.around(-lmb*numpy.exp(-lmb/N)/N, 30), k=k)

        def estRateBounds(k, f, N, X): return X + scipy.special.gammaincc(N+1, k*f) - \
            scipy.special.gammaincc(
            N+1, numpy.real(-N*lmbW(lmb=k*f, N=N, k=0)))

        def estNoReacBound(f, X): return -numpy.log(1-X)/f
        #   . unimolecular reactants for additional estimation
        empty = [0 for i in range(self.ntime)]
        for spec in NoReacSpec:
            if spec not in self.species:
                NoReacSpec.remove(spec)
                continue
            reac = '%s:' % (spec)
            if reac not in self.reaction:
                self.reaction[reac] = list(empty)
            if spec not in self.species:
                self.species[spec] = list(empty)
        #   . bimolecular reactants for additional estimation
        for i in range(len(NoReacSpec)):
            for j in range(len(NoReacSpec)-i):
                specA = NoReacSpec[i]
                specB = NoReacSpec[i:][j]
                reac = '%s,%s:' % (specA, specB)
                if reac not in self.reaction:
                    self.reaction[reac] = list(empty)
        self.species[''] = list(empty)

        # . calculate reaction rates
        self.log.printBody(Text='computing rate constants...', Indent=1)
        self.rate = {}
        self.err = {}
        idx = 1
        for i in range(self.ntime):
            if max(self.time) - self.time[-i-1] > 1E3/Timestep:
                idx = i
                break
        fc = 10/(6.022*Vol)  # 1/(vol*NA*1E-24) : [molecules/A3] to [mol/cm3]
        ft = 1E15/Timestep  # 1/fs -> 1/s
        lng = len(self.reaction)
        running = 0
        for reac in sorted(self.reaction):
            print('%.02f%%\r'.rjust(12) % (100*float(running)/lng), end=' ')
            sys.stdout.flush()
            A = reac.split(':')[0].split(',')
            nA = len(A)-1
            B = reac.split(':')[1].split(',')
            nB = len(B)-1
            labelA = ','.join(A)+':'+','.join(B)
            labelB = ','.join(B)+':'+','.join(A)
            if '' not in A:
                self.rate[labelA] = [0.0, 0]
                self.err[labelA] = [0.0, 0.0]
            if '' not in B:
                self.rate[labelB] = [0.0, 0]
                self.err[labelB] = [0.0, 0.0]
            intA = 0.0
            pos = 0
            intB = 0.0
            neg = 0
            for i in range(self.ntime-1):
                # . concentration correction
                corrA = {spec: A.count(spec)-1 for spec in A}
                corrB = {spec: B.count(spec)-1 for spec in B}
                # . compute integrals
                tmpA = 1
                tmpB = 1
                dt = self.time[i+1] - self.time[i]
                #   . reactants
                for spec in A:
                    tmpA *= max(self.species[spec][i] - corrA[spec], 0)
                    corrA[spec] -= 1
                intA += tmpA*dt
                if self.reaction[reac][i] > 0:
                    pos += self.reaction[reac][i]
                #   . products
                for spec in B:
                    tmpB *= max(self.species[spec][i] - corrB[spec], 0)
                    corrB[spec] -= 1
                intB += tmpB*dt
                if self.reaction[reac][i] < 0:
                    neg -= self.reaction[reac][i]
            # . compute rate constants and estimate uncertainties
            if intA > 0:
                self.rate[labelA] = [pos/intA, pos]
                if pos == 0:
                    kup = estNoReacBound(f=intA, X=X)
                    klo = 0.0
                else:
                    # . initial guess for upper bound. for X=0.9 the following empiric approximations apply:
                    #   lambda_lo = N - N^0.6
                    #   lambda_up = N + N^0.6 + 2
                    #   with kup = lambda_up / int
                    #   tested with X=0.8 and X=0.95 as well
                    kup_0 = self.rate[labelA][0] + (pos**0.6+2) / intA
                    kup = scipy.optimize.leastsq(
                        func=estRateBounds, x0=kup_0, args=(intA, pos, X))[0][0]
                    klo = float(
                        numpy.real(-pos*lmbW(lmb=kup*intA, N=pos, k=0)/intA))
                kup *= ft/(fc**nA)
                klo *= ft/(fc**nA)
                self.err[labelA] = [klo, kup]
                self.rate[labelA][0] *= (ft/(fc**nA))
            if intB > 0:
                self.rate[labelB] = [neg/intB, neg]
                if neg == 0:
                    kup = estNoReacBound(f=intB, X=X)
                    klo = 0.0
                else:
                    kup_0 = self.rate[labelB][0] + (neg**0.6+2) / intB
                    kup = scipy.optimize.leastsq(
                        func=estRateBounds, x0=kup_0, args=(intB, neg, X))[0][0]
                    klo = float(
                        numpy.real(-neg*lmbW(lmb=kup*intB, N=neg, k=0)/intB))
                kup *= ft/(fc**nB)
                klo *= ft/(fc**nB)
                self.err[labelB] = [klo, kup]
                self.rate[labelB][0] *= (ft/(fc**nB))
            running += 1
        print('100.00%'.rjust(12))
        self.species.pop('', None)
        self.log.printBody(Text='... rate constants done\n', Indent=1)

        # . indexing of species
        #
        idx = 0
        self.index = {}
        for spec in sorted(self.species):
            self.index[spec] = idx
            idx += 1
        self.log.printBody(
            Text='... list of species indices generated', Indent=1)

        # . indexing of reactions
        #
        idx = 0
        for reac in sorted(self.reaction):
            self.index[reac] = idx
            idx += 1
        self.log.printBody(
            Text='... list of reaction indices generated', Indent=1)

    # @brief	concentration profiles of species and reactions
    #		in latex
    # @param	Plot		list of species or reactions to
    #				be plotted
    # @param	Timestep	length of single timestep in
    #				[fs]
    # @return	True
    # @version	2014/07/10:
    #		The concentration profiles for species and
    #		reactions are plotted in a matplotlib.pyplot
    #		subplot-grid. The method can read species as
    #		SMILES and indices in the format 'S<idx>'.
    #		Reactions are recognized as SMILES in the format
    #		'<A>,<B>,...:<C>,<D>,...' with <A>, <B>, ... are
    #		SMILES of the respective reacting molecules.
    #		Further, indices in the format 'R<idx>' can be
    #		supplied. The indices for species and reactions
    #		are used in the mechanism and labeling species.
    #
    #		Note: while species may be displayed in 3 or
    #		even more columns, displaying reactions in more
    #		than a single column tends to produce useless
    #		output (due to the very long subplot titles)
    #
    # @version	2016/03/08:
    #		dicts replaced by lists
    #

    def drawProfile(self, Plot, Timestep=0.1):
        # . initialization of parameters
        #
        color = ['blue', 'red', 'green', 'cyan', 'purple']
        nsub = len(color)
        if nsub >= len(Plot):
            nsub = len(Plot)
            msub = 1
        elif len(Plot) % nsub == 0:
            msub = len(Plot) // nsub
        else:
            msub = len(Plot) // nsub + 1

        species = []
        reaction = []
        # . reception of SMILES
        for entry in Plot:
            if entry in self.species:
                species.append(entry)
            if entry in self.reaction:
                reaction.append(entry)
        # . reception of indices
        if not species and not reaction:
            tmpSpec = [spec for spec in sorted(self.species)]
            tmpReac = [reac for reac in sorted(self.reaction)]
            for entry in Plot:
                if entry[0] == 'S':
                    try:
                        species.append(tmpSpec[int(entry[1:])])
                    except:
                        None
                if entry[0] == 'R':
                    try:
                        reaction.append(tmpReac[int(entry[1:])])
                    except:
                        None
        self.log.printComment(Text='Plotting a ('+repr(nsub)+','+repr(msub)+') subplot grid of the following species and reactions:\n\n' +
                              ', '.join(spec for spec in species)+'\n\n'+', '.join(reac for reac in reaction), onlyBody=False)
        #

        ax = []
        i = 0
        # . generation of species concentration plots
        #
        for spec in species:
            time = [float(t)*Timestep/1000000 for t in self.time]
            conc = [self.species[spec][i] for i in range(self.ntime)]
            for t in list(time[:-2]):
                idx = len(time) - 1 - time[::-1].index(t)
                time.insert(idx+1, time[idx+1])
                conc.insert(idx, conc[idx])
            if i % nsub != 0:
                ax.append(plt.subplot2grid((nsub, msub), (i % nsub, i //
                                                          nsub), sharex=ax[i-i % nsub], colspan=1, rowspan=1))
            else:
                ax.append(plt.subplot2grid((nsub, msub), (i %
                                                          nsub, i//nsub), colspan=1, rowspan=1))
                [plt.setp(ax[j].get_xticklabels(), visible=False)
                 for j in range(i-nsub, i-1) if i > 0]
            ax[-1].set_yticks(list(range(0, int(max(conc))+2,
                                         int((max(conc)-min(conc))*float(nsub)*0.1)+1)))
            ax[-1].set_ylim(0, max(conc)+1)
            ax[-1].set_title('S'+repr(self.index[spec]) +
                             ': '+self.formula[spec])
            ax[-1].plot(time, conc, color=color[i % nsub])
            ax[-1].grid()
            i += 1

        # . generation of reaction concentration plots
        #
        for reac in reaction:
            time = [float(t)*Timestep /
                    1000000 for t in sorted(self.reaction[reac])]
            conc = [self.reaction[reac][i] for i in range(self.ntime)]
            for t in list(time[:-2]):
                idx = len(time) - 1 - time[::-1].index(t)
                time.insert(idx+1, time[idx+1])
                conc.insert(idx, conc[idx])
            if i % nsub != 0:
                ax.append(plt.subplot2grid((nsub, msub), (i % nsub, i //
                                                          nsub), sharex=ax[i-i % nsub], colspan=1, rowspan=1))
            else:
                ax.append(plt.subplot2grid((nsub, msub), (i %
                                                          nsub, i//nsub), colspan=1, rowspan=1))
                [plt.setp(ax[j].get_xticklabels(), visible=False)
                 for j in range(i-nsub, i-1) if i > 0]
            reactants = ' + '.join(self.formula[r]
                                   for r in reac.split(':')[0].split(','))
            products = ' + '.join(self.formula[p]
                                  for p in reac.split(':')[1].split(','))
            ax[-1].set_yticks(list(range(0, max(conc)+2,
                                         int((max(conc)-min(conc))*float(nsub)*0.1)+1)))
            ax[-1].set_ylim(0, max(conc)+1)
            ax[-1].set_title('R'+repr(self.index[reac]) +
                             ': '+reactants+' -> '+products)
            ax[-1].plot(time, conc, color=color[i % nsub])
            ax[-1].grid()
            i += 1

        # . setup of shared x-axis
        #
        for j in range(len(ax)):
            ax[j].set_xticks(numpy.arange(0, numpy.around(
                max(time), 2)+0.5, numpy.around((max(time)-min(time))*float(msub)*0.05, 2)))
            ax[j].set_xlim(min(time), max(time)+0.1)
            plt.setp(ax[j].xaxis.get_majorticklabels(), rotation=90)
            if j % nsub == nsub-1 or j == len(ax)-1:
                ax[j].set_xlabel('Time [ns]')
        if nsub == len(Plot):
            for j in range(nsub-1):
                plt.setp(ax[j].get_xticklabels(), visible=False)
            ax[nsub-1].set_xlabel('Time [ns]')
        elif i % nsub != 0:
            [plt.setp(ax[j].get_xticklabels(), visible=False)
             for j in range(i-i % nsub, i-1)]
        else:
            [plt.setp(ax[j].get_xticklabels(), visible=False)
             for j in range(i-nsub, i-1)]

        # . display plot
        #
        if len(Plot) % nsub != 0:
            plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=-1.5)
        else:
            plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=0.0)
        plt.show()
        return True

    # @brief	saves species and reactions concentration data
    #		in a table
    # @return	True
    # @version	2014/07/31:
    #		Especially large and/or many work files may
    #		cause long conversion times for species. Once
    #		concentration profiles have been extracted from
    #		unformatted reaction, the data is available by
    #		reading the table file produced by this
    #		function. Rows are seperated by '\n' and columns
    #		are  seperated by ';'. The file endings are
    #		'.spec' and '.reac' with an additional '.tab'.
    #
    # @version	2014/10/02:
    #		time resolved rate constant data written to file
    #		with '.rate.tab' extension
    #
    # @version	2016/03/08:
    #		dicts replaced by lists
    #
    def writeData(self):
        # . write species concentrations
        writer = open(self.main+'.spec.tab', 'w')
        writer.write('t [steps];'+';'.join('S'+repr(self.index[spec])
                                           for spec in sorted(self.species) if spec)+'\n')
        writer.write(';'+';'.join(spec for spec in sorted(self.species))+'\n')

        for i in range(self.ntime):
            line = repr(self.time[i])+';'+';'.join('%d' % (self.species[spec][i])
                                                   for spec in sorted(self.species))+'\n'
            writer.write(line)
        writer.close()
        #

        # . write reaction "concentrations"
        writer = open(self.main+'.reac.tab', 'w')
        writer.write('t [steps];'+';'.join('R'+repr(self.index[reac])
                                           for reac in sorted(self.reaction))+'\n')
        labels = []
        for reac in sorted(self.reaction):
            reactants = ' + '.join('S'+repr(self.index[r])
                                   for r in reac.split(':')[0].split(',') if r)
            products = ' + '.join('S'+repr(self.index[p])
                                  for p in reac.split(':')[1].split(',') if p)
            labels.append(reactants+' -> '+products)
        writer.write(';'+';'.join(reac for reac in labels)+'\n')
        labels = []
        for reac in sorted(self.reaction):
            reactants = ' + '.join(self.formula[r]
                                   for r in reac.split(':')[0].split(',') if r)
            products = ' + '.join(self.formula[p]
                                  for p in reac.split(':')[1].split(',') if p)
            labels.append(reactants+' -> '+products)
        writer.write(';'+';'.join(reac for reac in labels)+'\n')
        writer.write(';'+';'.join(reac for reac in sorted(self.reaction))+'\n')

        for i in range(self.ntime):
            line = repr(self.time[i])+';'+';'.join(repr(self.reaction[reac][i])
                                                   for reac in sorted(self.reaction))+'\n'
            writer.write(line)
        writer.close()
        #

        # . write rate constants
        writer = open(self.main+'.rate.tab', 'w')
        tmp = ['R<ID>;S<ID>\'s;Formula\'s;SMILES;k;klo;kup;N']
        for reac in sorted(self.reaction):
            A = reac.split(':')[0].split(',')
            Ai = ['S'+repr(self.index[a]) for a in A if a]
            Af = [self.formula[a] for a in A if a]
            B = reac.split(':')[1].split(',')
            Bi = ['S'+repr(self.index[b]) for b in B if b]
            Bf = [self.formula[b] for b in B if b]
            forward = [' + '.join(Ai)+' -> '+' + '.join(Bi),
                       ' + '.join(Af)+' -> '+' + '.join(Bf)]
            reverse = [' + '.join(Bi)+' -> '+' + '.join(Ai),
                       ' + '.join(Bf)+' -> '+' + '.join(Af)]
            tmp.append('R%d;%s;%s;%s;%02e;%02e;%02e;%d' % (
                self.index[reac], forward[0], forward[1], reac, self.rate[reac][0], self.err[reac][0], self.err[reac][1], self.rate[reac][1]))
            if '' not in B:
                invreac = ','.join(B)+':'+','.join(A)
                tmp.append('R%d*;%s;%s;%s;%02e;%02e;%02e;%d' % (self.index[reac], reverse[0], reverse[1], invreac,
                                                                self.rate[invreac][0], self.err[invreac][0], self.err[invreac][1], self.rate[invreac][1]))
        for entry in tmp:
            writer.write(entry+'\n')
        writer.close()
        #

    # @brief	list of reactions and their ID used in the
    #		mechanism in latex
    # @return	True
    # @version	2014/07/07:
    #		The reactions represented by IDs in the
    #		mechanism are listed in a file in latex format.
    #		After removing the coment signs '%' on the top
    #		and bottom of the file it is ready for
    #		compilation.
    #
    # @version	2016/03/08:
    #		dicts replaced by lists
    #
    def writeTexReactions(self):
        writer = open(self.main+'.tex', 'w')
        writer.write('%\\documentclass{article}\n')
        writer.write('%\\usepackage[version=3]{mhchem}\n')
        writer.write('%\\usepackage{longtable}\n')
        writer.write('%\\begin{document}\n')
        writer.write('\\begin{longtable}{c|lc}\n')
        writer.write(' ID & Reaction & Total Flux \\\\ \\hline\n')
        for reac in sorted(self.reaction):
            if '' in reac.split(':')[0].split(',') or '' in reac.split(':')[1].split(','):
                continue
            flux = sum(self.reaction[reac][i] for i in range(self.ntime))
            reactants = ' + '.join(
                '\\ce{'+self.formula[r] + '}' for r in reac.split(':')[0].split(',') if r)
            products = ' + '.join(
                '\\ce{'+self.formula[p]+'}' for p in reac.split(':')[1].split(',') if p)
            writer.write(' '+repr(self.index[reac])+' & '+reactants +
                         ' $\\to$ '+products+' & '+repr(flux)+' \\\\\n')
        writer.write(' \\hline\n')
        writer.write('\\end{longtable}\n')
        writer.write('%\\end{document}')
        writer.close()
        return True

    # @brief	converts SMILES to 2D representations by using
    #		openbabel
    # @param	Resize		resize of pictures after
    #				convertion [%]
    # @return	True
    # @version	2014/07/06:
    #		The canonical SMILES extracted from ReaxFF
    #		simulations do not include double and triple
    #		bonds. Although bond orders are given returned
    #		by ReaxFF, these values often oscillate and may
    #		be unreliable therefore. Instead, the bond
    #		orders are determined on-the-fly before
    #		generating 2D representations of the species.
    #
    #		Note: Due to the rarity of quadruple bonds the
    #		code is restricted to adding double and triple
    #		bonds
    #
    # @version	2014/09/16:
    #		Added resize parameter for manual adjustment of
    #		picture size. Different 'convert' versions
    #		produced different results, thus required an
    #		additional parameter.
    #
    def depictMolecules(self, Resize):
        # . generation of folder
        #
        mol = openbabel.OBMol()
        folder = self.main+'.pic/'

        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        self.log.printComment(
            Text='Generating 2D depiction of molecules using "obabel". Scalable vector graphics and converted PNG files being stored in '+folder, onlyBody=False)

        # . depiction of molecules
        for spec in sorted(self.species):
            #   . setup of openbabel.OBMol() object
            self.conv.SetInAndOutFormats('smi', 'mdl')
            self.conv.ReadString(mol, spec)
            for atom in openbabel.OBMolAtomIter(mol):
                for j in range(atom.ImplicitHydrogenCount()):
                    h = mol.NewAtom()
                    h.SetAtomicNum(1)
                    mol.AddBond(atom.GetIdx(), h.GetIdx(), 1)
            #   . interaction with shell
            self.conv.SetInAndOutFormats('mdl', 'can')
            name = self.conv.WriteString(mol).strip()
            path = folder+'spec_'+repr(self.index[spec])

            run(['obabel', '-:'+name, '-xa', '-h', '-O', path+'.svg'])
            run(['obabel', '-:'+name, '-xm', '-xa',
                 '-xb', 'none', '-h', '-O', path+'.png'])

            # Requires ImageMagick
            #run(['convert', '-density', '1200', '-trim', path+'.svg', path+'.png'])
            # run(['convert', '-border', '30x30', '-bordercolor',
            #     '#FFFFFF', path+'.png', path+'.png'])
            #run(['convert', '-resize', repr(Resize)+'%', path+'.png', path+'.png'])
            self.log.printBody(Text=name+' ---> '+path+'.png', Indent=1)
            #
        return True

    # @brief	calculates branching ratios for all species
    # @version	2014/09/01:
    #		for comparing with experimental and other
    #		numerical results absolute values are
    #		meaningless. This function computes the relative
    #		flux of species to other species with respect to
    #		different reactions connecting them. Note: edge
    #		weights will still be absolute to allow for
    #
    # @version	2016/03/08:
    #		dicts replaced by lists
    #
    def calcBranching(self):
        self.ratio = {spec: {} for spec in self.species}
        tmp = {spec: {} for spec in self.species}
        loss = {spec: 0 for spec in self.species}
        for r in self.reaction:
            for reac in r.split(':')[0].split(','):
                if r not in tmp[reac]:
                    tmp[reac][r] = sum(self.reaction[r][i]
                                       for i in range(self.ntime))
                    loss[reac] += tmp[reac][r]

        for r in self.reaction:
            for reac in r.split(':')[0].split(','):
                if loss[reac] != 0:
                    self.ratio[reac][r] = tmp[reac][r]/float(loss[reac])
                else:
                    self.ratio[reac][r] = tmp[reac][r]
        return True

    # @brief	creates mechanism in GML format
    # @param	Threshold	plot only edges with flux above
    #				threshold
    # @param	Range		dict defining the ranges for
    #				element counts to be included in
    #				the mechanism
    # @param	Group		atomic number defining the
    #				element which is used to group
    #				species according to their count
    #				of the respective element
    # @return	True
    # @version	2014/07/06:
    #		if a reaction occurs only at the very first
    #		timestep it is considered to be a virtual
    #		reaction creating molecules from atoms and is
    #		discarded therefore.
    #
    # @version	2014/09/01:
    #		absolute flux values for edges have been
    #		replaced by their branching ratios. This allows
    #		for better comparison with other data sources
    #		and reveals the amount of reactions not
    #		displayed.
    #
    # @version	2016/03/08:
    #		dicts replaced by lists
    #
    def createMechanism(self, Threshold=1, Range={}, Group=0, Label=False):
        # . initial communication with user
        #
        self.log.printComment(Text='Mechanism is generated from list of reactions using a reaction flux threshold of '+repr(Threshold) +
                              ' and the following list of restrictions in the form <atomic number>: <min> to <max>\n\n'+'\n'.join(repr(a)+': '+repr(Range[a][0])+' to '+repr(Range[a][1]) for a in Range), onlyBody=False)

        # . reduction of species: filtering element content
        inrange = []
        gid = {}
        for spec in self.species:
            #   . counting of elements
            mol = openbabel.OBMol()
            self.conv.SetInAndOutFormats('smi', 'mdl')
            self.conv.ReadString(mol, spec)
            count = {a: 0 for a in Range}
            for atom in openbabel.OBMolAtomIter(mol):
                if atom.GetAtomicNum() != 1:
                    if atom.GetAtomicNum() in count:
                        count[atom.GetAtomicNum()] += 1
                    else:
                        count[atom.GetAtomicNum()] = 1
                    if 1 in count:
                        count[1] += atom.ImplicitHydrogenCount()
                    else:
                        count[1] = atom.ImplicitHydrogenCount()
                else:
                    if 1 in count:
                        count[1] += 1
                    else:
                        count[1] = 1
            #   . evalution of element count and grouping of species
            if all(count[atom] >= Range[atom][0] and count[atom] <= Range[atom][1] for atom in count if atom in Range):
                inrange.append(spec)
                if Group in count:
                    gid[spec] = count[Group]
                else:
                    gid[spec] = 0
        #

        # . reduction of reactions: filtering total flux
        #
        include = []
        exclude = []
        for reac in sorted(self.reaction):
            weight = sum([self.reaction[reac][i] for i in range(self.ntime)])
            if abs(weight) >= Threshold and (weight == 0 or self.reaction[reac][0]/weight != 1):
                if any(spec in reac for spec in inrange):
                    reaction = [reac.split(':')[0].split(
                        ','), reac.split(':')[1].split(',')]
                    if sorted(reaction[0]) != sorted(reaction[1]):
                        include.append([reac, weight])
                    else:
                        exclude.append([reac, weight])
                else:
                    exclude.append([reac, weight])
            elif abs(weight) < Threshold:
                exclude.append([reac, weight])

        # . generation of mechanism: nodes (species) and edges (reactions)
        graph = nx.DiGraph()
        folder = self.main+'.pic/'
        [graph.add_node(spec, label=spec, image=folder+'spec_' +
                        repr(self.index[spec])+'.png') for spec in inrange]
        self.calcBranching()
        for inc in include:
            reaction = [inc[0].split(':')[0].split(
                ','), inc[0].split(':')[1].split(',')]
            for reac in reaction[0]:
                for prod in reaction[1]:
                    if reac != prod and reac in inrange and prod in inrange:
                        if Label == 'flux':
                            edge = '['+repr(self.index[inc[0]]) + \
                                ']: '+repr(inc[1])
                        elif Label == 'branching':
                            edge = '['+repr(self.index[inc[0]]) + \
                                ']: %4.01f%%' % (100*self.ratio[reac][inc[0]])
                        else:
                            edge = '['+repr(self.index[inc[0]])+']'
                        try:
                            e = graph.edges[reac, prod]
                            if edge not in e['label']:
                                edge = e['label']+' / '+edge
                                e['label'] = edge
                            w = int(e['weight']) + inc[1]
                            e['weight'] = w
                        except:
                            graph.add_edge(reac, prod, label=edge,
                                           weight=inc[1], dir='forward')
        # . elemination of passive species
        for spec in list(inrange):
            if len(graph.succ[spec]) == 0 and len(graph.pred[spec]) == 0:
                graph.remove_node(spec)
                del inrange[inrange.index(spec)]
                del gid[spec]
        # . communication of species and reactions included in the mechanism
        if inrange:
            self.log.printBody(
                Text='\nReaction mechanism includes the following '+repr(len(inrange))+' species', Indent=1)
            self.log.printBody(Text=', '.join(
                spec for spec in inrange), Indent=2)
        if include:
            self.log.printBody(Text='\nReaction mechanism includes the following ' +
                               repr(len(include))+' reactions\ncount   reaction', Indent=1)
            [self.log.printBody(Text=repr(inc[1]).ljust(
                8)+inc[0], Indent=2) for inc in include]
        if exclude:
            self.log.printBody(Text='\nReaction mechanism excludes '+repr(len(exclude)) +
                               ' reactions and '+repr(len(self.species)-len(inrange))+' species\n', Indent=1)
        #

        # . save as GML graph
        nx.write_gml(graph, self.main+'.gml')

        # . save a png output too

        # extent is [center - scale, center + scale] (default: [-1, 1]).
        graph_pos = nx.spring_layout(graph)

        fig = plt.figure()
        graphax = fig.add_axes([0, 0, 1, 1])
        graphax.axis('off')
        nx.draw_networkx_edges(graph, graph_pos, ax=graphax)
        nx.draw_networkx_labels(graph, graph_pos, ax=graphax)

        graph_edge_labels = {(u, v): d['label']
                             for u, v, d in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(
            graph, graph_pos, edge_labels=graph_edge_labels, ax=graphax)

        x_bounds = numpy.array(graphax.get_xbound())
        y_bounds = numpy.array(graphax.get_ybound())

        thumbnail_size = 0.1  # for 60 pixels

        for key, value in graph_pos.items():
            img = plt.imread(graph.nodes[key]['image'], format='PNG')
            img_ori = [(value[0]-x_bounds[0])/(numpy.abs(x_bounds).sum()),
                       (value[1]-y_bounds[0])/(numpy.abs(y_bounds).sum())-0.15]

            img_width = numpy.size(img, 0)/100*thumbnail_size
            img_height = numpy.size(img, 1)/100*thumbnail_size

            # the extend is[left, bottom, width, height]
            axicon = fig.add_axes(img_ori + [img_width, img_height])
            axicon.axis('off')

            axicon.imshow(img, aspect='equal')
            #fig.figimage(img, xo=value[0], yo=value[1], origin='lower')

        plt.savefig(self.main+'.png', dpi=600, format='png',
                    bbox_inches='tight', pad=3.0)

        return True


#######################################
if __name__ == "__main__":
    log = Log.Log(Width=70)

    # HEADER
    #
    text = ['<source file> [-main <main>] [-gen <label-style>] [-dp <size>] [-t <threshold>] [-r <atomic number>:<min>-<max>] [-g <atomic number>] [-p <names>] [-step <timestep>] [-start <start>] [-end <end>] [-X <uncertainty limit>] [-extra <list of comma-separated SMILES>]',
            'options:',
            '-main: name for output like <main>.gml, the graph file or <main>.pic/, the 2D depiction folder (default="Default")',
            '-gen: activates mechanism generation and stores reactions in LaTeX format. <label-style> "id" prints only reaction IDs on connections, "flux" adds the total net flux and "branching" adds the ratio of ractions computed from net fluxes',
            '-dp: create 2D depiction for molecules and store them in the <main>.pic/ folder (lack of 2D pictures may cause failure). The <size> parameter is determining the size of the resulting pictures and needs to be adjusted according to your "ImageMagick" / "convert" version.',
            '-t: threshold for reaction flux which limits the reactions considered for mechanism generation to those occuring <threshold> times at least, default=1',
            '-r: restrictions on element counts for species in the form <atomic number>:<min count>-<max count> (e.g. 6:1-2 for species with carbon atom counts between 1 and 2), default is no restrictions',
            '-g: define atomic number of element which should be used to group the species, default=0 i.e. no grouping',
            '-p: define SMILES or indices species and/or reactions for plotting them as follows <SMILE> or S<idx>. Reactions can be requested by supplying their SMILES: <A>,<B>:<C>,<D> with <A>,<B>,... being SMILES of the reacting molecules; or their indices: R<idx>. Default=[], i.e. no plotting',
            '-step: give the timestep of the simulation in [fs] (default=0.1)',
            '-start: skip timesteps smaller than <start> for rate computation (default=0)',
            '-end: end-flag for rate computation (default=-1)',
            '-X: percentage limit for rate constant uncertainty estimation',
            '-extra: list of species for which unimolecluar and bimolecular reaction probabilities will be computed. Separte SMILES with comma (,)']
    log.printHead(
        Title='ChemTraYzer - ReaxFF Data Analyzing',
        Version='2015-05-05',
        Author='Malte Doentgen, LTT RWTH Aachen University',
        Email='chemtrayzer@ltt.rwth-aachen.de',
        Text='\n\n'.join(text))

    # INPUT
    #
    if len(sys.argv) > 1:
        argv = sys.argv
    else:
        log.printComment(
            Text=' Use keyboard to type in work filenames. The <return> button will not cause leaving the input section. Use <strg>+<c> or write "done" to proceed.', onlyBody=False)
        argv = ['input']
        done = False
        while not done:
            try:
                tmp = input('')
            except:
                done = True
            if 'done' in tmp.lower():
                done = True
            else:
                if '!' in tmp:
                    argv += tmp[:tmp.index('!')].split()
                else:
                    argv += tmp.split()

    log.printBody(Text='Reading files ...', Indent=0)

    # INTERPRET INPUT
    #
    work = []
    main = 'Default'
    gen = False
    depict = False
    threshold = 1
    restrict = {}
    group = 0
    plot = []
    timestep = 0.1
    resize = 0.2
    label = 'id'
    start = 0
    end = -1
    X = 0.95
    noReacSpec = []
    i = 1
    while i < len(argv):
        if argv[i].lower() == '-main':
            try:
                main = argv[i+1]
                i += 1
            except:
                log.printIssue(
                    Text='-main: option expected additional argument, got nothing and will use default="Default" for naming output', Fatal=False)
        elif argv[i].lower() == '-gen':
            gen = True
            try:
                label = argv[i+1]
                i += 1
            except:
                log.printIssue(
                    Text='-gen: option expected additional argument, got nothing and will use default="id" for labeling graph edges', Fatal=False)
        elif argv[i].lower() == '-dp':
            depict = True
            try:
                resize = int(float(argv[i+1])*100)
                i += 1
            except:
                log.printIssue(
                    Text='-dp: option expected additional argument, got string/nothing and will use default=20%', Fatal=False)
        elif argv[i].lower() == '-t':
            try:
                threshold = int(argv[i+1])
                i += 1
            except:
                log.printIssue(
                    Text='-t: option expected integer, got string/nothing and will use default=1', Fatal=False)
        elif argv[i].lower() == '-r':
            try:
                tmp = argv[i+1]
                i += 1
            except:
                log.printIssue(
                    Text='-r: option expected additional argument, got nothing and will use default i.e. no restrictions', Fatal=False)
            if ':' in tmp and '-' in tmp:
                restrict[int(tmp.split(':')[0])] = [
                    int(tmp.split(':')[1].split('-')[0]), int(tmp.split(':')[1].split('-')[1])]
            else:
                log.printIssue(
                    Text='-r: faulty format of argument, restriction will be ignored. WARNING: if no argument was supplyied the subsequent options are corrupted', Fatal=False)
        elif argv[i].lower() == '-g':
            try:
                group = int(argv[i+1])
                i += 1
            except:
                log.printIssue(
                    Text='-g: option expected integer, got string/nothing and will use default=0, i.e. no grouping', Fatal=False)
        elif argv[i].lower() == '-p':
            try:
                plot.append(argv[i+1])
                i += 1
            except:
                log.printIssue(
                    Text='-plot: option expected string, got nothing and will use default=[], i.e. no plotting', Fatal=False)
        elif argv[i].lower() == '-step':
            try:
                timestep = float(argv[i+1])
                i += 1
            except:
                log.printIssue(
                    Text='-step: option expected integer, got string/nothing and will use default=0.1', Fatal=False)
        elif argv[i].lower() == '-start':
            try:
                start = int(argv[i+1])
                i += 1
            except:
                log.printIssue(
                    Text='-start: option expected integer, got string/nothing and will use default=0', Fatal=False)
        elif argv[i].lower() == '-end':
            try:
                end = int(argv[i+1])
                i += 1
            except:
                log.printIssue(
                    Text='-end: option expected integer, got string/nothing and will use default=-1, i.e. not end-flag', Fatal=False)
        elif argv[i].lower() == '-x':
            try:
                X = float(argv[i+1])
                i += 1
            except:
                log.printIssue(
                    Text='-X: option expected float, got string/nothing and will use defalut=0.9', Fatal=False)
        elif argv[i].lower() == '-extra':
            try:
                noReacSpec = argv[i+1].split(',')
                i += 1
            except:
                log.printIssue(
                    Text='-extra: option expected string, got nothing and will use default=[]', Fatal=False)
        else:
            try:
                line = open(argv[i], 'r').readline()
                work.append(argv[i])
            except:
                log.printIssue(Text='attempt to read from ' +
                               argv[i]+' failed. Please check whether file is broken or does not exist. File will be ignored ...', Fatal=False)
                line = ''
        i += 1
    if len(work) > len(set(work)):
        work = set(work)
        log.printIssue(
            Text='redundant work files found, condensing to unique set of files', Fatal=False)

    def readWork(Work):
        reader = open(Work, 'r')
        line = reader.readline().strip()
        # reac-file generated via post-processing
        line = line.replace('WORK', '')
        # reac-file generated via simulation.py
        line = line.replace('ON THE FLY ...', '')
        words = line.split()
        try:
            temperature = float(words[0])
        except:
            temperature = -1
        try:
            volume = float(words[1])
        except:
            volume = -1
        try:
            timestep = float(words[2])
        except:
            timestep = -1
        worktime = -1
        reaction = {}
        done = False
        while not done:
            words = reader.readline().split(':')
            try:
                int(words[0])
            except:
                if len(words) == 1 and words[0] == '':
                    done = True
                continue
            if len(words) > 1:
                try:
                    int(words[0])
                    if int(words[0]) not in reaction:
                        reaction[int(words[0])] = [[words[1].split('\n')[0].split(
                            ','), words[2].split('\n')[0].split(',')]]
                    else:
                        reaction[int(words[0])].append(
                                [words[1].split('\n')[0].split(','), words[2].split('\n')[0].split(',')])
                except:
                    pass
            else:
                done = True
        reader.close()
        if reaction and (worktime == -1 or worktime < max(reaction)):
            worktime = max(reaction)
        return worktime, temperature, volume, timestep, reaction

    # FILE INITIALIZATION
    #
    reaction = {}
    if work:
        log.printComment(Text='The following list of work files will be merged into a single list of reactions. All virtual reactions required for initial molecule creation are stored at time=0 !', onlyBody=False)
        log.printBody(Text=', '.join(f for f in work), Indent=1)
        log.printBody(Text='', Indent=1)
        tshift = 0.0
        for f in work:
            wt, T, V, dt, tmp = readWork(f)
            for time in tmp:
                if time+tshift in reaction:
                    reaction[time+tshift] += tmp[time]
                else:
                    reaction[time+tshift] = tmp[time]
            tshift = max(reaction)
            log.printBody(Text='timesteps '+repr(min(tmp)) +
                          ' to '+repr(wt)+' read from "'+f+'"', Indent=1)
        log.printBody(Text='', Indent=1)
    else:
        log.printIssue(Text='No input files found ...', Fatal=True)

    # REACTION ANALYSIS
    #
    anly = Analyzing(Reaction=reaction, Main=main, Start=start, End=end, Vol=(
        V > 0)*V + (V < 0)*1.0, Timestep=(dt > 0)*dt + (dt < 0)*0.1, X=X, NoReacSpec=noReacSpec)
    anly.writeData()
    if depict:
        anly.depictMolecules(Resize=resize)
        log.printBody(
            Text='\n>>> DONE: molecule depections generated and stored in '+main+'.pic/\n', Indent=1)
    if gen:
        anly.createMechanism(Threshold=threshold,
                             Range=restrict, Group=group, Label=label)
        anly.writeTexReactions()
        log.printBody(Text='>>> DONE: mechanism written to '+main +
                      '.gml and reactions written to '+main+'.tex\n', Indent=1)
    if plot:
        anly.drawProfile(Plot=plot, Timestep=timestep)
