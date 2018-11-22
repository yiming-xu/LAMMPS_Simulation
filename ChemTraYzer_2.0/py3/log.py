##################################################################
# The MIT License (MIT)						 #
#								 #
# Copyright (c) 2014 RWTH Aachen University, Malte Doentgen      #
#								 #
# Permission is hereby granted, free of charge, to any person    #
# obtaining a copy of this software and associated documentation #
# files (the "Software"), to deal in the Software without        #
# restriction, including without limitation the rights to use,   #
# copy, modify, merge, publish, distribute, sublicense, and/or   #
# sell copies of the Software, and to permit persons to whom the #
# Software is furnished to do so, subject to the following	 #
# conditions:							 #
#								 #
# The above copyright notice and this permission notice shall be #
# included in all copies or substantial portions of the Software.#
#								 #
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
## @file	log.py
## @author	Malte Doentgen
## @date	2014/07/24
## @brief	This file contains the 'Log' class which is used
#		for communication with the user during run

import sys
import time
import textwrap

## @class	Log
## @brief	object for properly returning information, warnings and
#		errors
## @version	2014/07/04:
#		Communication with users is printed to the shell with
#		the methods given in this class. These methods adjust
#		the text width to a certain value to yield information
#		in a readable way.
class Log:
	## @brief	Constructor
	## @parameter	Width		maximum number of characters in
	#				a line before line ends
	## @return	None
	## @version	2014/07/04:
	#		The textwrappers used by the print methods are
	#		initialized.
	#
	def __init__(self, Width=70):
		self.width = Width
		self.body = textwrap.TextWrapper(initial_indent=' ', subsequent_indent=' ', width=Width)
		self.comment = textwrap.TextWrapper(initial_indent='! ', subsequent_indent='! ', width=Width)
		self.error = textwrap.TextWrapper(initial_indent='   ', subsequent_indent='   ', width=Width)
		
		self.commentLine = '!'+('-').join('' for i in range(Width+1))+'!'
		self.errorLine = ('*').join('' for i in range(Width+3))
	
	## @brief	prints standard output to the shell
	## @parameter	Text		string to be printed
	## @parameter	Indent		integer defining the width of
	#				indenting space
	## @return	None
	## @version	2014/07/04:
	#		The 'Text' is printed with 'Indent' spaces on
	#		the left size of each line. The maximum line
	#		length plus indenting spaces is 'self.width'.
	#
	def printBody(self, Text='', Indent=0):
		self.body.initial_indent = ' '+(' ').join('' for i in range(Indent+1))
		self.body.subsequent_indent = self.body.initial_indent
		fragments = Text.split('\n')
		for frag in fragments:
			if not frag: wrappedText= ['']
			else: wrappedText = self.body.wrap(frag)
			print(('\n').join(wrappedText))
	
	## @brief	prints text to the shell surrounded by
	#		exclamation marks
	## @parameter	Text		string to be printed
	## @parameter	onlyBody	bool for printing top/bottom
	#				comment lines
	## @return	None
	## @version	2014/07/04:
	#		The 'Text' is printed either with top/bottom
	#		comment lines (exclamation marks filled with
	#		asterisks) or without. The latter is forced by
	#		'onlyBody=True'. This can be used to add to a
	#		comment section which was opened at some place
	#		else earlier.
	#
	#		Note: when using 'onlyBody=True' the top/bottom
	#		comment lines must be added manually
	#		(if required).
	#
	def printComment(self, Text='', onlyBody=False):
		if not onlyBody: print(self.commentLine)
		fragments = Text.split('\n')
		for frag in fragments:
			if not frag: wrappedText = ['!']
			else: wrappedText = self.comment.wrap(frag)
			for i in range(len(wrappedText)):
				wrappedText[i] += (' ').join('' for i in range(self.width+1-len(wrappedText[i])))+' !'
			print(('\n').join(wrappedText))
		if not onlyBody: print(self.commentLine)
	
	## @brief	displays software information in the shell
	## @parameter	Title		name of the software
	## @parameter	Version		identification of version
	## @parameter	Author		name of the author
	## @parameter	Email		email of the author
	## @parameter	Date		date of creation / last change
	#				/ ...
	## @parameter	Text		comments and/or instructions
	## @return	None
	## @version	2014/07/04:
	#		The head printing methods starts with giving the
	#		current date and UTC time, followed by some
	#		official information to the user (version,
	#		author, ...). The text parameter should be used
	#		for giving instructions to the user and
	#		informing about any changes or known problems
	#		/ limits of the software
	#
	def printHead(self, Title='', Version='', Author='', Email='', Date='', Text=''):
		fill = self.width + 1 - (len(Title) + 4)
		print()
		print(time.asctime(time.gmtime(time.time()))+' / UTC')
		print()
		print('!'+('-').join('' for i in range(int(fill/2 +fill%2)))+'  '+Title+'  '+('-').join('' for i in range(1+int(fill/2)))+'!')
		self.printComment(Text='', onlyBody=True)
		if Version: self.printComment(Text='   Version '+Version, onlyBody=True)
		if Author: self.printComment(Text='   Author  '+Author, onlyBody=True)
		if Email: self.printComment(Text='   Email   '+Email, onlyBody=True)
		if Date: self.printComment(Text='   Date    '+Date, onlyBody=True)
		if Text: self.printComment(Text='\n'+Text, onlyBody=True)
		else: self.printComment(Text='', onlyBody=True)
		print(self.commentLine)
		print()
	
	## @brief	prints a warning or error and can abort the
	#		program
	## @parameter	Text		string to be printed
	## @parameter	Fatal		bool for forcing exit
	## @return	None
	## @version	2014/07/04:
	#		The most important information are failure
	#		reports. Users exactly want to know what is
	#		failing. The more detailed a failure report the
	#		easier fixing the problem. Setting 'Fatal=True'
	#		will cause the end of the program. Use this
	#		option only if proceeding will cause an
	#		uncontrolled malfunction.
	#
	def printIssue(self, Text='', Fatal=False):
		print(self.errorLine)
		if Fatal: print('ERROR:')
		else: print('WARNING:')
		fragments = Text.split('\n')
		for frag in fragments:
			if not frag: wrappedText = ['']
			else: wrappedText = self.error.wrap(frag)
			print(('\n').join(wrappedText))
		if Fatal: print('\nEXIT')
		print(self.errorLine)
		if Fatal: sys.exit()

