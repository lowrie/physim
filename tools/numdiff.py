#!/usr/bin/env python
# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Reports the differences between respective floating point numbers that
are stored in the two specified files.

If any differences are found, then the exit status is 1 (and DIFFER is
printed if verbosity > 0); otherwise, the exit status is 0 (and SAME
is printed if verbosity > 0).

Note that if both -t and -r are active, then both the
relative and absolute tolerances must be satisfied.

The two files are read line by line and numeric
differences are taken between their respective columns of data.
Columns are delineated by whitespace.  A column may be non-numeric, in
which case a string comparison is done.

For example, a line of data such as

       x = 5.4

is treated as three columns (x, =, 5.4) with the third column numeric.
If the same line in the other file is

       x = 5.3

and tolerance (-t) < 0.1, then numdiff will note that column 3 has an
absolute difference of 0.1 and a relative difference of 0.1 / 10.7.
If instead the second file contains

       y = 5.4

a difference is also reported because the strings x and y differ.

Note that the line

       x=5.3323412

is treated as a single column of string data.  Someday, this case
might be handled differently, although then one should also consider
that

       x1=5.3323412

has two numeric entries (1 and 5.3323412).  For now, be sure to use
whitespace to delineate numeric columns.
'''

import os, sys, argparse, copy, glob, re

class Tolerances:
    '''
    Floating-point tolerances object.

    Attributes:

      absolute = tolerance for |f1 - f2|
      relative = tolerance for |f1 - f2| / (|f1| + |f2| + epsilon)
      epsilon  = paramter in relative error
    '''
    def __init__(self, absolute=1.0e-8, relative=1.0e-8,
                 epsilon=1.0e-15):
        self.absolute = absolute
        self.relative = relative
        self.epsilon  = epsilon
    def equals(self, f1, f2):
        '''
        Checks whether f1 == f2 to a tolerance.  Returns
        the 3-tuple (equal, abs_difference, relative_diff)
        '''
        a = abs(f1 - f2)
        r = a / (abs(f1) + abs(f2) + self.epsilon)
        equal = 1
        if self.absolute is not None:
            equal = equal and (a < self.absolute)
        if self.relative is not None:
            equal = equal and (r < self.relative)
        return equal, a, r


class Error(Exception):
    '''
    Base class for exceptions in this module.
    '''
    pass

class AdvanceFileToStringError(Error):
    '''
    Errors thrown in function advanceFileToString.
    '''
    def __init__(self, string, fileName):
        self.string = string
        self.fileName = fileName
    def __str__(self):
        return f'\nCannot find string {self.string} in file {self.fileName}'

def advanceFileToString(file, startString, lineNum):
    '''
    Advances file to the first line with startString.  Returns the
    line number.
    '''
    if len(startString) == 0:
        return lineNum
    while 1:
        line = file.readline()
        if len(line) == 0:
            raise AdvanceFileToStringError(startString, file.name)
        lineNum = lineNum + 1
        if line.find(startString) >= 0:
            return lineNum

class Stats:
    '''
    Stats for differences between two values.

    Attributes:

      abs = absolute difference for floats; otherwise, Y for
            if the strings differ, N if not.
      rel = relative difference for float data
      line1 = line string in file #1
      line2 = line string in file #2
      lineNum1 = line number in file #1
      lineNum2 = line number in file #2
      columnNum = data column number
      tolerances = Tolerances object for float data
      equal = if true, values are equal
    '''
    def __init__(self, line1='', line2='', lineNum1=0, lineNum2=0,
                 tolerances=Tolerances()):
        self.abs = 0
        self.rel = 0
        self.line1 = line1
        self.line2 = line2
        self.lineNum1 = lineNum1
        self.lineNum2 = lineNum2
        self.tolerances = tolerances
        self.equal = 1
    def findDiff(self, s1, s2, columnNum):
        '''
        Finds the difference between the strings s1 and s2.
        Returns true if a difference was found, false otherwise.
        '''
        self.columnNum = columnNum
        
        try:
            # See if the strings represent floats
            v1 = float(s1) 
            v2 = float(s2)
            (self.equal, self.abs, self.rel) = \
                         self.tolerances.equals(v1, v2)
        except ValueError:
            # not float data; see if strings are the same
            if s1 == s2:
                self.equal = 1
                self.abs = 'N'
            else:
                self.equal = 0
                self.abs = 'Y'
                
        return not self.equal
    def diffIsString(self):
        '''
        Returns true if the difference is for string data.
        '''
        return type(self.abs) == type('')

class TotalStats:
    '''
    Accumlated Stats.

    Attributes:

      maxAbs = maximum abs.
      maxRel = maximum rel.
      totalFloats = number of floats compared.
      diffFloats = number of floats that differed.
      totalStrings = number of strings compared.
      diffStrings = number of strings that differed.
    '''
    def __init__(self):
        self.maxAbs = Stats()
        self.maxRel = Stats()
        self.totalFloats = 0
        self.totalStrings = 0
        self.diffFloats = 0
        self.diffStrings = 0
    def accumlate(self, stats):
        '''
        Accumulates stats into total stats.
        '''
        if stats.diffIsString():
            self.totalStrings = self.totalStrings + 1
            if stats.abs == 'Y':
                self.diffStrings = self.diffStrings + 1
        else:
            self.totalFloats = self.totalFloats + 1
            if stats.abs > self.maxAbs.abs:
                self.maxAbs = copy.deepcopy(stats)
            if stats.rel > self.maxRel.rel:
                self.maxRel = copy.deepcopy(stats)
            if not stats.equal:
                self.diffFloats = self.diffFloats + 1

class Numdiff:
    '''
    Finds numerical differences between two files.

    Attributes control behavior for calls to diff():

      verbosity = 0: No output. Only the exit status of diff() is set, 
                     0 if all files the same, 1 if any differ.
                  1: If the files differ, a brief summary is output.
                  2: A brief summary is output.
                  3: All lines with numeric or string differences are output.
      skip = list of re.compile() objects, for which if both lines
             satisfy, the comparison is skipped.
      tolerances = a Tolerances object for numerical comparisons.
      printContextLines = if true and verbosity level is 2, prints any
                          line in file1 that contains no numeric data.
      startString = in each file, advance to the line containing this
                    string before starting comparison.
    '''
    def __init__(self, startString='', printContextLines=False,
                 tolerances=Tolerances(), verbosity=1, skip=[]):
        self.startString = startString
        self.printContextLines = printContextLines 
        self.tolerances = tolerances
        self.verbosity = verbosity
        self.skip = skip
    def _gotoStartString(self, file, fileName):
        '''
        Advances file to self.startString
        '''
        lineNum = 0
        if len(self.startString) > 0:
            lineNum = advanceFileToString(file, self.startString, 0)
        return lineNum
    def _skipLine(self, line):
        '''
        Returns true if line satisfies any of the skip regex.
        '''
        for regex in self.skip:
            if regex.search(line):
                return 1

        return 0
    def _splitLines(self, line1, line2):
        '''
        Splits the strings line1 and line2 into columns and
        pads the short lines with blanks so that each line has the
        same number of columns.
        '''
        s1 = line1.split()
        s2 = line2.split()
        numCol = len(s1)

        if numCol > len(s2):
            for i in range(numCol - len(s2)):
                s2.append(' ')
        elif numCol < len(s2):
            numCol = len(s2)
            for i in range(numCol - len(s1)):
                s1.append(' ')
        return (s1, s2, numCol)
    def _compareLines(self, line1, line2, lineNum1, lineNum2, tstats):
        '''
        Compares lines line1 and line2, and accumulates the
        differences into stats.
        '''

        (s1, s2, numCol) = self._splitLines(line1, line2)

        # Find the difference between each respective column
        # and accumulate the stats
        
        s = Stats(line1, line2, lineNum1, lineNum2, self.tolerances)
            
        printLine = 0
        diff = [] # differences

        for i in range(numCol):
            printLine = printLine | s.findDiff(s1[i], s2[i], i+1)
            diff.append(s.abs)
            tstats.accumlate(s)

        # Print the differences
            
        if self.verbosity > 2:
            if printLine:
                if lineNum1 == lineNum2:
                    print(f'Line {lineNum1}:')
                else:
                    print(f'Line {lineNum1} of file 1, line {lineNum2} of file 2:')
                print(line1, end='')
                print(line2, end='')
                print('diffs:')
                for i in diff:
                    if type(i) == type(''):
                        print(f' {i}', end='')
                    else:
                        print(f' {i:.4e}', end='')
                print('')
            elif self.printContextLines and len(diff) == 0:
                print(f'** {line1}', end='')
                
        return tstats
    def _diffStats(self, fileName1, fileName2):
        '''
        Takes the difference between files fileName1 and fileName2,
        and returns a TotalStats object.
        '''
        tstats = TotalStats()
        file1 = open(fileName1,'r')
        file2 = open(fileName2,'r')

        # Advance to starting string

        lineNum1 = self._gotoStartString(file1, fileName1)
        lineNum2 = self._gotoStartString(file2, fileName2)

        if self.verbosity > 1:
            print(f'\nComparing {fileName1} and {fileName2}:')
            if lineNum1 > 0:
                print(f'Advanced {fileName1} to line {lineNum1}')
            if lineNum2 > 0:
                print(f'Advanced {fileName2} to line {lineNum2}')
                
        # Loop until EOF on either file
        
        while 1:
            line1 = file1.readline()
            lineNum1 = lineNum1 + 1
            while self._skipLine(line1):
                line1 = file1.readline()
                lineNum1 = lineNum1 + 1

            line2 = file2.readline()
            lineNum2 = lineNum2 + 1
            while self._skipLine(line2):
                line2 = file2.readline()
                lineNum2 = lineNum2 + 1

            warn = 'Warning: File %s has more non-skipped lines than file %s'
            if len(line1) == 0:
                if len(line2) > 0 and self.verbosity > 1:
                    print(warn % (fileName2, fileName1))
                break;
            if len(line2) == 0 and self.verbosity > 1:
                print(warn % (fileName1, fileName2))
                break;
            tstats = self._compareLines(line1, line2,
                                        lineNum1, lineNum2, tstats)

        file1.close()
        file2.close()
        return tstats
    def _statsSummary(self, fileName1, fileName2, tstats):
        '''
        Prints a summary of the stats (depending on value of
        self.verbosity).  If any differences are found, then 1
        is returned (and DIFFER is printed if verbosity > 0);
        otherwise, 0 is returned (and SAME is printed
        if verbosity > 0).
        '''

        differ = 0
        if tstats.diffFloats + tstats.diffStrings > 0:
            differ = 1

        if self.verbosity > 0:
            printsummary = True
            if self.verbosity == 1:
                if differ:
                    print(f'\nFiles {fileName1} and {fileName2} differ:')
                else:
                    printsummary = False
            if printsummary:
                print(f'Total strings compared = {tstats.totalStrings}')
                print(f'Strings that differed  = {tstats.diffStrings}')
                print(f'Total numbers compared = {tstats.totalFloats}')
                print(f'Numbers that differed  = {tstats.diffFloats}')
                if tstats.diffFloats > 0:
                    print(f'Max absolute = {tstats.maxAbs.abs:.4e} on lines'
                        f' ({tstats.maxAbs.lineNum1},{tstats.maxAbs.lineNum2}),'
                        f' column {tstats.maxAbs.columnNum} (relative = {tstats.maxAbs.rel:.4e})')
                    print(f'Max relative = {tstats.maxRel.rel:.4e} on lines'
                        f' ({tstats.maxRel.lineNum1},{tstats.maxRel.lineNum2}),'
                        f' column {tstats.maxRel.columnNum} (absolute = {tstats.maxRel.abs:.4e})')
                if self.verbosity > 1:
                    if differ:
                        print('DIFFER')
                    else:
                        print('SAME')

        return differ

    def diff(self, fileName1, fileName2):
        '''
        Takes the difference between files fileName1 and fileName2 and
        return a 2-tuple (status, stats).

        status is 0 if there is a difference, 1 if not.
        stats is a TotalStats object.
        '''
        tstats = self._diffStats(fileName1, fileName2)
        filesDiffer = self._statsSummary(fileName1, fileName2, tstats)
        return filesDiffer, tstats

class Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    pass

############################################################################
########################## Beginning of main program #######################

if __name__ == '__main__':

    tol = Tolerances() # to extract defaults
    # Parse command line options
    parser = argparse.ArgumentParser('numpy.py', description=__doc__, formatter_class=Formatter)
    parser.add_argument('path1', help='File name or directory path.')
    parser.add_argument('path2', help='File name or directory path to compare with path1.'
                        ' If path1 and path2 are both files, then the difference is performed between the two files.'
                        ' If both directories, then all files are found under the path1, and'
                        ' the respective file is found in path2, assuming the same directory'
                        ' structure. Each pair of files is then compared.')
    parser.add_argument('--contextlines', default=False, action='store_true',
           help='If verbosity level is 2, print lines in the first path that'
           ' contain no numeric data (default: %(default)s).')
    parser.add_argument('--find', default='', metavar='STRING',
                        help='In each file, advance to the line containing'
                        ' STRING before starting comparison.')
    parser.add_argument('--include', default='*',
           help='When path1 and path2 are directories, include only satisfy this regex.'
           ' (default: %(default)s).')
    parser.add_argument('--relative', nargs='*', type=float,
                        default = [tol.relative, tol.epsilon],
                        metavar='TOL [EPSILON]',
                        help=' Set the relative difference tolerance.'
                        ' The values f1 and f2 are considered equal if'
                        ' |f1 - f2| / (|f1| + |f2| + EPSILON) <= TOL.'
                        ' May be combined with --tolerance.'
                        ' Specify --norelative to turn off.'
                        ' (default: %(default)s).')
    parser.add_argument('--norelative', default=False, action='store_true',
           help='Turn off --relative checking.'
           ' (default: %(default)s).')
    parser.add_argument('--skip', metavar='REGEX', nargs='*', action='append',
           help='Skip lines that satisfy the regular expression REGEX.'
           ' Skipped lines do not determine whether two files are'
           ' different. Two files may contain a different number of'
           ' skipped lines, or content in those lines, and still be'
           ' considered the same.')
    parser.add_argument('--tolerance', default=tol.absolute, type=float,
           help=' Set the absolute difference tolerance.  The values'
           ' f1 and f2 are considered equal if |f1 - f2| <= TOLERANCE.'
           ' May be combined with --relative.'
           ' Specify --notolerance to turn off. (default: %(default)s).')
    parser.add_argument('--notolerance', default=False, action='store_true',
           help='Turn off --tolerance checking.'
           ' (default: %(default)s).')
    parser.add_argument('--verbosity', default=1, type=int,
                        choices=[0,1,2,3],
                        help='Sets the verbosity level to standard out.'
                        ' 0: No output. Only the exit status is set, 0 if'
                        '    all files the same, 1 if any differ.'
                        ' 1: If the files differ, a brief summary is output.'
                        ' 2: A brief summary is always output.'
                        ' 3: All lines with numeric or string differences are output.'
                        ' (default: %(default)s).')
    args = parser.parse_args()

    numdiff = Numdiff(startString=args.find, printContextLines=args.contextlines,
                      verbosity=args.verbosity)
    if args.skip is not None:
        for s in args.skip:
            numdiff.skip.append(re.compile(s))
    tol.absolute = args.tolerance
    if args.notolerance:
        tol.absolute = None
    if len(args.relative) > 2:
        parser.error('Too many arguments for option --relative')
    if args.norelative:
        tol.relative = None
    else:
        tol.relative = args.relative[0]
        if len(args.relative) == 2:
            tol.epsilon = args.relative[1]
    numdiff.tolerances = tol

    # Form the pairs of files to compare

    filePairs = []
    numloners = 0 # don't exist in one path
    if os.path.isdir(args.path1):
        if not os.path.isdir(args.path2):
            parser.error(f'{args.path1} is a directory, but {args.path2} is not.')
        # Pair files in path1 with the corresponding file in path2
        d = os.path.join(args.path1, '**', args.include)
        files = glob.glob(d, recursive=True)
        files.sort()
        for f1 in files:
            if os.path.isfile(f1):
                pc = os.path.commonpath([args.path1, f1])
                stub = f1[len(pc)+1:]
                f2 = os.path.join(args.path2, stub)
                if os.path.exists(f2):
                    filePairs.append([f1, f2])
                else:
                    print(f'WARNING: Found file {f1}, but corresponding file {f2} does not exist.')
                    numloners += 1
        # See if any files in path2 do not exist in path1
        d = os.path.join(args.path2, '**', args.include)
        files = glob.glob(d, recursive=True)
        for f2 in files:
            if os.path.isfile(f2):
                pc = os.path.commonpath([args.path2, f2])
                stub = f2[len(pc)+1:]
                f1 = os.path.join(args.path1, stub)
                if not os.path.exists(f1):
                    print(f'WARNING: Found file {f2}, but corresponding file {f1} does not exist.')
                    numloners += 1
    elif os.path.isfile(args.path1):
        if not os.path.exists(args.path2):
            parser.error(f'{args.path2} not found.')
        elif not os.path.isfile(args.path2):
            parser.error(f'{args.path1} is a file, but {args.path2} is not.')
        filePairs.append([args.path1, args.path2])
    else:
        parser.error(f'{args.path1} not found.')

    # Do the difference
    numdiffer = 0 #
    for f in filePairs:
        (filesDiffer, stats) = numdiff.diff(f[0], f[1])
        if filesDiffer:
            numdiffer += 1
    if args.verbosity > 0 and len(filePairs) > 1:
        print('\nSummary over all file comparisons:')
        print(f'{numdiffer} differ.')
        print(f'{len(filePairs) - numdiffer} are the same.')
        if numloners > 0:
            print(f'{numloners} files do not exist in both paths.')

    sys.exit(numdiffer > 0)
