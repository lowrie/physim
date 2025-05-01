#!/usr/bin/env python
# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for running a series of benchmark problems.
'''
import argparse
import copy
import numpy as np
import os
import sys
from datetime import datetime

import base.stepper
import euler.driver as driver

def default_hydro_case() -> driver.Case:
    '''
    Returns a default hydro case.
    '''
    return driver.Case('sod', 100)

def add_variable_cases(basecase: driver.Case) -> list[driver.Case]:
    '''
    Returns a list of cases with different variable options set.

    basecase: A Case object to use to set the variable options.
    '''
    cases = []
    for variable in ['c', 'p', 'ie']:
        c = copy.deepcopy(basecase)
        c.variable = variable
        cases.append(c)
    return cases

def blast_cases(dofine: bool) -> list[driver.Case]:
    '''
    Returns a list of cases for the blast-wave problem.
    '''
    cases = []
    basecase  = default_hydro_case()
    basecase.problem = 'blast'
    basecase.numcells = 400
    basecase.integrator = 'hancock'
    basecase.output = '{variable}_{limiter}_{integrator}_{numcells}'
    for limiter in ['dminmod', 'minmod']:
        basecase.limiter = limiter
        cases.extend(add_variable_cases(basecase))

    basecase.limiter = 'dminmod'

    case = copy.deepcopy(basecase)
    case.variable = 'p'
    case.integrator = 'RK2'
    cases.append(case)

    if dofine:
        case = copy.deepcopy(basecase)
        case.numcells = 3200
        case.integrator = 'hancock'
        case.variable = 'p'
        cases.append(case)

    return cases

def simple_cases(dofine: bool) -> list[driver.Case]:
    '''
    Returns a list of cases for the simple-wave problem.
    '''
    cases = []
    basecase  = default_hydro_case()
    basecase.problem = 'simple'
    basecase.output = '{variable}_{integrator}_{limiter}_{numcells}'
    cells = [100, 200, 400, 800]
    if dofine:
        cells.extend([1600, 3200, 6400])
    for numcells in cells:
        for integrator in ['hancock', 'RK2']:
            for limiter in ['dminmod', 'none']:
                basecase.numcells = numcells
                basecase.integrator = integrator
                basecase.limiter = limiter
                cases.extend(add_variable_cases(basecase))
    
    return cases

def sod_cases(_) -> list[driver.Case]:
    '''
    Returns a list of cases for Sod's problem.
    '''
    cases = []
    basecase  = default_hydro_case()
    basecase.problem = 'sod'
    basecase.numcells = 100
    basecase.output = '{variable}_{integrator}'
    cases.extend(add_variable_cases(basecase))

    case = copy.deepcopy(basecase)
    case.variable = 'p'
    case.integrator = 'RK2'
    cases.append(case)

    return cases

def sod_flip_cases(_) -> list[driver.Case]:
    '''
    Returns a list of cases for the flipped Sod'sproblem.
    '''
    cases = []
    case  = default_hydro_case()
    case.problem = 'sod_flip'
    case.numcells = 100
    cases.append(case)
    return cases

def leblanc_cases(_) -> list[driver.Case]:
    '''
    Returns a list of cases for LeBlanc's problem.
    '''
    cases = []
    basecase  = default_hydro_case()
    basecase.problem = 'leblanc'
    basecase.numcells = 100
    basecase.output = '{variable}_{limiter}_{integrator}'
    for limiter in ['dminmod', 'minmod']:
        basecase.limiter = limiter
        cases.extend(add_variable_cases(basecase))

    basecase.limiter = 'dminmod'
    case = copy.deepcopy(basecase)
    case.variable = 'p'
    case.integrator = 'RK2'
    cases.append(case)

    return cases

def twoExpansion_cases(_) -> list[driver.Case]:
    '''
    Returns a list of cases for the double-expansion problem.
    '''
    cases = []
    basecase  = default_hydro_case()
    basecase.problem = 'twoExpansion'
    basecase.numcells = 100
    basecase.output = '{variable}_{limiter}_{integrator}'
    for limiter in ['dminmod', 'minmod']:
        basecase.limiter = limiter
        cases.extend(add_variable_cases(basecase))

    basecase.limiter = 'dminmod'
    case = copy.deepcopy(basecase)
    case.variable = 'p'
    case.integrator = 'RK2'
    cases.append(case)

    return cases

def test3_cases(_) -> list[driver.Case]:
    '''
    Returns a list of cases for the Toro's Test 3 problem.
    '''
    cases = []
    basecase  = default_hydro_case()
    basecase.problem = 'test3'
    basecase.numcells = 100
    basecase.output = '{variable}_{limiter}_{integrator}'
    for limiter in ['dminmod', 'minmod']:
        basecase.limiter = limiter
        cases.extend(add_variable_cases(basecase))

    basecase.limiter = 'dminmod'
    case = copy.deepcopy(basecase)
    case.variable = 'p'
    case.integrator = 'RK2'
    cases.append(case)

    return cases

def test5_cases(_) -> list[driver.Case]:
    '''
    Returns a list of cases for Toro's Test 5 problem.
    '''
    cases = []
    basecase  = default_hydro_case()
    basecase.problem = 'test5'
    basecase.numcells = 100
    basecase.output = '{variable}_{integrator}'
    cases.extend(add_variable_cases(basecase))

    case = copy.deepcopy(basecase)
    case.variable = 'p'
    case.integrator = 'RK2'
    cases.append(case)

    return cases

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser('benchmarks.py', description='Runs benchmarks.')

    defoutputdir = 'benchmark_' + f'{datetime.now():%Y%m%d%H%M}'
    parser.add_argument('problem', nargs='*', help=f'Problem names to run (default: {driver.problems})')
    parser.add_argument('--outputdir', default=defoutputdir, help='Output directory (default: %(default)s).')
    parser.add_argument('--dofine', default=False, action='store_true', help='Do fine-grid cases (default: %(default)s).')

    args = parser.parse_args()
    if len(args.problem) == 0:
        args.problem = driver.problems

    # We don't know the log_file name yet.
    # But we set sys.stderr here, so that we can log exceptions below.
    logger = base.stepper.Logger()
    sys.stderr = logger
    
    # keep a list of the exceptions
    exceptions = []
    # run the cases for each problem name
    for p in args.problem:
        # form the output directory for this problem
        outbase = os.path.join(args.outputdir, p)
        if not os.path.exists(outbase):
            os.makedirs(outbase)
        # create the list of cases to run by calling one
        # of the functions above
        cases = locals()[p+'_cases'](args.dofine)
        # run each case
        for c in cases:
            c.output = os.path.join(outbase, c.output)
            thiscase = c.output.format(**vars(c))
            print(f'Running {thiscase}')
            try:
                driver.run_case(c, logger)
            except FileExistsError:
                print('...output already exists...skipping.')
            except Exception as e:
                logger.write(f'Exception occurred: {e}\n')
                exceptions.append((thiscase, f'{e}'))
    # output a summary of all the runs
    fd = open(os.path.join(args.outputdir, 'summary.dat'), 'w')
    logsummary = base.stepper.Logger(fd)
    if len(exceptions) == 0:
        logsummary.write('No case failed!\n')
    else:
        logsummary.write(f'Summary: {len(exceptions)} cases failed.\n')
        for i, ex in enumerate(exceptions):
            logsummary.write(f'{i}. {ex[0]}\n')
            logsummary.write(f'   Exception: {ex[1]}\n')
