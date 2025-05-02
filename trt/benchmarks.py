#!/usr/bin/env python
# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for running a series of benchmark trt problems.
'''
import argparse
import numpy as np
import os
import sys
from datetime import datetime

import base.stepper

import trt.driver as driver

def marshak_cases(_):
    '''
    Returns a list of cases for the Marshak wave problem.
    '''
    cases = [driver.Case('marshak', 100)]
    return cases

def relax_cases(_):
    '''
    Returns a list of cases for the relaxation problem.
    '''
    cases = [driver.Case('relax', 10)]
    return cases

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser('benchmarks.py', description='Runs benchmarks.')

    defoutputdir = 'benchmark_' + f'{datetime.now():%Y%m%d%H%M}'
    parser.add_argument('problem', nargs='*')
    parser.add_argument('--outputdir', default=defoutputdir, help='Output directory (default: %(default)s).')
    parser.add_argument('--dofine', default=False, action='store_true', help='Do fine-grid cases (default: %(default)s).')

    args = parser.parse_args()
    if len(args.problem) == 0:
        args.problem = driver.problems[:]

    # We don't know the log_file name yet.
    # But we set sys.stderr here, so that we can log exceptions below.
    logger = base.stepper.Logger()
    sys.stderr = logger
    
    exceptions = []
    for p in args.problem:
        outbase = os.path.join(args.outputdir, p)
        if not os.path.exists(outbase):
            os.makedirs(outbase)
        cases = locals()[p+'_cases'](args.dofine)
        for c in cases:
            c.output = os.path.join(outbase, c.output)
            thiscase = c.output.format(**vars(c))
            print(f'Running {thiscase}')
            try:
                driver.run_case(c, logger, output_logger=False)
            except FileExistsError:
                print('...output already exists...skipping.')
            except Exception as e:
                logger.write(f'Exception occurred: {e}\n')
                exceptions.append((thiscase, f'{e}'))
    fd = open(os.path.join(args.outputdir, 'summary.dat'), 'w')
    logsummary = base.stepper.Logger(fd)
    if len(exceptions) == 0:
        logsummary.write('No case failed!\n')
    else:
        logsummary.write(f'Summary: {len(exceptions)} cases failed.\n')
        for i, ex in enumerate(exceptions):
            logsummary.write(f'{i}. {ex[0]}\n')
            logsummary.write(f'   Exception: {ex[1]}\n')
