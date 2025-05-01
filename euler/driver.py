#!/usr/bin/env python
# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for running a single Euler case.
'''
import argparse
import numpy as np
import sys

import base.bc as bc
import base.material as material
import base.mesh as mesh
import base.stepper as stepper

import euler.flux as flux
import euler.ic as ic
import euler.reconstruction as reconstruction
import euler.state as state

# input options
problems = ['sod', 'sod_flip', 'simple', 'twoExpansion', 'test3', 'test5', 'leblanc', 'blast']
fluxers = ['exact', 'hllc']
limiters = ['none', 'dminmod', 'minmod']
variables = ['c', 'p', 'ie', 'char']
integrators = ['RK1', 'RK2', 'hancock']
reconstructors = ['constant', 'central']

class Case(stepper.Case):
    '''
    Container that defines the options for a Euler case.
    '''
    def __init__(self, problem: str, numcells: int,
                 output: str='{problem}_{numcells}',
                 rampdt_cycle: int=10,
                 cfl: float=0.8, enforceconservation: bool=False,
                 flux: str='hllc', integrator: str='hancock',
                 limiter: str='dminmod', recon: str='central',
                 variable: str='c', **kwargs):
        '''
        problem: Problem name identifier.
        numcells: Number of mesh cells.
        rampdt_cycle: Ramp up time step over this many cycles.
        cfl: Safety factor for time step.
        flux: Flux solve: {"hllc", "exact"}
        integrator: Time integrator: {"hancock", "RK1", "RK2"}
        limiter: Limiter: {None, "minmod, "dminmod"}
        recon: Reconstruction operator: {"constant", "central"}
        variable: Variables to reconstruct {"c", "p", "char", "ie"}
        '''
        super().__init__(output=output, rampdt_cycle=rampdt_cycle, **kwargs)
        self.problem = problem
        self.numcells = numcells
        self.cfl = cfl
        self.enforceconservation = enforceconservation
        self.flux = flux
        self.integrator = integrator
        self.limiter = limiter
        self.recon = recon
        self.variable = variable

class Hydro(stepper.Physics):
    '''
    Hydro physics class.
    '''
    def __init__(self,
                 case: Case,
                 logger: stepper.Logger,
                 u: np.ndarray,
                 eos: material.BaseEOS,
                 mesh: mesh.Mesh1D_Equally_Spaced,
                 bc1d: bc.BC1D):
        super().__init__(case, logger)
        self.u = u
        self.eos = eos
        self.mesh = mesh

        if case.flux == 'exact':
            self.fluxer = flux.flux_exact
        else:
            self.fluxer = flux.flux_hllc

        if case.limiter == 'none':
            limiter = None
        else:
            limiter = case.limiter

        if case.recon == 'constant':
            self.reconer = reconstruction.Constant(mesh, bc1d)
        else:
            self.reconer = reconstruction.Central(mesh, bc1d, limiter, case.variable,
                                                  eos, case.enforceconservation)
    def get_max_dt(self) -> float:
        '''
        Returns the maximum allowable time step, based
        on the max absolute eigenvalue in the current solution.
        '''
        nc = self.mesh.numcells()
        dt = 1.0e50
        for i in range(nc):
            v = self.u.v(i)
            e = self.u.e(i)
            c = self.eos.c_rho_e(self.u.rho[i], e)
            eig = abs(v) + c
            dt = min(dt, self.mesh.dxcell(i) / eig)
        dt *= self.case.cfl
        return dt
    def update(self, dt: float) -> None:
        '''
        Updates the solution state over time dt.
        '''
        dtdx = dt / self.mesh.dxcell()
        dthancock = None
        if self.case.integrator == 'hancock':
            dthancock = dt
        u = self.u
        uReconL, uReconR = self.reconer.recon(u, dthancock)
        f = self.fluxer(uReconL, uReconR, self.eos)
        uStage1 = state.ConservativeVec(u.length())
        num_components = u.num_components()
        for c in range(num_components):
            uStage1[c, :] = u[c, :] - dtdx * (f[c, 1:] - f[c, :-1])
        if self.case.integrator in ['RK1', 'hancock']:
            # For single-stage integrators, we are done.
            for c in range(num_components):
                u[c, :] = uStage1[c, :]
        else:
            # We must be doing RK2, so do the next stage
            # to get the final value.
            for c in range(num_components):
                uStage1[c, :] = 0.5 * (uStage1[c, :] + u[c, :])
            uReconL, uReconR = self.reconer.recon(uStage1)
            f = self.fluxer(uReconL, uReconR, self.eos)
            for c in range(num_components):
                u[c, :] -= dtdx * (f[c, 1:] - f[c, :-1])
    def output_final(self):
        '''
        Write a plot file, which has the format with the following
        columns of data:

        x rho v p e
        '''
        v = state.Conservative_to_DVP(self.u, self.eos)
        e = self.eos.e_rho_p(v.rho, v.p)
        out = open(self.plotfile, 'w')
        case = self.case
        out.write(f'# {case.problem} {case.numcells} {case.cfl}\n')
        if self.reconer.name == 'constant':
            out.write(f'# Constant reconstruction\n')
        else:
            out.write(f'# {self.reconer.name} {case.limiter} {case.variable} {case.integrator}\n')
        for i in range(self.mesh.numcells()):
            out.write(f'{self.mesh.xcell(i):15.8e} {v.rho[i]:15.8e} {v.v[i]:15.8e} {v.p[i]:15.8e} {e[i]:15.8e}\n')
        

def run_case(case: Case,
             logger: stepper.Logger) -> None:
    '''
    Runs a problem defined by case.

    case: A Case object.
    logger: A Logger() object.
    '''
    u, material, m1d, bc1d, case.tmax = ic.initialize(case.problem, case.numcells)
    hydro = Hydro(case, logger, u, material.eos, m1d, bc1d)
    stepper.stepper(hydro)

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser('driver.py', description='Runs the Euler code.')

    c = Case('sod', 100) # use to get default values
    # The options must be a subset of the Case attributes
    parser.add_argument('problem', choices=problems, help='Problem name to run.')
    parser.add_argument('numcells', type=int,
                        help='Number of mesh cells.')
    parser.add_argument('--cfl', type=float, default=c.cfl,
                        help='CFL number (default: %(default)s).')
    parser.add_argument('--enforceconservation', default=c.enforceconservation, action='store_true',
                        help='Enforce conservative reconstruction. Only affects central recon,'
                        ' with non-conservative variables. Very experimental (default: %(default)s).')
    parser.add_argument('--flux', choices=fluxers, default=c.flux, help='Flux function. (default: %(default)s)')
    parser.add_argument('--integrator', choices=integrators, default=c.integrator, help='Time integrator (default: %(default)s).')
    parser.add_argument('--limiter', choices=limiters, default=c.limiter, help='Limiter (default: %(default)s).')
    parser.add_argument('--output', default=c.output, help='Output filename prefix (default: %(default)s).')
    parser.add_argument('--maxcycle', default=c.maxcycle, type=int, help='Maximum number of cycles to run (default: %(default)s).')
    parser.add_argument('--recon', choices=reconstructors, default=c.recon, help='Type of reconstruction (default: %(default)s).')
    parser.add_argument('--variable', choices=variables, default=c.variable, help='Variable to reconstruct (default: %(default)s).')

    args = parser.parse_args()
    # Copy the CLI options into the Case object
    for k in vars(args):
        setattr(c, k, getattr(args, k))

    # Setup the logger and run the case.
    logger = stepper.Logger()
    sys.stderr = logger
    run_case(c, logger)
