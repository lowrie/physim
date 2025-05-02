#!/usr/bin/env python
# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for running a single radiation case.
'''
import argparse
import numpy as np
import scipy.linalg as linalg
import sys

import base.bc as bc
import base.material as material
import base.mesh as mesh
import base.stepper as stepper
from base.constants import light_speed

import trt.ic as ic
import trt.state as state

# input options
problems = ['marshak', 'relax']
integrators = ['BE']

class Case(stepper.Case):
    '''
    Container that defines the options for a radiation case.
    '''
    def __init__(self,
                 problem: str,
                 numcells: int,
                 output: str='{problem}_{numcells}',
                 rampdt_cycle: int=10,
                 first_dt_frac: float=0.000001,
                 dt_dE_frac: float=0.05,
                 rho: float=1.0,
                 **kwargs):
        super().__init__(output=output, rampdt_cycle=rampdt_cycle, **kwargs)
        self.problem = problem
        self.numcells = numcells
        self.first_dt_frac = first_dt_frac
        self.dt_dE_frac = dt_dE_frac
        self.rho = rho

class Radiation(stepper.Physics):
    '''
    Physics class for radiation-only problems.
    '''
    def __init__(self,
                 case: Case,
                 logger: stepper.Logger,
                 mat: material.Material,
                 mesh1d: mesh.Mesh1D_Equally_Spaced,
                 bc1d: bc.BC1D,
                 output_logger: bool=True):
        super().__init__(case, logger, output_logger)
        self.eos = mat.eos
        self.mesh = mesh1d
        self.absorption = mat.absorption
        self.totalxc = mat.totalxc
        self.bcLeft = bc1d.left
        self.bcRight = bc1d.right
        self.case = case
        self.max_deltaE = None
        self.dt_previous = None
    def get_max_dt(self, u: state.RadiationVec) -> float:
        '''
        Computes the maximum allowable time step, as follows:

        1) for the first time step, a fraction of the mean-flight time for absorption,
        2) otherwise, targetting the maximum change in radiation energy, based on the 
           previous time step.  The max change is computed during the previous call to
           update().
        '''
        if self.max_deltaE is None:
            # first time step
            T = self.eos.T_rho_e(self.case.rho, u.e)
            relax_rate = np.max(light_speed * self.absorption(self.case.rho, T))
            dt = self.case.first_dt_frac / relax_rate
        else:
            # Target based on max change in radiation energy, but don't
            # let the timestep increase too fast.
            dt = self.dt_previous * min(1.1, self.case.dt_dE_frac / self.max_deltaE)
        return dt
    def update(self, u: state.RadiationVec, dt: float) -> None:
        '''
        Updates the radiation field using backward-Euler time integration, with
        the T^4 terms linearized.

        See also the base class.
        '''
        dx = self.mesh.dxcell()
        rho = self.case.rho
        T = self.eos.T_rho_e(rho, u.e)
        ne = self.mesh.numedges()
        nc = self.mesh.numcells()
        # Find the left and right temperatures at the edges, in order
        # to determine the respective total cross-section.
        TL = np.zeros(ne)
        TR = np.zeros(ne)
        TL[1:] = T[:]
        TR[:-1] = T[:]
        if self.bcLeft.name == 'periodic':
            TL[0] = T[-1]
            TR[-1] = T[0]
        else:
            if self.bcLeft.name == 'dirichlet':
                TL[0] = self.eos.T_rho_e(rho, self.bcLeft.u.e)
            else:
                T[0] = T[0]
            if self.bcRight.name == 'dirichlet':
                TR[-1] = self.eos.T_rho_e(rho, self.bcRight.u.e)
            else:
                TR[-1] = T[-1]
        # Average the left and right temperatures and determine the
        # diffusion coefficient.
        Tavg = np.pow(0.5 * (np.pow(TL, 4) + np.pow(TR, 4)), 0.25)
        dtc = dt * light_speed
        diff_coeff = dtc / (3.0 * self.totalxc(rho, Tavg) * dx * dx)
        # Build the banded matrix to solve and right-hand side.
        # 
        # The matrix is tri-diagonal, unless
        # the BCs are periodic, which then places an entry in the upper
        # right and lower left of the matrix.  Since the matrix is
        # symmetric, only the diagonal and lower bands need to be stored.
        # See the documentation for scipy.linalg.solveh_banded().
        siga = self.absorption(rho, T)
        T3 = np.pow(T, 3)
        T4 = T3 * T
        cv = self.eos.cv_rho_e(rho, u.e)
        c0 = dtc * siga / (rho * cv + 4 * dtc * state.radiation_constant * T3 * siga)
        c1 = -c0 * state.radiation_constant * T4
        rhs = dtc * siga * state.radiation_constant * (T4 + 4 * c1 * T3) + u.rade
        numbands = 2
        if self.bcLeft.name == 'periodic':
            numbands = 3
        ab = np.zeros((numbands, nc))
        # diagonal
        ab[0,:] = 1 + dtc * siga * (1 - 4 * state.radiation_constant * c0 * T3) + \
            diff_coeff[:-1] + diff_coeff[1:]
        # lower-diagonal
        ab[1,:-1] = -diff_coeff[1:-1]
        if self.bcLeft.name == 'periodic':
            # matrix corner
            ab[2, 0] = -diff_coeff[0]
        else:
            if self.bcLeft.name == 'dirichlet':
                rhs[0] += 2 * diff_coeff[0] * self.bcLeft.u.rade
                ab[0,0] += diff_coeff[0]
            else: # treat as Neumann
                ab[0,0] -= diff_coeff[0]
            if self.bcRight.name == 'dirichlet':
                rhs[-1] += 2 * diff_coeff[-1] * self.bcRight.u.rade
                ab[0,-1] += diff_coeff[-1]
            else: # treat as Neumann
                ab[0,-1] -= diff_coeff[-1]
        # Solve the matrix equation for the new rad energy
        tote = u.rade.copy()
        u.rade = linalg.solveh_banded(ab, rhs, overwrite_ab=True, overwrite_b=True, lower=True)
        # Update the temperature
        T += c0 * u.rade + c1
        u.e = self.eos.e_rho_T(rho, T)
        # Determine the max absolution change in radiation energy for use by get_max_dt().
        toteNew = u.rade
        self.max_deltaE = np.max(np.abs((toteNew - tote) / tote))
        self.dt_previous = dt
    def output_final(self, u: state.RadiationVec) -> None:
        '''
        Write a plot file, which has the format with the following
        columns of data:

        x T Tr
        '''
        T = self.eos.T_rho_e(self.case.rho, u.e)
        Tr = u.tr()
        out = open(self.plotfile, 'w')
        case = self.case
        out.write(f'# {case.problem} {case.numcells}\n')
        for i in range(self.mesh.numcells()):
            out.write(f'{self.mesh.xcell(i):15.8e} {T[i]:15.8e} {Tr[i]:15.8e}\n')

def run_case(case: Case,
             logger: stepper.Logger,
             output_logger: bool=True) -> None:
    '''
    Runs a problem defined by case.
    '''
    u, material, m1d, bc1d, case.tmax = ic.initialize(case.problem, case.numcells)
    rad = Radiation(case, logger, material, m1d, bc1d, output_logger)
    stepper.stepper(rad, u)

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    np.seterr(all='raise')

    parser = argparse.ArgumentParser('driver.py', description='Runs radiation code.')

    c = Case('marshak', 100) # use to get default values
    # The options must be a subset of the Case attributes
    parser.add_argument('problem', choices=problems, help='Problem name to run.')
    parser.add_argument('numcells', type=int,
                        help='Number of mesh cells.')
    parser.add_argument('--dt_dE_frac', type=float, default=c.dt_dE_frac,
                        help='Target fraction of max energy chance per time step.')
    parser.add_argument('--output', default=c.output, help='Output filename prefix.')
    parser.add_argument('--maxcycle', default=c.maxcycle, type=int, help='Maximum number of cycles to run.')

    args = parser.parse_args()
    # Copy the CLI options into the Case object
    for k in vars(args):
        setattr(c, k, getattr(args, k))

    # Setup the logger and run the case.
    logger = stepper.Logger()
    sys.stderr = logger
    run_case(c, logger)
