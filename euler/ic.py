# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for setting initial conditions.
'''
import abc
import math
from typing import Optional

import base.bc
from base.ic import ICBase
import base.mesh
import base.material

import euler.state as state

class ICRiemann(ICBase):
    '''
    Initial condition for the Riemann problem.
    '''
    def __init__(self,
                 mesh: base.mesh.Mesh1D_Equally_Spaced,
                 uL: state.ConservativeVec,
                 uR: state.ConservativeVec,
                 xDiaphram: Optional[float]=None):
        '''
        mesh: mesh object.
        uL: Left state ConservativeVec
        uR: Right state ConservativeVec
        xDiaphram: x-location of jump between left and right states.
            If None, then the jump is placed at the domain center.
        '''
        super().__init__('riemann')
        self.mesh = mesh
        if xDiaphram is None:
            xDiaphram = 0.5 * (mesh.xedge(0) + mesh.xedge(mesh.numcells()))
        self.xDiaphram = xDiaphram
        self.uL = uL
        self.uR = uR
    def get_ic(self) -> state.ConservativeVec:
        '''
        Returns numcells-array of ConservativeVec
        '''
        nc = self.mesh.numcells()
        c = state.ConservativeVec(nc)
        for i in range(nc):
            if self.mesh.xcell(i) < self.xDiaphram:
                c.set_vec(i, self.uL)
            else:
                c.set_vec(i, self.uR)
        return c

class ICSimple(ICBase):
    '''
    Initial condition for a simple wave problem.
    '''
    def __init__(self,
                 mesh: base.mesh.Mesh1D_Equally_Spaced,
                 eos: base.material.GammaEOS,
                 rho0: float=1,
                 p0: float=1):
        '''
        mesh: mesh object.
        eos: GammaEOS object
        rho0: Reference density.
        p0: Reference pressure.
        '''
        super().__init__('simple')
        self.mesh = mesh
        self.eos = eos
        self.rho0 = rho0
        self.p0 = p0
    def _b(self, x: float):
        '''
        b-function for the IC.
        '''
        return 1 + 0.2 * math.sin(2 * math.pi * x)
    def get_ic(self) -> state.ConservativeVec:
        '''
        Returns numcells-array of ConservativeVec
        '''
        gamma = self.eos.gamma
        nc = self.mesh.numcells()
        prim = state.DVPVec(nc)
        c0 = math.sqrt(gamma * self.p0 / self.rho0)
        for i in range(nc):
            v = 2 * self._b(self.mesh.xcell(i)) / (gamma + 1)
            cs = 0.5 * v * (gamma - 1)
            p = self.p0 * math.pow(cs / c0, 2 * gamma / (gamma - 1))
            rho = self.rho0 * math.pow(p / self.p0, 1 / gamma)
            vec = state.DVPVec(rho=rho, v=v, p=p)
            prim.set_vec(i, vec)
        return state.DVP_to_Conservative(prim, self.eos)

class ICBlastwave(ICBase):
    '''
    Initial condition for the Woodward-Collela blast-wave problem.
    '''
    def __init__(self,
                 mesh: base.mesh.Mesh1D_Equally_Spaced,
                 eos: base.material.BaseEOS):
        '''
        mesh: mesh object.
        eos: GammaEOS object
        '''
        super().__init__('blast')
        self.mesh = mesh
        vL = state.DVPVec(rho=1, v=0, p=1000)
        vC = state.DVPVec(rho=1, v=0, p=0.01)
        vR = state.DVPVec(rho=1, v=0, p=100)
        self.uL = state.DVP_to_Conservative(vL, eos)
        self.uC = state.DVP_to_Conservative(vC, eos)
        self.uR = state.DVP_to_Conservative(vR, eos)
    def get_ic(self) -> state.ConservativeVec:
        '''
        Returns numcells-array of ConservativeVec
        '''
        nc = self.mesh.numcells()
        c = state.ConservativeVec(nc)
        for i in range(nc):
            if self.mesh.xcell(i) < 0.1:
                c.set_vec(i, self.uL)
            elif self.mesh.xcell(i) > 0.9:
                c.set_vec(i, self.uR)
            else:
                c.set_vec(i, self.uC)
        return c

def initialize(problem: str,
               numcells: int) -> tuple[state.ConservativeVec,
                                       base.material.Material,
                                       base.mesh.Mesh1D_Equally_Spaced,
                                       base.bc.BC1D,
                                       float]:
    '''
    Initializes a problem.

    problem: Problem name.
    numcells: number of mesh cells.

    Returns:
    u: numcell-array of the initial condition
    mat: Material object.
    m1d: Mesh object for a 1-D mesh.
    bc1d: boundary condition object.
    tmax: Final time.
    '''
    mat = base.material.Material()
    if problem in ['sod', 'sod_flip', 'twoExpansion', 'test3', 'test5', 'leblanc']:
        mat.eos = base.material.GammaEOS(1.4)
        xL = 0
        xR = 1
        if problem == 'sod':
            dvpL = state.DVPVec(rho=1, v=0, p=1)
            dvpR = state.DVPVec(rho=0.125, v=0, p=0.1) 
            tmax = 0.2
        elif problem == 'sod_flip':
            dvpR = state.DVPVec(rho=1, v=0, p=1)
            dvpL = state.DVPVec(rho=0.125, v=0, p=0.1) 
            tmax = 0.2
        elif problem == 'twoExpansion':
            dvpL = state.DVPVec(rho=1, v=-2, p=0.4) 
            dvpR = state.DVPVec(rho=1, v=2, p=0.4)
            tmax = 0.15
        elif problem == 'test3':
            dvpL = state.DVPVec(rho=1, v=0, p=1000) 
            dvpR = state.DVPVec(rho=1, v=0, p=0.01)
            tmax = 0.012
        elif problem == 'test5':
            dvpL = state.DVPVec(rho=5.99924, v=19.5975, p=460.894) 
            dvpR = state.DVPVec(rho=5.99242, v=-6.19633, p=46.0950)
            tmax = 0.035
        elif problem == 'leblanc':
            dvpL = state.DVPVec(rho=2, v=0, p=1000000000) 
            dvpR = state.DVPVec(rho=0.001, v=0, p=1)
            tmax = 0.0001
            xL = -10
            xR = 10
        m1d = base.mesh.Mesh1D_Equally_Spaced(numcells, xLeft=xL, xRight=xR)
        uL = state.DVP_to_Conservative(dvpL, mat.eos)
        uR = state.DVP_to_Conservative(dvpR, mat.eos)
        bcLeft = base.bc.BCDirichlet(uL)
        bcRight = base.bc.BCDirichlet(uR)
        icr = ICRiemann(m1d, uL, uR)
    elif problem == 'simple':
        m1d = base.mesh.Mesh1D_Equally_Spaced(numcells, xRight=2)
        gamma = 5/3.
        mat.eos = base.material.GammaEOS(gamma)
        bcLeft = base.bc.BCPeriodic()
        bcRight = base.bc.BCPeriodic()
        rho0 = 1
        p0 = rho0 * math.pow((gamma - 1) / (gamma + 1), 2) / gamma
#        p0 = 1
        icr = ICSimple(m1d, mat.eos, rho0=rho0, p0=p0)
        tmax = 0.75
    elif problem == 'blast':
        m1d = base.mesh.Mesh1D_Equally_Spaced(numcells)
        mat.eos = base.material.GammaEOS(1.4)
        icr = ICBlastwave(m1d, mat.eos)
        bcLeft = base.bc.BCReflection()
        bcRight = base.bc.BCReflection()
        tmax = 0.038
    elif problem == 'marshak':
        m1d = base.mesh.Mesh1D_Equally_Spaced(numcells)
        mat.absorp.constant = 100.0
        mat.scat.constant = 0.0
        icr = ICMarshak(m1d, mat)
        bcLeft = base.bc.BCDirichlet(icr.uWall)
        bcRight = base.bc.BCDirichlet(icr.u)
        tmax = 0.05
    elif problem == 'relax':
        m1d = base.mesh.Mesh1D_Equally_Spaced(numcells)
        mat.absorp.constant = 1.0
        mat.scat.constant = 0.0
        icr = ICRelax(m1d, mat)
        bcLeft = base.bc.BCReflection()
        bcRight = base.bc.BCReflection()
        tmax = 0.01
    else:
        raise ValueError(f'Unrecognized problem {problem}')
    u = icr.get_ic()
    bc1d = base.bc.BC1D(bcLeft, bcRight)
    return u, mat, m1d, bc1d, tmax

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    '''
    Tests.
    '''
    uL = state.ConservativeVec(rho=2, rhov=0, rhoE=5)
    uR = state.ConservativeVec(rho=1, rhov=1, rhoE=1)
    m1d = base.mesh.Mesh1D_Equally_Spaced(10)
    ic = ICRiemann(m1d, uL, uR)
    u = ic.get_ic()
    print(f'{u}')
    geos = base.material.GammaEOS()
    ic = ICSimple(m1d, geos)
    u = ic.get_ic()
    print(f'{u}')
