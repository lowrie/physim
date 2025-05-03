# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for setting initial conditions.
'''
import math

from base.ic import ICBase
import base.bc
import base.material
import base.mesh

import trt.state as state

class ICMarshak(ICBase):
    def __init__(self,
                 mesh: base.mesh.Mesh1D_Equally_Spaced,
                 mat: base.material.Material):
        super().__init__('marshak')
        self.mesh = mesh
        Twall = 10.0 # keV
        T = 0.001 # keV
        rho = 1.0 # g/cc
        eos = mat.eos
        eWall = eos.e_rho_T(rho, Twall)
        self.uWall = state.RadiationVec(e=eWall)
        self.uWall.set_tr(Twall)
        e = eos.e_rho_T(rho, T)
        self.u = state.RadiationVec(rho=rho, e=e)
        self.u.set_tr(T)
    def get_ic(self) -> state.RadiationVec:
        '''
        Returns RadiationVec(numcells)
        '''
        nc = self.mesh.numcells()
        c = state.RadiationVec(length=nc, rho=self.u.rho,
                               rade=self.u.rade, e=self.u.e)
        return c

class ICRelax(ICBase):
    def __init__(self,
                 mesh: base.mesh.Mesh1D_Equally_Spaced,
                 mat: base.material.Material):
        super().__init__('relax')
        self.mesh = mesh
        Tr = 10.0 # keV
        T = 0.001 # keV
        rho = 1.0 # g/cc
        eos = mat.eos
        e = eos.e_rho_T(rho, T)
        self.u = state.RadiationVec(rho=rho, e=e)
        self.u.set_tr(Tr)
    def get_ic(self) -> state.RadiationVec:
        '''
        Returns numcells-array of RadiationVec
        '''
        nc = self.mesh.numcells()
        c = state.RadiationVec(length=nc, rho=self.u.rho,
                               rade=self.u.rade, e=self.u.e)
        return c

def initialize(problem: str,
               numcells: int) -> tuple[state.RadiationVec,
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
    bc1d: Boundary condition object.
    tmax: Final time.
    '''
    mat = base.material.Material()
    if problem == 'marshak':
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
    Do a test.
    '''
    m1d = base.mesh.Mesh1D_Equally_Spaced(10)
    mat = base.material.Material()

    ic = ICMarshak(m1d, mat)
    u = ic.get_ic()
    print(f'{u}')
    ic = ICRelax(m1d, mat)
    u = ic.get_ic()
    print(f'{u}')
