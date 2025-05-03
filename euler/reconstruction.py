# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Reconstruction classes.
'''

import abc
from typing import Optional
import numpy as np

import base.bc
import base.mesh

import euler.state as state

class ReconBase(abc.ABC):
    '''
    Base class for all reconstructions.
    '''
    def __init__(self,
                 name: str,
                 mesh1d: base.mesh.Mesh1D_Equally_Spaced,
                 bc1d: base.bc.BC1D):
        '''
        name: String identifier of reconstruction.
        mesh1d: Mesh object.
        bc1d: Boundary condition object.
        '''
        self.name = name
        self.mesh = mesh1d
        self.bcLeft = bc1d.left
        self.bcRight = bc1d.right
    @abc.abstractmethod
    def recon(self,
              u: state.ConservativeVec,
              dt: Optional[float]=None) -> tuple[state.ConservativeVec,
                                                 state.ConservativeVec]:
        '''
        Returns reconstructed left and right states
        as ConservativeVec(numedges).

        u: ConservativeVec(numcells) of cell-centered values.
        dt: Current time step.  This may be used for a predictor.
        '''
        pass
    def applyBC(self,
                cL: state.ConservativeVec,
                cR: state.ConservativeVec) -> None:
        num_components = cL.num_components()
        if self.bcLeft.name == 'periodic':
            for c in range(num_components):
                cL[c, 0] = cL[c, -1]
                cR[c, -1] = cR[c, 0]
        else:
            if self.bcLeft.name == 'dirichlet':
                cL.set_vec(0, self.bcLeft.u)
            elif self.bcLeft.name == 'reflection':
                r = cR.get_vec(0).get_reflected()
                cL.set_vec(0, r)
            if self.bcRight.name == 'dirichlet':
                cR.set_vec(-1, self.bcRight.u)
            elif self.bcRight.name == 'reflection':
                r = cL.get_vec(-1).get_reflected()
                cR.set_vec(-1, r)

class Constant(ReconBase):
    '''
    This reconstruction represents the solution as a constant in each cell.
    '''
    def __init__(self,
                 mesh1d: base.mesh.Mesh1D_Equally_Spaced,
                 bc1d: base.bc.BC1D):
        super().__init__('constant', mesh1d, bc1d)
    def recon(self,
              u: state.ConservativeVec,
              _dt: Optional[float]=None) -> tuple[state.ConservativeVec,
                                                  state.ConservativeVec]:
        '''
        See base class.  dt argument is not used.
        '''
        ne = self.mesh.numedges()
        nc = self.mesh.numcells()
        cL = state.ConservativeVec(ne)
        cR = state.ConservativeVec(ne)
        num_components = u.num_components()
        for c in range(num_components):
            cL[c, 1:] = u[c, :]
            cR[c, :-1] = u[c, :]
        self.applyBC(cL, cR)
        return cL, cR

class Central(ReconBase):
    '''
    This reconstruction is a linear function in each cell.  The unlimited
    slope is formed using a central-difference approximation. With the
    limiting option, this reconstruction is the foundation of MUSCL.

    If dt is not None in a call to recon(), then a Hancock predictor is used
    to obtain the predicted left and right states. This results in an
    overall method that is often referred to as MUSCL-Hancock.
    '''
    def __init__(self,
                 mesh1d: base.mesh.Mesh1D_Equally_Spaced,
                 bc1d: base.bc.BC1D,
                 limiter: Optional[str]=None,
                 variable='c',
                 eos=None,
                 enforce_conservative=False):
        super().__init__('central', mesh1d, bc1d)
        '''
        mesh: mesh object.
        bc1d: boundary condition object.
        limiter: Type of slope limiter. minmod, dminmod (double minmod), or None.
        variable: Variable set to reconstruct. Options are:
            c: conservative {rho,rho*v,rho*E}
            p: primitive {rho,v,p}
            ie: {rho,rho*v,rho*e}
        eos: EOS object
        enforce_conservation: If true, shift the slopes so that 
        '''
        if limiter not in ['minmod', 'dminmod', None]:
            raise ValueError(f'Unrecognized limiter {limiter}')
        self.limiter = limiter
        if variable not in ['c', 'p', 'ie']:
            raise ValueError(f'Unrecognized variable {variable}')
        self.eos = eos
        self.enforce_conservative = enforce_conservative
        # Set the conversion functions to and from ConservativeVec 
        # and the variables being limited.
        if variable == 'p':
            self.wvar = state.DVPVec
            self.cons_to_w = state.Conservative_to_DVP
            self.w_to_cons = state.DVP_to_Conservative
            self.w_to_flux = state.DVP_to_Flux
        elif variable == 'ie':
            self.wvar = state.ConsIEVec
            self.cons_to_w = state.Conservative_to_ConsIE
            self.w_to_cons = state.ConsIE_to_Conservative
            self.w_to_flux = state.ConsIE_to_Flux
        else: # must be None
            self.wvar = None
            self.cons_to_w = None
            self.w_to_cons = None
            self.w_to_flux = state.Conservative_to_Flux
    def recon(self,
              u: state.ConservativeVec,
              dt: Optional[float]=None) -> tuple[state.ConservativeVec,
                                                 state.ConservativeVec]:
        '''
        See base class.  If dt is not None, then the Hancock predictor is
        applied.
        '''
        ne = self.mesh.numedges()
        nc = self.mesh.numcells()
        if self.limiter == 'dminmod':
            limfactor = 2
        elif self.limiter == 'minmod':
            limfactor = 1
        # Create the left and right states of the variable
        # type being reconstructed
        if self.wvar is None:
            w = u
            wL = state.ConservativeVec(length=ne)
            wR = state.ConservativeVec(length=ne)
            s = state.ConservativeVec(length=nc)
            diff = state.ConservativeVec(length=ne)
        else:
            w = self.cons_to_w(u, self.eos)
            wL = self.wvar(length=ne)
            wR = self.wvar(length=ne)
            s = self.wvar(length=nc)
            diff = self.wvar(length=ne)
        # Compute the slopes centered on each face
        num_components = u.num_components()
        for c in range(num_components):
            diff[c, :-1] = (w[c,:] - np.roll(w[c, :], 1)) / self.mesh.dxcell()
            diff[c, -1] = diff[c, 0]
            # Unless periodic BCs, zero the slopes in the cells
            # adjacent to the boundary.  This is only first-order
            # accurate and should be improved.
            if self.bcLeft.name != 'periodic':
                diff[c, 0] = 0
                diff[c, -1] = 0
        # Compute the unlimited slopes
        for c in range(num_components):
            s[c, :] = 0.5 * (diff[c, 1:] + diff[c, :-1])
        if self.limiter is not None:
            # Then apply the slope limiter
            for c in range(num_components):
                absdiff = limfactor * np.abs(diff[c, :])
                sgndiff = np.sign(diff[c, :])
                zero_extrema = 0.5 * (1 + sgndiff[1:] * sgndiff[:-1])
                s[c, :] = np.minimum(np.abs(s[c, :]), absdiff[1:])
                s[c, :] = np.minimum(s[c, :], absdiff[:-1]) * zero_extrema * sgndiff[:-1]
        # Find the edge states using the (limited) slopes
        dx2 = self.mesh.dxcell() * 0.5
        for c in range(num_components):
            wL[c, 1:] = w[c, :] + dx2 * s[c, :]
            wR[c, :-1] = w[c, :] - dx2 * s[c, :]
        # Convert back to conservative variables
        if self.wvar is None:
            cL = wL
            cR = wR
        else:
            cL = self.w_to_cons(wL, self.eos)
            cR = self.w_to_cons(wR, self.eos)
            if self.enforce_conservative:
                # Shift reconstructed edge values to ensure their average
                # is the original cell-centered value. TODO: Consider a multiplier
                # instead, at least for non-negative quantities.
                for c in range(num_components):
                    uavg = 0.5 * (cL[c, 1:] + cR[c, :-1])
                    cL[c, 1:] += u[c, :] - uavg
                    cR[c, :-1] += u[c, :] - uavg
        self.applyBC(cL, cR)
        if dt is not None:
            # Apply Hancock predictor
            dm = 0.5 * dt / self.mesh.dxcell()
            fL = state.Conservative_to_Flux(cL, self.eos)
            fR = state.Conservative_to_Flux(cR, self.eos)
            for c in range(num_components):
                dw = dm * (fR[c, :-1] - fL[c, 1:])
                cL[c, 1:] += dw
                cR[c, :-1] += dw
                cL[c, 0] += dw[-1]
                cR[c, -1] += dw[0]
        return cL, cR

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    '''
    Tests.
    '''
    import ic
    import base.material
    bcLeft = base.bc.BCReflection()
    uOut = state.ConservativeVec(rho=3, rhov=10, rhoE=10)
    bcRight = base.bc.BCDirichlet(uOut)
    bc1d = base.bc.BC1D(bcLeft, bcRight)
    uL = state.ConservativeVec(rho=2, rhov=7, rhoE=5)
    uR = state.ConservativeVec(rho=1, rhov=1, rhoE=1)
    m1d = base.mesh.Mesh1D_Equally_Spaced(10)
    icr = ic.ICRiemann(m1d, uL, uR)
    u = icr.get_ic()
    recon = Constant(m1d, bc1d)
    urL, urR  = recon.recon(u, 0)
    print(f'urL: {urL}')
    print(f'urR: {urR}')
    u = icr.get_ic()
    eos = base.material.GammaEOS()
    recon = Central(m1d, bc1d, eos=eos)
    urL, urR  = recon.recon(u, 0)
    print(f'urL: {urL}')
    print(f'urR: {urR}')
