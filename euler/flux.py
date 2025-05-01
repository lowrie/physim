# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
(Approximate) Riemann flux solvers.
'''

import numpy as np

from base.material import GammaEOS, BaseEOS

import euler.state as state
import euler.riemann as riemann

def flux_exact(uL: state.ConservativeVec,
               uR: state.ConservativeVec,
               eos: GammaEOS) -> state.FluxVec:
    '''
    Returns the exact Riemann flux.

    uL: left state.
    uR: right state.
    eos: GammaEOS object.
    '''
    nedges = uL.length()
    f = state.FluxVec(nedges)
    for i in range(nedges):
        flux = riemann.riemann_flux(uL.get_vec(i), uR.get_vec(i), eos)
        f.set_vec(i, flux)
    return f

def hllc_star_state(v: state.DVPVec,
                    u: state.ConservativeVec,
                    s: float,
                    sStar: float) -> state.FluxVec:
    '''
    Returns the flux vector corresponding to the star-state for the HLLC flux.

    v: left or right state
    u: ConservativeVec corresponding to v
    s: sL or sR
    sStar: star wavespeed 
    '''
    ratio = (s - v.v) / (s - sStar)
    frho = u.rho * ratio
    frhov = frho * sStar
    frhoE = ratio * (u.rhoE + (sStar - v.v) * (sStar * u.rho + v.p / (s - v.v)))
    return state.FluxVec(rho=frho, rhov=frhov, rhoE=frhoE)

def flux_hllc(uL: state.ConservativeVec,
              uR: state.ConservativeVec,
              eos: BaseEOS):
    '''
    Returns the HLLC flux.

    uL: left state.
    uR: right state.
    eos: EOS object.
    '''
    dvpL = state.Conservative_to_DVP(uL, eos)
    dvpR = state.Conservative_to_DVP(uR, eos)
    hL = (uL.rhoE + dvpL.p) / uL.rho
    hR = (uR.rhoE + dvpR.p) / uR.rho
    rootRhoL = np.sqrt(uL.rho)
    rootRhoR = np.sqrt(uR.rho)
    vHat = (rootRhoL * dvpL.v + rootRhoR * dvpR.v) / (rootRhoL + rootRhoR)
    hHat = (rootRhoL * hL + rootRhoR * hR) / (rootRhoL + rootRhoR)
    cHat = np.sqrt((eos.gamma - 1) * (hHat - 0.5 * vHat * vHat))
    sL = vHat - cHat
    sR = vHat + cHat

    # Ensure sL and sR are at least as large as L,R eigenvalues.
    # Toro doesn't mention this, and it's key for the vacuum problems.
    cL = eos.c_rho_p(dvpL.rho, dvpL.p)
    cR = eos.c_rho_p(dvpR.rho, dvpR.p)
    sL = np.minimum(sL, dvpL.v - cL)
    sR = np.maximum(sR, dvpR.v + cR)

    sStar = (dvpR.p - dvpL.p + uL.rhov * (sL - dvpL.v) - uR.rhov * (sR - dvpR.v)) / \
        (uL.rho * (sL - dvpL.v) - uR.rho * (sR - dvpR.v))
    starL = hllc_star_state(dvpL, uL, sL, sStar)
    starR = hllc_star_state(dvpR, uR, sR, sStar)
    fL = state.DVP_to_Flux(dvpL, eos)
    fR = state.DVP_to_Flux(dvpR, eos)
    f = state.FluxVec(fL.length())
    for c in range(uL.num_components()):
        f[c, :] = np.where(sL > 0,
                           fL[c, :], 
                           np.where(sStar >= 0,
                                    fL[c, :] + sL * (starL[c, :] - uL[c, :]),
                                    np.where(sR >= 0,
                                             fR[c, :] + sR * (starR[c, :] - uR[c, :]),
                                             fR[c, :])))
        # The logic above corresponds to:
        # if sL > 0:
        #     f = fL
        # elif sStar >= 0:
        #     f = fL + sL * (starL - uL)
        # elif sR >= 0:
        #     f = fR + sR * (starR - uR)
        # else:
        #     f = fR
    return f

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    '''
    Tests.
    '''
    eos = GammaEOS()
    uL = state.ConservativeVec(10)
    uR = state.ConservativeVec(uL.length())
    f = flux_exact(uL, uR, eos)
    print(f'exact: {f}')
    f = flux_hllc(uL, uR, eos)
    print(f'hllc: {f}')