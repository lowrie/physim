# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for computing the exact solution to the Riemann problem.
'''
import math

from base.material import GammaEOS
import euler.state as state

def v_rare(dvp0: state.DVPVec,
           p: float,
           gamma: float,
           sgn: int) -> tuple[float, float, float, list[float]]:
    '''
    Computes the state after a rarefaction wave for
    a v + sgn * cs fan. Input:

    dvp0: pre-fan state
    p: Pressure post-fan
    gamma: EOS gamma
    sgn: +1 or -1

    Returns:
    v: Post-fan velocity
    dv: dv/dp
    rho: Post-fan density
    [v0 + sgn * c0, v + sgn * c]: Eigenvalues on either side of fan
    '''
    c0 = math.sqrt(gamma * dvp0.p / dvp0.rho)
    term1 = sgn * 2 * c0 / (gamma - 1)
    exp = (gamma - 1) / (2 * gamma)
    pRatio = p / dvp0.p
    cRatio = math.pow(pRatio, exp)
    v = dvp0.v - term1 * (1 - math.pow(pRatio, exp))
    dv = term1 * exp * math.pow(pRatio, exp - 1) / dvp0.p
    rho = dvp0.rho * math.pow(pRatio, 1 / gamma)
    c = c0 * cRatio
    return v, dv, rho, [dvp0.v + sgn * c0, v + sgn * c]

def v_shock(dvp0: state.DVPVec,
            p: float,
            gamma: float,
            sgn: int) -> tuple[float, float, float, list[float]]:
    '''
    Computes the state after a shock wave for
    a v + sgn * cs shock. Input

    dvp0: pre-shock state
    p: Pressure post-shock
    gamma: EOS gamma
    sgn: +1 or -1

    Returns:
    v: Post-shock velocity
    dv: dv/dp
    rho: Post-shock density
    [s]: shock speed
    '''
    c0 = math.sqrt(gamma * dvp0.p / dvp0.rho)
    pRatio = p / dvp0.p
    term1 = sgn * c0 * math.sqrt(2 / gamma)
    sqr = math.sqrt((gamma + 1) * pRatio + gamma - 1)
    v = dvp0.v - term1 * (1 - pRatio) / sqr
    dv = term1 * (1 / sqr + 0.5 * (1 - pRatio) * (gamma + 1) / math.pow(sqr, 3)) / dvp0.p
    mStar2 = sqr * sqr / (2 * gamma)
    rho = dvp0.rho * (gamma + 1) * mStar2 / ((gamma - 1) * mStar2 + 2)
    s = dvp0.v + sgn * math.sqrt(mStar2) * c0
    return v, dv, rho, [s]

def rare_xt(x_over_t: float,
            gamma: float,
            dvp0: state.DVPVec,
            sgn: int) -> state.DVPVec:
    '''
    Returns the solution within a self-similar rarefaction fan,
    centered on the origin, for a v + sgn * cs fan. Input:

    x_over_t: x/t value
    gamma: EOS gamma
    dvp0: DVPVec pre-fan state
    sgn: +1 or -1

    Returns a DVPVec of the solution.
    '''
    c0 = math.sqrt(gamma * dvp0.p / dvp0.rho)
    gm1_over_gp1 = (gamma - 1) / (gamma + 1)
    two_over_gp1 = 2 / (gamma + 1)
    v = gm1_over_gp1 * dvp0.v + two_over_gp1 * (x_over_t - sgn * c0)
    c = two_over_gp1 * c0 + sgn * gm1_over_gp1 * (x_over_t - dvp0.v)
    rho = dvp0.rho * math.pow(c / c0, 2 / (gamma - 1))
    p = dvp0.p * math.pow(rho / dvp0.rho, gamma)
    return state.DVPVec(rho=rho, v=v, p=p)

def riemann_solution(dvpL: state.DVPVec,
                     dvpR: state.DVPVec,
                     gamma: float) -> tuple[state.DVPVec,
                                            state.DVPVec,
                                            list[float],
                                            list[float]]:
    '''
    Computes exact Riemann solution.  Input:

    dvpL: left state.
    dvpR: right state.
    gamma: EOS gamma
    
    Returns:

    dvpLm: state to the left of the contact
    dvpRm: state to the right of the contact
    sL: List of wavespeed(s) for the left acoustic wave.  If a
        single value, then a shock. If two values,
        then an expansion fan, with the first value on the speed
        in the left state, and the second value the speed
        on the left side of the contact.
    sR: List of wavespeed(s) for the right acoustic wave.  If a
        single value, then a shock. If two values,
        then an expansion fan, with the first value on the speed
        in the right state, and the second value the speed
        on the right side of the contact.
    '''
    p = 0.5 * (dvpL.p + dvpR.p) # initial guess
    pmin = 1.0e-5 * min(dvpL.p, dvpR.p)
    error = 1
    while error > 1.0e-10:
#        print(p, dvpL.p, dvpR.p)
        if p > dvpL.p:
            vmL, dvL, rhomL, sL = v_shock(dvpL, p, gamma, -1)
        else:
            vmL, dvL, rhomL, sL = v_rare(dvpL, p, gamma, -1)
        if p > dvpR.p:
            vmR, dvR, rhomR, sR = v_shock(dvpR, p, gamma, 1)
        else:
            vmR, dvR, rhomR, sR = v_rare(dvpR, p, gamma, 1)
        error = abs(vmR - vmL)
#        print(error, p, vmL, vmR, dvL - dvR)
        p -= (vmL - vmR) / (dvL - dvR)
        p = max(pmin, p)
    v = 0.5 * (vmL + vmR)
    dvpLm = state.DVPVec(rho=rhomL, v=v, p=p)
    dvpRm = state.DVPVec(rho=rhomR, v=v, p=p)
    return dvpLm, dvpRm, sL, sR

def riemann_flux(uL: state.ConservativeVec,
                 uR: state.ConservativeVec,
                 eos: GammaEOS):
    '''
    Computes the exact Riemann flux; that is, the solution
    on the t-axis. Input:

    uL: left state.
    uR: right state.
    eos: eos object.

    Returns a FluxVec of the flux.
    '''
    dvpL = state.Conservative_to_DVP(uL, eos)
    dvpR = state.Conservative_to_DVP(uR, eos)
    dvpLm, dvpRm, sL, sR = riemann_solution(dvpL, dvpR, eos.gamma)
    # set solution based on wave location w.r.t. x=0 (t-axis)
    if dvpLm.v < 0: # contact to left
        if sR[-1] > 0: # inner characteristic to right
            dvp0 = dvpRm
        else: # inner characteristic to left
            if sR[0] < 0: # outer characteristic to left
                dvp0 = dvpR
            else: # outer characteristic to right; in fan
                dvp0 = rare_xt(0, eos.gamma, dvpR, 1)
    else: # contact to right
        if sL[-1] < 0: # inner characteristic to left
            dvp0 = dvpLm
        else: # inner characteristic to right
            if sL[0] > 0: # outer characteristic to right
                dvp0 = dvpL
            else: # outer characteristic to left; in fan
                dvp0 = rare_xt(0, eos.gamma, dvpL, -1)
    frho = dvp0.rho * dvp0.v
    frhov = frho * dvp0.v + dvp0.p
    e = eos.e_rho_p(dvp0.rho, dvp0.p)
    frhoE = frho * (e + 0.5 * dvp0.v * dvp0.v) + dvp0.v * dvp0.p
    f = state.FluxVec(rho=frho, rhov=frhov, rhoE=frhoE)
    return f

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    '''
    Tests.
    '''
    geos = GammaEOS(5/3)
    dvpL = state.DVPVec(rho=1, v=0, p=5)
    dvpR = state.DVPVec(rho=1, v=0, p=1)
    dvpLm, dvpRm, sL, sR = riemann_solution(dvpL, dvpR, geos.gamma)
    print(f'p {dvpLm.p} v {dvpLm.v} rhoL {dvpLm.rho} rhoR {dvpRm.rho} sL {sL} sR {sR}')
    uL = state.DVP_to_Conservative(dvpL, geos)
    uR = state.DVP_to_Conservative(dvpR, geos)
    f = riemann_flux(uL, uR, geos)
    print(f'flux: {f}')
    