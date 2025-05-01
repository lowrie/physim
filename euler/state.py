# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
State vectors for euler.
'''
from typing import Optional
import numpy as np

from base.state import BaseVec
from base.material import BaseEOS, GammaEOS
from base.my_types import ScalarOrArray

class ConservativeVec(BaseVec):
    '''
    Vector of conserved quantities: rho, rho*v, rho*E
    '''
    def __init__(self,
                 length: Optional[int]=None,
                 rho: ScalarOrArray=1.0,
                 rhov: ScalarOrArray=0.0,
                 rhoE: ScalarOrArray=1.0,
                 **kwargs):
        super().__init__(length, rho=rho, rhov=rhov, rhoE=rhoE, **kwargs)
    def create_default(self) -> 'ConservativeVec':
        return ConservativeVec()
    def v(self, i: Optional[int]=None) -> ScalarOrArray:
        '''
        Return the velocity.
        '''
        if i is None:
            return self.rhov / self.rho
        else:
            return self.rhov[i] / self.rho[i]
    def e(self, i: Optional[int]=None) -> ScalarOrArray:
        '''
        Return the specific internal energy.
        '''
        if i is None:
            return (self.rhoE - 0.5 * self.rhov * self.rhov / self.rho) / self.rho
        else:
            return (self.rhoE[i] - 0.5 * self.rhov[i] * self.rhov[i] / self.rho[i]) / self.rho[i]
    def get_reflected(self) -> 'ConservativeVec':
        '''
        Return the reflected state (for reflection BCs)
        '''
        return ConservativeVec(rho=self.rho, rhov=-self.rhov, rhoE=self.rhoE)

class FluxVec(BaseVec):
    '''
    Flux vector: rho*v, rho*v^2+p, rho*v*H

    The components are labeled with their corresponding conservative variable,
    rho, rho*v, rho*E.
    '''
    def __init__(self,
                 length: Optional[int]=None,
                 rho=1.0,
                 rhov=0.0,
                 rhoE=1.0):
        super().__init__(length, rho=rho, rhov=rhov, rhoE=rhoE)
    def create_default(self) -> 'FluxVec':
        return FluxVec()

class DVPVec(BaseVec):
    '''
    Primitive vector: rho, v, p
    '''
    def __init__(self,
                 length: Optional[int]=None,
                 rho: ScalarOrArray=1.0,
                 v: ScalarOrArray=0.0,
                 p: ScalarOrArray=1.0):
        super().__init__(length, rho=rho, v=v, p=p)
    def create_default(self) -> 'DVPVec':
        return DVPVec()

class ConsIEVec(BaseVec):
    '''
    Conservative vector with internal energy: rho, rho*v, rho*e
    '''
    def __init__(self,
                 length: Optional[int]=None,
                 rho: ScalarOrArray=1.0,
                 rhov: ScalarOrArray=0.0,
                 rhoe: ScalarOrArray=1.0):
        super().__init__(length, rho=rho, rhov=rhov, rhoe=rhoe)
    def create_default(self) -> 'ConsIEVec':
        return ConsIEVec()

class CharVec(BaseVec):
    '''
    Charctertistic vector.  Components are the Riemann invariants.
    This capability needs work to be used as a limiter variable.
    It should be a linearization about a given state.
    '''
    def __init__(self,
                 length: Optional[int]=None,
                 vmc: ScalarOrArray=1.0,
                 s: ScalarOrArray=1.0,
                 vpc: ScalarOrArray=1.0):
        super().__init__(length, vmc=vmc, s=s, vpc=vpc)
    def create_default(self) -> 'CharVec':
        return CharVec()

def Conservative_to_DVP(c: ConservativeVec,
                        eos: BaseEOS) -> DVPVec:
    '''
    Returns a DVPVec corresponding to a given ConservativeVec.
    '''
    p = DVPVec(rho=c.rho, v=c.rhov, p=c.rhoE)
    p.v /= p.rho
    e = c.rhoE / p.rho - 0.5 * p.v * p.v
    p.p = eos.p_rho_e(p.rho, e)
    return p

def DVP_to_Conservative(p: DVPVec,
                        eos: BaseEOS) -> ConservativeVec:
    '''
    Returns a ConservativeVec corresponding to a given DPVVec.
    '''
    c = ConservativeVec(rho=p.rho, rhov=p.v, rhoE=p.p)
    c.rhov *= p.rho
    e = eos.e_rho_p(p.rho, p.p)
    c.rhoE = p.rho * e + 0.5 * p.v * c.rhov 
    return c

def Conservative_to_ConsIE(c: ConservativeVec,
                           _) -> ConsIEVec:
    '''
    Returns a ConsIEVec corresponding to a given ConservativeVec.
    '''
    cie = ConsIEVec(rho=c.rho, rhov=c.rhov, rhoe=c.rhoE)
    cie.rhoe -= 0.5 * cie.rhov * cie.rhov / cie.rho
    return cie

def ConsIE_to_Conservative(cie: ConsIEVec,
                            _) -> ConservativeVec:
    '''
    Returns a ConservativeVec corresponding to a given ConsIEVec.
    '''
    c = ConservativeVec(rho=cie.rho, rhov=cie.rhov, rhoE=cie.rhoe)
    c.rhoE += 0.5 * cie.rhov * cie.rhov / cie.rho
    return c

def Conservative_to_Char(c: ConservativeVec,
                         eos: GammaEOS) -> CharVec:
    '''
    Returns a CharVec corresponding to a given ConservativeVec.
    '''
    dvp = Conservative_to_DVP(c, eos)
    cs = 2 * eos.c_rho_p(dvp.rho, dvp.p) / (eos.gamma - 1)
    ch = CharVec(length=c.length())
    ch.vmc[:] = cs[:] - dvp.v
    ch.vpc[:] = cs[:] + dvp.v
    ch.s[:] = np.log(dvp.p / np.pow(dvp.rho, eos.gamma))
    #ch.s[:] = dvp.p / np.pow(dvp.rho, eos.gamma)
    return ch

def Char_to_Conservative(ch: CharVec,
                         eos: GammaEOS) -> ConservativeVec:
    '''
    Returns a ConservativeVec corresponding to a given CharVec.
    '''
    dvp = DVPVec(ch.length())
    dvp.v[:] = 0.5 * (ch.vpc - ch.vmc)
    cs = 0.25 * (eos.gamma - 1) * (ch.vpc + ch.vmc)
    exps = np.exp(ch.s)
#    exps = ch.s
    dvp.rho[:] = np.pow(cs * cs / (eos.gamma * exps), 1 / (eos.gamma - 1))
    dvp.p[:] = exps * np.pow(dvp.rho, eos.gamma)
    return DVP_to_Conservative(dvp, eos)

def DVP_to_Flux(p: DVPVec,
                eos: BaseEOS) -> FluxVec:
    '''
    Returns a FluxVec corresponding to a given DVPVec.
    '''
    frho = p.rho * p.v
    e = eos.e_rho_p(p.rho, p.p)
    frhov = frho * p.v + p.p
    frhoE = frho * (e + 0.5 * p.v * p.v) + p.v * p.p
    return FluxVec(rho=frho, rhov=frhov, rhoE=frhoE)

def Conservative_to_Flux(c: ConservativeVec,
                         eos: BaseEOS) -> ConservativeVec:
    '''
    Returns a FluxVec corresponding to a given ConservativeVec.
    '''
    dvp = Conservative_to_DVP(c, eos)
    return DVP_to_Flux(dvp, eos)

def ConsIE_to_Flux(cie: ConsIEVec,
                   eos: BaseEOS) -> FluxVec:
    '''
    Returns a FluxVec corresponding to a given ConsIEVec.
    '''
    c = ConsIE_to_Conservative(cie, eos)
    return Conservative_to_Flux(c, eos)

if __name__ == '__main__':
    '''
    Tests.
    '''
    from base.material import GammaEOS
    # scalar checks
    c = ConservativeVec()
    print(f'c {c}')
    gammaEOS = GammaEOS(5/3.)
    p2 = Conservative_to_DVP(c, gammaEOS)
    print(f'p2 {p2}')
    c = DVP_to_Conservative(p2, gammaEOS)
    print(f'c {c}')
    # array checks
    np.set_printoptions(threshold=100)
    c = ConservativeVec(10)
    print(f'array c {c}')
    p2 = Conservative_to_DVP(c, gammaEOS)
    print(f'array p2 {p2}')
    c = DVP_to_Conservative(p2, gammaEOS)
    print(c)
    c.rho[2] = 500
    print(c)
    cs = c.get_vec(5)
    print(cs)
    cs.rho = 100
    print(c)
    c.set_vec(5, cs)
    print(c)
    cs = c.get_vec(3, deepcopy=False)
    cs.rho = 111.0
    print(c)
