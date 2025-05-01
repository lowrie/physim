# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for material properties:

Equation-of-state and radiation cross-sections.
'''

import abc
import numpy as np

from base.my_types import ScalarOrArray

class BaseEOS(abc.ABC):
    '''
    Base class for equations of state. We only support gamma-law EOS,
    but this interface helps emphasize which parts of the code work
    with this general interface and which are specific to gamma-law.
    '''
    def __init__(self, name: str):
        self.name = name
    @abc.abstractmethod
    def p_rho_e(self,
                rho: ScalarOrArray,
                e: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns pressure at rho, e.
        '''
        pass
    @abc.abstractmethod
    def e_rho_p(self,
                rho: ScalarOrArray,
                p: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns specific internal energy at rho, p.
        '''
        pass
    @abc.abstractmethod
    def c_rho_e(self,
                rho: ScalarOrArray,
                e: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns sound speed at rho, e.

        For a gamma EOS, rho is ignored.
        '''
        pass
    @abc.abstractmethod
    def c_rho_p(self,
                rho: ScalarOrArray,
                p: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns sound speed at rho, p.
        '''
        pass
    @abc.abstractmethod
    def T_rho_e(self,
                rho: ScalarOrArray,
                e: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns temperature at rho, e.
        '''
        pass
    @abc.abstractmethod
    def e_rho_T(self,
                rho: ScalarOrArray,
                T: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns specific interal energy at rho, T.
        '''
        pass
    @abc.abstractmethod
    def cv_rho_e(self,
                 rho: ScalarOrArray,
                 e: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns pressure at rho, e.
        '''
        pass

class GammaEOS(BaseEOS):
    '''
    Interface for a gamma-law EOS.
    '''
    def __init__(self, gamma=5/3.0, cv=1.0):
        '''
        gamma: Value of gamma, or ratio of specfic heats.
        cv: Specific heat at constant volume.
        '''
        super().__init__('gamma-law')
        self.gamma = gamma
        self.cv = cv
    def p_rho_e(self,
                rho: ScalarOrArray,
                e: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns pressure at rho, e.
        '''
        return (self.gamma - 1) * rho * e
    def e_rho_p(self,
                rho: ScalarOrArray,
                p: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns specific internal energy at rho, p.
        '''
        return p / ((self.gamma - 1) * rho)
    def c_rho_e(self,
                _rho: ScalarOrArray,
                e: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns sound speed at rho, e.

        For a gamma EOS, rho is ignored.
        '''
        return np.sqrt(self.gamma * (self.gamma - 1) * e)
    def c_rho_p(self,
                rho: ScalarOrArray,
                p: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns sound speed at rho, p.
        '''
        return np.sqrt(self.gamma * p / rho)
    def T_rho_e(self,
                _rho: ScalarOrArray,
                e: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns temperature at rho, e.

        For a gamma EOS, rho is ignored.
        '''
        return e / self.cv
    def e_rho_T(self,
                _rho: ScalarOrArray,
                T: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns specific interal energy at rho, T.

        For a gamma EOS, rho is ignored.
        '''
        return T * self.cv
    def cv_rho_e(self,
                 _rho: ScalarOrArray,
                 _e: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns pressure at rho, e.

        For a gamma EOS, rho and e are ignored.
        '''
        return self.cv

class XCConstant:
    '''
    Defines a constant cross section.
    '''
    def __init__(self, constant: float=1.0):
        '''
        constant: The constant value of the cross section.
        '''
        self.constant = constant
    def cross_section(self,
                      rho: ScalarOrArray,
                      T: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns the cross section evaluated at rho, T.
        '''
        # If rho or T are arrays, return an array of constants.
        # Otherwise, return a scalar value.
        length = 0
        for a in [rho,T]:
            if isinstance(a, np.ndarray):
                length = len(a)
        if length == 0:
            return self.constant
        else:
            return np.full(length, self.constant, dtype=float)

class XCPowerLaw(XCConstant):
    '''
    Defines a power-law cross-section of the form:

    xc = constant * rho^exprho * T^expT
    '''
    def __init__(self,
                 constant: float=1.0,
                 exprho: float=1.0,
                 expT: float=1.0):
        '''
        constant: Proportionality constant
        exprho: Mass density exponent
        expT: Material temperature exponent

        Note that if exprho=expT=0, it is more efficient to use XCConstant.
        '''
        super().__init__(constant)
        self.exprho = exprho
        self.expT = expT
    def cross_section(self,
                      rho: ScalarOrArray,
                      T: ScalarOrArray) -> ScalarOrArray:
        '''
        Returns the cross section evaluated at rho, T.
        '''
        return self.constant * np.pow(rho, self.exprho) * np.pow(T, self.expT)

class Material:
    '''
    Material properties container.
    '''
    def __init__(self,
                 eos: BaseEOS=GammaEOS(),
                 absorp: XCConstant=XCConstant(),
                 scat: XCConstant=XCConstant()):
        '''
        eos: EOS object.
        absorp: Absorption cross-section object.
        scat: Scattering cross-section object.
        '''
        self.eos = eos
        self.absorp = absorp
        self.scat = scat
    def absorption(self,
                   rho: ScalarOrArray,
                   T: ScalarOrArray) -> ScalarOrArray:
        return self.absorp.cross_section(rho, T)
    def scattering(self,
                   rho: ScalarOrArray,
                   T: ScalarOrArray) -> ScalarOrArray:
        return self.scat.cross_section(rho, T)
    def totalxc(self,
                rho: ScalarOrArray,
                T: ScalarOrArray) -> ScalarOrArray:
        return self.absorption(rho, T) + self.scattering(rho, T)

if __name__ == '__main__':
    '''
    Tests.
    '''
    eos = GammaEOS()
    absorp = XCPowerLaw(6.3, 2.0, 3.0)
    scat = XCConstant()
    mat = Material(eos, absorp, scat)
    print(mat.eos.gamma)
    print(mat.absorption(1.0, 10.0))
    print(mat.scattering(1.0, 10.0))
    print(mat.totalxc(1.0, 10.0))

