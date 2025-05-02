# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
State vectors for trt.
'''

from typing import Optional
import numpy as np

from base.constants import radiation_constant
from base.my_types import ScalarOrArray
import base.state

class RadiationVec(base.state.BaseVec):
    '''
    Radiation state (diffusion).
    '''
    def __init__(self,
                 length=None,
                 rade: ScalarOrArray=1.0,
                 e: ScalarOrArray=1.0,
                 **kwargs):
        super().__init__(length, rade=rade, e=e, **kwargs)
    def create_default(self) -> 'RadiationVec':
        return RadiationVec()
    def set_tr(self,
               tr: ScalarOrArray) -> None:
        '''
        Sets the radiation energy based on the input radiation temperature.
        '''
        self.rade = radiation_constant * np.pow(tr, 4)
    def tr(self) -> ScalarOrArray:
        '''
        Returns the radiation temperature.
        '''
        return np.pow(self.rade / radiation_constant, 0.25)

if __name__ == '__main__':
    '''
    Tests.
    '''
    rvec = RadiationVec()
    print(rvec)
