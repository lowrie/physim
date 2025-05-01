# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for setting initial conditions.
'''
import abc

import base.state as state

class ICBase(abc.ABC):
    '''
    Base class for all initial conditions.
    '''
    def __init__(self, name: str):
        self.name = name
    @abc.abstractmethod
    def get_ic(self) -> state.BaseVec:
        '''
        Returns numcells-array of ConservativeVec
        '''
        pass

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    '''
    Tests.
    '''
