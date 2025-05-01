# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Boundary condition classes.
'''
import base.state as state

class BCBase:
    '''
    Base class for all boundary conditions
    '''
    def __init__(self, name: str):
        '''
        name: Identifier string
        '''
        self.name = name

class BCPeriodic(BCBase):
    '''
    Periodic boundary condition.
    '''
    def __init__(self):
        super().__init__('periodic')

class BCReflection(BCBase):
    '''
    Reflection boundary condition.
    '''
    def __init__(self):
        super().__init__('reflection')

class BCDirichlet(BCBase):
    '''
    Dirichlet boundary condition.

    Attribute u is the state of the BC.
    '''
    def __init__(self, u: state.BaseVec):
        super().__init__('dirichlet')
        self.u = u

class BC1D:
    '''
    Contains the boundary conditions for 1-D problems.
    '''
    def __init__(self, left=BCPeriodic(), right=BCPeriodic()):
        if (isinstance(left, BCPeriodic) and not isinstance(right, BCPeriodic)) or\
            (isinstance(right, BCPeriodic) and not isinstance(left, BCPeriodic)):
            raise ValueError('Both boundary conditions must be periodic if one is periodic.')
        self.left = left
        self.right = right

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    '''
    Tests.
    '''
    bcs = BC1D()
    print(f'{bcs.left.name} {bcs.right.name}')
