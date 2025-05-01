# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Mesh classes
'''

class Mesh1D_Equally_Spaced:
    '''
    Data for a 1-D, equally-spaced mesh.
    '''
    def __init__(self, numcells: int, xLeft: float=0.0, xRight: float=1.0):
        '''
        numcells: Number of mesh cells.
        xLeft: The minimum x-coordinate.
        xRight: The maximum x-coordinate.
        '''
        self._numcells = numcells
        self._xLeft = xLeft
        self._xRight = xRight
        self._dx = (xRight - xLeft) / numcells
    def numcells(self) -> int:
        '''
        Returns number of cells.
        '''
        return self._numcells
    def numedges(self) -> int:
        '''
        Returns number of edges (or faces).
        '''
        return self._numcells + 1
    def xcell(self, index) -> float:
        '''
        Returns the cell-centered location at index,
        where 0 <= index < numcells().
        '''
        return self._xLeft + (index + 0.5) * self._dx
    def xedge(self, index) -> float:
        '''
        Returns the edge (or face) location at index,
        where 0 <= index < numedges()
        '''
        return self._xLeft + index * self._dx
    def dxcell(self, _index=0) -> float:
        '''
        Returns number of cells.
        '''
        return self._dx

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    '''
    Tests.
    '''
    mesh = Mesh1D_Equally_Spaced(10)
    print(f'{mesh.numcells()} {mesh.dxcell()} {mesh.xedge(0)} {mesh.xedge(mesh.numcells())}')
