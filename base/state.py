# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.

import abc
from typing import Optional
import numpy as np
import numbers

class BaseVec(abc.ABC):
    '''
    Base class for state-vector classes. A state vector has number of components.
    Each component may be a single float, which we refer to as a vector of floats.
    Alternatively, each component may be an array of floats, which we refer to as
    a vector of arrays.
    '''
    def __init__(self,
                 length: Optional[int]=None,
                 **kwargs):
        '''
        length: If an integer, then the array length of each component.
            If None, then see kwargs description.
        kwargs: Components and their initial values. Values may be my_types.ScalarOrArray.
            If an array, then the array size must match length, or length must be set to None.
            If a scalar, then if length is an integer, the array is initialized to this value.
        '''
        # Determine the final value of length (thus whether a vector of floats or
        # vector of arrays), and the list name component names.
        self.__dict__['_components'] = []
        self._length = length
        for key, value in kwargs.items():
            self._components.append(key)
            if isinstance(value, np.ndarray):
                if not np.issubdtype(value.dtype, np.number):
                    raise ValueError(f'Numpy arrays must have a numeric data type; {key} is {value.dtype}')
                lv = len(value)
                if self._length is None:
                    self._length = lv
                elif self._length != lv:
                    raise ValueError(f'All arrays must be the same length; found lengths {lv} and {self._length}')
            elif not isinstance(value, numbers.Number):
                raise ValueError(f'Values must be a numpy array or numeric; {key} is {type(value)}')
        num_components = len(kwargs)
        # Store all the component data in a single array.
        if self._length is None:
            # Each component is a float, so store the data in a 1-D array
            self._data = np.zeros(num_components)
        else:
            # Otherwise, store as a 2-D array.
            self._data = np.zeros((num_components, self._length))
        cindex = 0 # component index in _data
        for key, value in kwargs.items():
            if self.length() is None:
                setattr(self, key, value)
            else:
                self._data[cindex,:] = value
            cindex += 1
    def length(self) -> Optional[int]:
        '''
        Return the array length of each component, or None
        if each component is a float.
        '''
        return self._length
    def num_components(self) -> int:
        '''
        Returns the number of components.
        '''
        return len(self._components)
    def components(self) -> list[str]:
        '''
        Returns a list of the component names.
        '''
        return self._components
    def __repr__(self) -> str:
        '''
        Returns a string reprepentation.
        '''
        rep = ''
        for comp in self._components:
            if len(rep) > 0:
                rep += ', '
            rep += f'{comp} = {getattr(self, comp)}'
        return rep
    @abc.abstractmethod
    def create_default(self) -> 'BaseVec':
        '''
        Return the default vector, with each component a float.
        '''
        pass
    def get_vec(self, index: int, deepcopy: bool=True) -> 'BaseVec':
        '''
        Returns a vector of floats at an index of a vector of arrays.
        This function is appropriate only for a vector of arrays.
        If deepcopy is True, the returned object is a deep copy of the
        values.  If False, then a reference is returned, so that changes
        to returned vector of floats also changes the values in the 
        vector of arrays.
        '''
        if self.length() is None:
            raise ValueError(f'get_vec() may only be used with a vector of arrays.')
        if not isinstance(index, int):
            raise ValueError(f'get_vec() index must be type int')
        d = self.create_default()
        if deepcopy:
            for i in range(self.num_components()):
                d._data[i] = self._data[i, index]
        else:
            d._data = self._data[:, index]
        return d
    def set_vec(self, index: int, v: 'BaseVec') -> None:
        '''
        Sets the components at index for a vector of arrays to the
        corresponding values in a vector of floats.
        This function is appropriate only for a vector of arrays.
        '''
        if self.length() is None:
            raise ValueError(f'set_vec() may only be used with a vector of arrays.')
        if v.length() is not None:
            raise ValueError(f'set_vec() may only set to a vector of floats.')
        if not isinstance(index, int):
            raise ValueError(f'get_vec() index must be type int')
        for i in range(self.num_components()):
            self._data[i, index] = v._data[i]
    def _check_index(self, index) -> None:
        '''
        Checks whether an index, used with [], is valid.
        '''
        if self.length() is None:
            if not (isinstance(index, int) or isinstance(index, slice)):
                raise ValueError(f'Invalid index type for vector of floats: {type(index)}')
        else:
            if isinstance(index, tuple):
                if len(index) != 2:
                    raise ValueError(f'Index tuple must be length 2')
            else:
                raise ValueError(f'Index must be a tuple of length 2')
    def __getitem__(self, index: int) -> float:
        '''
        Returns a float at the array index.
        '''
        self._check_index(index)
        return self._data[index]
    def __setitem__(self, index: int, v: float):
        '''
        Sets the data at index to v.
        '''
        self._check_index(index)
        self._data[index] = v
    def __getattr__(self, name: str):
        '''
        Handles getting attributes from self._data
        '''
        try:
            # Try getting attributes that are components.
            cindex = self.__dict__['_components'].index(name)
            if self.length() is None:
                return self.__dict__['_data'][cindex]
            else:
                return self.__dict__['_data'][cindex,:]
        except ValueError:
            # Must be some other attribute, so just return
            # it as the default __getattr__ behavior.
            return self.__dict__.get(name)
    def __setattr__(self, name, value):
        '''
        Handles setting attributes whose data is stored
        in self._data
        '''
        try:
            # Try setting attributes that are components.
            cindex = self.__dict__['_components'].index(name)
            if self.length() is None:
                self.__dict__['_data'][cindex] = value
            else:
                self.__dict__['_data'][cindex,:] = value
        except ValueError:
            # Must be some other attribute, so just
            # do the default __setattr__ behavior.
            self.__dict__[name] = value


if __name__ == '__main__':
    '''
    Tests.
    '''
    from my_types import ScalarOrArray
    class Vec(BaseVec):
        def __init__(self,
                     length: Optional[int]=None,
                     a: ScalarOrArray=1.0,
                     b: ScalarOrArray=2.0,
                     **kwargs):
            super().__init__(length, a=a, b=b, **kwargs)
        def create_default(self) -> 'Vec':
            return Vec()
    # scalar checks
    c = Vec()
    print(f'Scalar c {c}')
    c.a = 5.5
    c.b = 6.6
    print(f'Scalar c {c}')
    # array checks
    c = Vec(10)
    print(f'Array c {c}')
    c.a = 5.5
    c.b = 6.6
    print(f'Array c {c}')
    d = Vec(a=c.a)
    print(f'Array d {d}')
    d.a[3] = 200
    d.b[5] = 300
    print(f'Array d {d}')
    d[0,0] = 1000 # same as d.a[0] = 1000
    d[1,0] = 2000 # same as d.b[0] = 2000
    print(f'Array d {d}')

