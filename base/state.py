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

    Vector data may be set and accessed several ways. See the tests below
    for examples.
    '''
    def __init__(self,
                 length: Optional[int]=None,
                 **kwargs):
        '''
        length: If an integer, then the array length of each component.
            If None, then see kwargs description.
        kwargs: Components and their initial values, in the form "name=value".
            Values may be my_types.ScalarOrArray.
            If value is an array and length is None, then the length is set by the array size.
            If an array and length is an int, then the array size must match length.
            Arrays are deep copied and all must be the same size.
            If a component is initialized as a float, then if length is an integer (either
            because it is specified or the array size of another initializer),
            the entire array is initialized to this value.
            If length is None and all components are initialized as a float,
            then the state vector is treated as a vector of floats.
            Class method names (e.g., length() and num_components()) cannot be
            component names. Also, avoid names that begin with an underscore.
        '''
        # Determine the final value of length (thus whether a vector of floats or
        # vector of arrays), and the list name component names.
        self.__dict__['_components'] = []
        self._length = length
        for key, value in kwargs.items():
            if key in dir(self):
                raise AttributeError(f'Unable to initialize component {key}; existing attribute name.')
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
                     a: ScalarOrArray=1.0,
                     b: ScalarOrArray=2.0,
                     **kwargs):
            super().__init__(a=a, b=b, **kwargs)
        # abstract method must be defined
        def create_default(self) -> 'Vec':
            return Vec()
    print('*** vector of float checks')
    print('\nInitialization: c = Vec()')
    c = Vec()
    print(f'c: {c}')

    print('\nSet c.a = 5.5 and c.b = 6.6:')
    c.a = 5.5
    c.b = 6.6
    print(f'c: {c}')

    print('\n*** vector of array checks')
    print('\nInitialization: c = Vec(length=10)')
    c = Vec(length=10)
    print(f'c: {c}')

    print('\nSet c.a = 5.5 and c.b = 6.6:')
    c.a = 5.5
    c.b = 6.6
    print(f'c: {c}')

    print('\nInitializing as: d = Vec(a=c.a)')
    print('length is taken from c.a and b=2.0 is the default initialization')
    d = Vec(a=c.a)
    print(f'd: {d}')

    print('\nSetting: d.a[3] = 7.3 and d.b[5] = 8.3:')
    d.a[3] = 7.3
    d.b[5] = 8.3
    print(f'd: {d}')

    print('\nDeep copy, so c.a is unchanged')
    print(f'c.a: {c.a}')

    print('\nSetting: d[0,6] = 9.4 and d[1,7] = 9.5:')
    print('Equivalent is d.a[6] = 9.4 and d.b[7] = 9.5')
    d[0,6] = 9.4 # same as d.a[6] = 9.4
    d[1,7] = 9.5 # same as d.b[7] = 9.5
    print(f'd: {d}')

    # Inheritence
    print('\n*** Inheritence IVec(Vec)')
    class IVec(Vec):
        def __init__(self,
                     c: ScalarOrArray=3.0,
                     **kwargs):
            super().__init__(c=c, **kwargs)
        def create_default(self) -> 'IVec':
            return IVec()
    print('\nInitialization: e = IVec(length=5)')
    e = IVec(length=5)
    print(f'e: {e}')

    print('\nSetting: e[0,1] = 1.1 and e[1,2] = 1.2 e[2,3] = 1.3')
    print('Equivalent is e.a[1] = 1.1, e.b[2] = 1.2 and e.c[3] = 1.3')
    e[0,1] = 1.1
    e[1,2] = 1.2
    e[2,3] = 1.3
    print(f'e: {e}')

    # Inheritence, override component name
    class OverrideVec(Vec):
        def __init__(self,
                     a: ScalarOrArray=66.0, # overrides default in Vec()
                     **kwargs):
            super().__init__(a=a, **kwargs)
        def create_default(self) -> 'OverrideVec':
            return IVec()

    print('\n*** Inheritence with component-name override.')
    print('\nOverrideVec(), with a=66.0 as the default.')
    r = OverrideVec()
    print(f'r: {r}')

    # Inheritence, override component name
    class OverrideVec(Vec):
        def __init__(self,
                     a: ScalarOrArray=66.0, # overrides default in Vec()
                     **kwargs):
            super().__init__(a=a, **kwargs)
        def create_default(self) -> 'OverrideVec':
            return IVec()

    print('\n*** Inheritence with component-name override.')
    print('\nOverrideVec(), with a=66.0 as the default.')
    r = OverrideVec()
    print(f'r: {r}')

    # Bad component name
    class VecBad(BaseVec):
        def __init__(self,
                     a: ScalarOrArray=30.0,
                     get_vec: ScalarOrArray = 2.0, # this is a method name; not permitted
                     **kwargs):
            super().__init__(a=a, get_vec=get_vec, **kwargs)
        def create_default(self) -> 'VecBad':
            return VecBad()

    print('\n*** Creating a object with an attribute (component name) error.')
    try:
        bad = VecBad()
    except AttributeError as msg:
        print(f'Caught AttributeError: {msg}')

