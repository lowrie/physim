# Copyright 2025 Robert B. Lowrie
# This software is covered by the MIT License.
# Please see the provided LICENSE file for more details.
'''
Tools for stepping a physics package through time.
'''
import abc
import os
import sys
from typing import Optional
from io import TextIOWrapper

import base.state as state

class Logger:
    '''
    This class allows for stderr messages to go to both the terminal and a log file.
    The intent is to set sys.stderr to an instance of this class.
    '''
    def __init__(self,
                 log_file: Optional[TextIOWrapper]=None):
        '''
        log_file: An output object, typically created by open()
            If None, output to stderr will behave normally.
            If an output object, stderr meassages will also be
            written to log_file.  You can set log_file after this
            object is created.
        '''
        self.terminal = sys.__stderr__
        self.log_file = log_file

    def write(self, message: str):
        '''
        Write message to the terminal and the log file.
        '''
        self.terminal.write(message)
        if self.log_file is not None:
            self.log_file.write(message)
        # if performance is an issue, you might want to delete this line
        # and require explict calls to flush().
        self.flush()

    def flush(self):
        '''
        Flush the output buffers
        '''
        self.terminal.flush()
        if self.log_file is not None:
            self.log_file.flush()

class Case:
    '''
    Base class for a case. A case contains input parameters for a problem.
    '''
    def __init__(self,
                 output: str='output',
                 tmax: float=1.0,
                 rampdt_cycle: int=1,
                 maxcycle: int=sys.maxsize):
        '''
        output: Prefix for output files.
        tmax: Final time value.
        rampdt_cycle: Ramp up the time step over this many cycles.
        maxcycle: Maximum number of time steps.
        '''
        self.output = output
        self.tmax = tmax
        self.rampdt_cycle = rampdt_cycle
        self.maxcycle = maxcycle
    def __repr__(self):
        rep = ''
        for comp in vars(self):
            if len(rep) > 0:
                rep += ', '
            rep += f'{comp} = {getattr(self, comp)}'
        return rep

class Physics(abc.ABC):
    '''
    Base class for physics classes. This class defines the
    interface for physics to be integrated in time.
    '''
    def __init__(self,
                 case: Case,
                 logger: Logger):
        '''
        case: A Case object.
        logger: A Logger object.
        '''
        self.case = case
        outfile = case.output.format(**vars(case)) + '.out'
        self.output = open(outfile, 'w')
        logger.log_file = self.output

        # Don't open the plot file until a call to output_final()
        # so that the file is not created if the case fails.
        self.plotfile = case.output.format(**vars(case)) + '.plt'
        if os.path.exists(self.plotfile):
            raise FileExistsError(f'File {self.plotfile} already exists!')
    def get_max_dt(self, u: state.BaseVec) -> float:
        '''
        Return the maximum allowable time step.
        '''
        return 1.0
    @abc.abstractmethod
    def update(self, u: state.BaseVec, dt: float) -> None:
        '''
        Update the physics.

        dt: Time step to take.
        '''
        pass
    def output_cycle(self,
                     u: state.BaseVec,
                     cycle: int,
                     t: float) -> None:
        '''
        Output that is sent to the output file
        each cycle.

        u: Current solution
        cycle: Current cycle number
        t: Current time value.
        '''
        self.output.write(f'{cycle} {t}\n')
    def output_final(self, u: state.BaseVec) -> None:
        '''
        Final output at the maximum time.
        '''
        pass

def stepper(physics:Physics,
            u: state.BaseVec) -> None:
    '''
    Steps physics through time, from time 0 to the final time.

    physics: A Physics object.
    u: The initial condition of the solution.
    '''
    case = physics.case
    t = 0
    cycle = 0
    done = False # use to stop exactly at tmax
    physics.output_cycle(u, cycle, t)
    while t < case.tmax and cycle < case.maxcycle and not done:
        dt = physics.get_max_dt(u)
        if cycle < case.rampdt_cycle:
            dt *= (cycle + 1) / case.rampdt_cycle
        if t + dt > case.tmax:
            dt = case.tmax - t
            done = True
        physics.update(u, dt)
        t += dt
        cycle += 1
        physics.output_cycle(u, cycle, t)
    physics.output_final(u)

#######################################################################################################################
# Main
#######################################################################################################################

if __name__ == '__main__':
    '''
    Do a test.
    '''
    from base.my_types import ScalarOrArray
    class Vec(state.BaseVec):
        def __init__(self,
                     length: Optional[int]=None,
                     a: ScalarOrArray=1.0,
                     b: ScalarOrArray=2.0,
                     **kwargs):
            super().__init__(length, a=a, b=b, **kwargs)
        def create_default(self) -> 'Vec':
            return Vec()
    class MyPhysics(Physics):
        def __init__(self, case, logger):
            super().__init__(case, logger)
        # Must create update()
        def update(self, u: Vec, dt: float):
            u.a += dt * u.b
            u.b += dt * u.a
    logger = Logger()
    sys.stderr = logger
    case = Case('physics_test{rampdt_cycle}', 20.0)
    physics = MyPhysics(case, logger)
    u = Vec(length=10)
    stepper(physics, u)
    print(f'Solution complete.  Output writte to {physics.output.name}')
    raise ValueError(f'Test of raising an error: this message should be written to output file {physics.output.name}.')
