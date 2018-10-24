
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt


# In[6]:


class CanonicalSystem():
    def __init__(self, runTime, dt, a_x=1.0, pattern='discrete'):
        self.runTime = runTime
        self.dt = dt
        self.pattern = pattern
        self.a_x = a_x
        
        if self.pattern == 'discrete':
            self.step = self.step_discrete
        elif self.pattern == 'rhythmic':
            self.step = self.step_rhythmic
        else:
            raise Exception("invalid pattern type: discrete or rythmic")
            
        self.n_step = int(self.runTime/self.dt)
        
        self.reset()
        
    def reset(self):
        self.x = 1.0
        
        return self.x
    
    def step_discrete(self, tau=1.0, error=0.0):
        """
        generate a single step of x for discrete loop movements
        by decaying from 1 to 0 according to dx = -a_x*x
        
        - tau float: gain on execution time
                     increase tau to make the system execute faster
        - error_coupling float: slow down if the error is > 1
        """
        error_coupling = 1.0/(1.0 + error)
        self.x += (-self.a_x*self.x*error_coupling)*tau*self.dt
        
        return self.x
    
    def step_rhythmic(self, tau=1.0, error=0.0):
        """
        generate a single step of x for rythmic closed loop movements
        by decaying from 1 to 0 according to dx = -a_x*x
        
        - tau float: gain on execution time
                     increase tau to make the system execute faster
        - error_coupling float: slow down if the error is > 1
        """
        error_coupling = 1.0/(1.0 + error)
        self.x += (1*error_coupling*tau)*self.dt
        
        return self.x  






