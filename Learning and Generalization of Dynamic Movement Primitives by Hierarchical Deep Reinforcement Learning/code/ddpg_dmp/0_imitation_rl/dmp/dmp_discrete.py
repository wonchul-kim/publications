from cs import CanonicalSystem
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

class DMP_discrete(object):
    def __init__(self, n_dmps, dt, runTime, y_0=0, goal=1,
                 a_y=None, b_y=None, **kwargs):
        """
        n_dmps int: number of dynamic motor primitives
        dt float: timestep for simulation
        y0 list: initial state of DMPs
        goal list: goal state of DMPs
        ay int: gain on attractor term y dynamics
        by int: gain on attractor term y dynamics
        """
        
        self.n_dmps = n_dmps
        self.dt = dt
        self.runTime = runTime
        
        if isinstance(y_0, (int, float)):
            y_0 = np.ones(self.n_dmps)*y_0
        self.y_0 = y_0
        
        if isinstance(goal, (int, float)):
            goal = np.ones(self.n_dmps)*goal
        self.goal = goal
        
        # from the paper
        self.a_y = np.ones(n_dmps)*25.0 if a_y is None else a_y
        self.b_y = self.a_y/4.0 if b_y is None else b_y
        
        self.cs = CanonicalSystem(dt=self.dt, runTime=self.runTime, **kwargs)
        self.n_step = self.cs.n_step
        
        self.check_offset()


        self.reset()
        
    
    def reset(self):
        self.y = self.y_0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset()
        
    def check_offset(self):
        """
        check to see if initial position and goal are the same 
        if they are, offset slightly so that the forcing term is not 0
        """
        for i in range(self.n_dmps):
            if self.y_0[i] == self.goal[i]:
                self.goal[i] += 1e-4

    def f_front_term(self, x, dmp_idx):
        """
        diminishing term for forcing function
        """
        return x*(self.goal[dmp_idx] - self.y_0[dmp_idx])

    def gen_goal(self, y_des):
        """Generate the goal for path imitation.
        For rhythmic DMPs the goal is the average of the
        desired trajectory.

        y_des np.array: the desired trajectory to follow
        """

        return np.copy(y_des[:, -1])
        
    def step(self, f_target, tau=1.0, error=0.0, external_force=None):
        """
        generate a single timestep of dmp

        - tau float: scales the timestep
                     increase tau to make the system execute faster
        - error float: optional system feedback
        """
        x = self.cs.step(tau=tau, error=error)

        for i in range(self.n_dmps):
            f = f_target

            # self.ddy[i] = (self.a_y[i]*
            #                   (self.b_y[i]*(self.goal[i] - self.y[i])
            #                       - self.dy[i]/tau)+ f[i])*self.f_front_term(x, i)
            self.ddy[i] = (self.a_y[i]*
                              (self.b_y[i]*(self.goal[i] - self.y[i])
                                  - self.dy[i]/tau)+ f[i])
            
            if external_force is not None:
                self.ddy[i] += external_force[i]
            
            error_coupling = 1/(1 + error)
            self.dy[i] += self.ddy[i]*tau*self.dt*error_coupling
            self.y[i] += self.dy[i]*self.dt*error_coupling
            
        return self.y, self.dy, self.ddy
    
    def imitate_path(self, y_des, plot=False):
        # set initiall state and goal
        if y_des.ndim == 1:
            y_des = y_des.reshape(1, len(y_des))
        self.y_0 = y_des[:, 0].copy() 
        self.y_des = y_des.copy()
        self.goal = self.gen_goal(y_des)
        self.check_offset()

        # to make y_des array as [time, value]
        path = np.zeros((self.n_dmps, self.n_step))
        t = np.linspace(0, self.cs.runTime, y_des.shape[1])
        for i in range(self.n_dmps):
            gen_path = scipy.interpolate.interp1d(t, y_des[i])
            for j in range(self.n_step):
                path[i, j] = gen_path(j*self.dt)
        y_des = path
        
        dy_des = np.diff(y_des)/self.dt # velocity
        # add zero to the beginning of every row
        dy_des = np.hstack((np.zeros((self.n_dmps, 1)), dy_des))

        ddy_des = np.diff(dy_des)/self.dt
        ddy_des = np.hstack((np.zeros((self.n_dmps, 1)), ddy_des))

        f_target = np.zeros((y_des.shape[1], self.n_dmps))

        x_traj = np.zeros(self.cs.n_step)
        for i in range(self.cs.n_step):
            x_traj[i] = self.cs.step()

        # find the force required to move along the desired trajectory
        for k in range(self.n_dmps):
            for idx in range(self.n_step):
                # f_target[idx, k] = (ddy_des[k, idx] - self.a_y[k]*
                #               (self.b_y[k]*(self.goal[k] - y_des[k, idx]) - dy_des[k, idx]))\
                #               /(self.f_front_term(x_traj[idx], k))
                f_target[idx, k] = (ddy_des[k, idx] - self.a_y[k]*
                            (self.b_y[k]*(self.goal[k] - y_des[k, idx]) - dy_des[k, idx]))\
                
        self.reset()

        return y_des, dy_des, ddy_des, f_target, x_traj

# >> W/ DDPG ################################################################  

    def ddpg_step(self, f, path, tau=1.0, external_force=None, error=0.0):
        x = self.cs.step(tau=tau, error=0.0)

        for i in range(self.n_dmps):
            # self.ddy[i] = (self.a_y[i]*
            #                   (self.b_y[i]*(self.goal[i] - self.y[i])
            #                       - self.dy[i]/tau) + f*self.f_front_term(x, i))*tau**2
            self.ddy[i] = (self.a_y[i]*
                              (self.b_y[i]*(self.goal[i] - self.y[i])
                                  - self.dy[i]/tau) + f)*tau**2
                  
            if external_force is not None:
                self.ddy[i] += external_force[i]
            
            error_coupling = 1/(1 + error)
            self.dy[i] += self.ddy[i]*tau*self.dt*error_coupling
            self.y[i] += self.dy[i]*self.dt*error_coupling

        return self.y, self.dy, self.ddy

    # def gen_reward(self, y, dy, path):
    #     term1 = (y[-1] - path[-1])**2
    #     term2 = sum(dy**2)
    #     term3 = 0

    #     return term1, term2, term3

    # def check_done(self, y, path):
    #     if (y[-1] - path[-1])**2 < 1e-5:
    #         return 1
    #     else:
    #         return 0