from __future__ import print_function
import numpy as np
import random
import matplotlib.pyplot as plt

class imitationLearning(object):

    def __init__(self, dmp, path, agent, imitation_batch_size=50, imitation_training_epochs = 2000, 
                       imitation_display_step = 100, imitation_model_path = './imitation_model/', imitation_figs_path='./results/imitation_figs/'):
        print('\n>>> 0. Initialize imitationLearning .....')
        self.dmp = dmp
        self.path = path
        self.imitation_batch_size = imitation_batch_size
        self.imitation_training_epochs = imitation_training_epochs
        self.imitation_display_step = imitation_display_step
        self.imitation_model_path = imitation_model_path
        self.agent = agent
        self.imitation_figs_path = imitation_figs_path

    def get_data(self):
        print("\n>>> 1. Get desired forcing term from demonstraion .....")
        y_des_traj = np.zeros((self.dmp.n_step, self.dmp.n_dmps))
        dy_des_traj = np.zeros((self.dmp.n_step, self.dmp.n_dmps))
        ddy_des_traj = np.zeros((self.dmp.n_step, self.dmp.n_dmps))

        y_des_traj, dy_des_traj, ddy_des_traj, f_target, clock_signal \
                            = self.dmp.imitate_path(y_des=np.array(self.path), plot=False)

        # Normalize data 
        self.data_X = clock_signal.reshape([clock_signal.shape[0], 1])
        self.data_Y = f_target

        self.n_samples = self.data_X.shape[0]

        self.mean_X = np.mean(self.data_X)
        self.std_X = np.std(self.data_X)
        self.mean_Y = np.mean(self.data_Y)
        self.std_Y = np.std(self.data_Y)

        self.train_X = (self.data_X - self.mean_X)/self.std_X
        self.train_Y = (self.data_Y - self.mean_Y)/self.std_Y

        return self.data_X, self.train_X, self.mean_X, self.std_X, \
               self.data_Y, self.train_Y, self.mean_Y, self.std_Y 


    def run(self):
        print('\n>>> 2. Model learning starts .....')    
        for epoch in range(self.imitation_training_epochs):
            for idx in range(self.train_X.shape[0]/self.imitation_batch_size):
                x = np.reshape(self.train_X[self.imitation_batch_size*idx:(idx+1)*self.imitation_batch_size], [self.imitation_batch_size, 1])
                y = np.reshape(self.train_Y[self.imitation_batch_size*idx:(idx+1)*self.imitation_batch_size], [self.imitation_batch_size, 1])
                imitation_cost = self.agent.imitation_train(x, y)
                        
        # Display logs per epoch step
            if (epoch+1) % self.imitation_display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(imitation_cost))

        print("     Optimization Finished!")
        training_cost = self.agent.imitation_train(self.train_X, self.train_Y)
        print("     Training cost=", training_cost, '\n')    
            
        for _ in range(10):
            idx = random.randrange(0, self.train_X.shape[0] - 1)
            action, _ = self.agent.pi(self.train_X[idx], apply_noise=False, compute_Q=False)
            print(action, self.train_Y[idx])
        print('============================================================================\n')    
                
        for _ in range(10):
            idx = random.randrange(0, self.train_X.shape[0] - 1)
            action, _ = self.agent.pi(self.train_X[idx], apply_noise=False, compute_Q=False)
            print(action*self.std_Y + self.mean_Y, self.data_Y[idx])
        print('============================================================================\n')
        
        
        results_f = []
        for idx in range(self.train_X.shape[0]):
            action, _ = self.agent.pi(self.train_X[idx], apply_noise=False, compute_Q=False)
            results_f.append(action*self.std_Y + self.mean_Y)
            
        self.agent.imitation_save_model(self.imitation_model_path)
        print('\n>>> 3. Saved the actor model ..................\n')

        return results_f

    def eval(self, results_f, plot=False):           
        self.dmp.reset()
        y_traj = np.zeros((self.dmp.n_step, self.dmp.n_dmps))
        dy_traj = np.zeros((self.dmp.n_step, self.dmp.n_dmps))
        ddy_traj = np.zeros((self.dmp.n_step, self.dmp.n_dmps))

        # when scale is multiplied by three times
        # self.dmp.goal[0] = 2

        for i in range(self.dmp.n_step):
            y_traj[i, :], dy_traj[i, :], ddy_traj[i, :] \
                                                = self.dmp.step(f_target=results_f[i])

        if plot is True:                
            plt.figure()
            plt.plot(y_traj[:, 0], lw=2)
            plt.plot(self.path, 'r--', lw=2)
            plt.title('imitation learning result')
            plt.xlabel('time')
            plt.ylabel('system trajectory')
            plt.legend(['DPM', 'Demonstraions'])
            plt.savefig(self.imitation_figs_path + 'ImitationLearning.png')
            plt.tight_layout()
            plt.show()

            plt.figure()
            plt.plot(results_f, lw=2)
            plt.title('forcing term')
            plt.xlabel('time')
            plt.ylabel('f trajectory')
            plt.savefig(self.imitation_figs_path + 'forcingTerm.png')
            plt.tight_layout()
            plt.show()


    def restore_model(self):
        print('\n>>> 3. Restored the model ..... \n')
        self.agent.imitation_restore_model(self.imitation_model_path)
