import matplotlib.pyplot as plt
import drawRobotics as dR
import numpy as np
from scipy.spatial.distance import cdist
from numpy import linalg
import math


class arm(object):
    def __init__(self, obs):
        th1Init, th2Init, th3Init, th4Init = 0.0, 0.0, 0.0, 0.0

        self.a2 = 1.5
        self.a3 = 0.5
        self.d3 = 1.5
        self.d4 = 1.0

        self.ORG_Base = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,-0.4], [0,0,0,1]])
        self.ORG_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])

        self.obs = obs


    def RotZ(self, a):
        return np.array( [[np.cos(a), -np.sin(a), 0, 0], 
                            [np.sin(a), np.cos(a), 0, 0], 
                            [0, 0, 1, 0],
                            [0, 0, 0, 1] ] )

    def RotX(self, a):
        return np.array( [[1, 0, 0, 0], 
                            [0, np.cos(a), -np.sin(a), 0],
                            [0, np.sin(a), np.cos(a), 0],
                            [0, 0, 0, 1] ] )

    def D_q(self, dq1,dq2,dq3):
        return np.array([[1,0,0,dq1],[0,1,0,dq2],[0,0,1,dq3],[0,0,0,1]])


    def calcORGs(self, q1, q2, q3, q4):
        th1 = dR.conv2Rad(q1)
        th2 = dR.conv2Rad(q2)
        th3 = dR.conv2Rad(q3)
        th4 = dR.conv2Rad(q4)

        Trans_0to1 = self.RotZ(th1)
        Trans_1to2 = np.dot(self.RotX(dR.conv2Rad(-90)), self.RotZ(th2))
        Trans_2to3 = np.dot(np.dot(self.D_q(self.a2,0,0), self.D_q(0,0,self.d3)), self.RotZ(th3))
        Trans_3to4 = np.dot(np.dot(np.dot(self.RotX(dR.conv2Rad(-90)), self.D_q(self.a3,0,0)), self.D_q(0,0,self.d4)), self.RotZ(th4))

        Trans_0to2 = np.dot(Trans_0to1, Trans_1to2)
        Trans_0to3 = np.dot(Trans_0to2, Trans_2to3)
        Trans_0to4 = np.dot(Trans_0to3, Trans_3to4)

        ORG_1 = np.dot(Trans_0to1, self.ORG_0)
        ORG_2 = np.dot(Trans_0to2, self.ORG_0)
        ORG_3 = np.dot(Trans_0to3, self.ORG_0)
        ORG_4 = np.dot(Trans_0to4, self.ORG_0)

        return ORG_1, ORG_2, ORG_3, ORG_4

    def render(self, ax, ORG_1, ORG_2, ORG_3, ORG_4):
    #    ax = fig.add_subplot(111, projection='3d')
        ax.cla()    
        ax.clear()
    #     ax.axis('off')

    #     dR.drawPointWithAxis(ax, ORG_0, lineStyle='--', vectorLength=1, lineWidth=2)
    #     dR.drawPointWithAxis(ax, ORG_1, vectorLength=0.5)
    #     dR.drawPointWithAxis(ax, ORG_2, vectorLength=0.5)
    #     dR.drawPointWithAxis(ax, ORG_3, vectorLength=0.5)
    #     dR.drawPointWithAxis(ax, ORG_4, vectorLength=0.5)

        dR.drawVector(ax, self.ORG_Base, self.ORG_0, arrowstyle='-', lineColor='c', proj=False, lineWidth=20)
        dR.drawVector(ax, self.ORG_0, ORG_1, arrowstyle='-', lineColor='k', proj=False, lineWidth=9)
        dR.drawVector(ax, ORG_1, ORG_2, arrowstyle='-', lineColor='k', proj=False, lineWidth=9)
        dR.drawVector(ax, ORG_2, ORG_3, arrowstyle='-', lineColor='k', proj=False, lineWidth=9)
        dR.drawVector(ax, ORG_3, ORG_4, arrowstyle='-', lineColor='k', proj=False, lineWidth=9)

    #     plt.plot(obs[0], obs[1], '.r')
    #     ax.scatter(obs[0], obs[1], c= 'r', s= 100)
        ax.scatter(self.obs[0], self.obs[1], self.obs[2], color = 'red', s = 100)
        ax.scatter(ORG_1[0, 3], ORG_1[1, 3], ORG_1[2, 3], c ='w', s=300)  
        ax.scatter(ORG_2[0, 3], ORG_2[1, 3], ORG_2[2, 3], c ='w', s=50)
        ax.scatter(ORG_3[0, 3], ORG_3[1, 3], ORG_3[2, 3], c ='w', s=50)
    #     ax.scatter(ORG_4[0, 3], ORG_4[1, 3], ORG_4[2, 3], c ='w', marker = '>', s=50)
            
        
        ax.scatter(ORG_1[0, 3], ORG_1[1, 3], ORG_1[2, 3], s = 30)  
        ax.scatter(ORG_2[0, 3], ORG_2[1, 3], ORG_2[2, 3],s = 10)
        ax.scatter(ORG_3[0, 3], ORG_3[1, 3], ORG_3[2, 3],s = 10)
    #     ax.scatter(ORG_4[0, 3], ORG_4[1, 3], ORG_4[2, 3],s = 10)

        ax.set_xlim([-2,3]), ax.set_ylim([-2,3]), ax.set_zlim([-2,2])
        ax.set_xlabel('X axis'), ax.set_ylabel('Y axis'), ax.set_zlabel('Z axis')
        ax.view_init(azim=10, elev=40)
        
    #    fig.show()
        plt.pause(0.001)     # in the case of  'qt'
    #    fig.canvas.draw()


    def step(self, val, ax, display_on):

        th1 = val[0]
        th2 = val[1]
        th3 = val[2]
        th4 = 0
        
    #     print(th1, th2, th3)
    #     np.where(th1 < 90, th1, 90)
    #     np.where(th1 > 0, th1, 0)

    #     np.where([th2,th3] < 60, [th2,th3], 60)
    #     np.where([th2,th3] > -60, [th2, th3], -60)
        validation = True

    #     if th1 > 90:
    #         th1 = 90
    # #        print('...invalid angle in th1...');
    #         validation = False
    #     elif th1 < -90:
    #         th1 = -90
    # #        print('...invalid angle in th1...');
    #         validation = False
    #     if th2 > 60:
    #         th2 = 60
    # #        print('...invalid angle in th2...');
    #         validation = False
    #     elif th2 < -60:
    #         th2 = -60
    # #        print('...invalid angle in th2...');
    #         validation = False
    #     if th3 > 60:
    #         th3 = 60
    # #        print('...invalid angle in th3...');
    #         validation = False
    #     elif th3 < -60:
    #         th3 = -60
    # #        print('...invalid angle in th3...');
    #         validation = False
        
        ORG_1, ORG_2, ORG_3, ORG_4 = self.calcORGs(th1, th2, th3, th4)
        end_effector = [ORG_4[0, 3], ORG_4[1, 3], ORG_4[2, 3]] 

        reward, done = self.evaluation(end_effector, validation)
        
        if display_on == True:
            self.render(ax, ORG_1, ORG_2, ORG_3, ORG_4)
        
        return reward, end_effector, done
    
    def evaluation(self, end_position, validation):
        # action = np.asmatrix(action)
        dis = np.sqrt((self.obs[0]-end_position[0])**2 + (self.obs[1]-end_position[1])**2 + (self.obs[2]-end_position[2])**2)
    #    alpha = 1
    #    beta = np.matrix([[0.15, 0, 0], [0, 0.1, 0], [0, 0, 0.1]])
    #    if validation == True:
    #        reward = -alpha*(dis**2) - action*beta*np.transpose(action)
    #    else:
    #        reward = -alpha*(dis**2) - action*beta*np.transpose(action) - 10
        
        # reward = -5*dis - 0.5*math.log10(dis) - 0.1*np.linalg.norm(action)
        reward = -1*dis #- 0.1*math.log10(dis)
        # reward = -1/(dis + 1)

        if dis < 0.2:
            done = True
            reward = 10
        else:
            done = False

        return reward, done

    def reset(self, ax, display_on):

        state = np.array([0, 0, 0])
        ORG_1, ORG_2, ORG_3, ORG_4 = self.calcORGs(0, 0, 0, 0)
        end_effector = [ORG_4[0, 3], ORG_4[1, 3], ORG_4[2, 3]] 

        if display_on == True:
            self.render(ax, ORG_1, ORG_2, ORG_3, ORG_4)
        
        return end_effector, state