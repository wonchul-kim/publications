#!/usr/bin/env python

import actionlib
from actionlib import SimpleActionClient, SimpleGoalState
import rospy
from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from gazebo_msgs.msg import LinkStates
from time import sleep 
import numpy as np
import math

ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

INITIAL_STATES = [0, -3.1415*(1./3), 3.1415*(2./3), 
                  -3.1415*(1./3), 3.1415*(1./2), 0]

class Arm:
    def __init__(self, arm_name, target):
        self.name = arm_name
        self.target = target
        self.jta = actionlib.SimpleActionClient('arm_controller/follow_joint_trajectory',
                                        FollowJointTrajectoryAction)
        rospy.loginfo('Waiting for joint trajectory action of %s' % self.name)
        self.jta.wait_for_server()
        rospy.loginfo('Found joint trajectory action!!')
        self.reset_env()

    def move1(self, angles, dt):
        """ Creates an action from the trajectory and sends it to the server"""
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ARM_JOINTS

        point = JointTrajectoryPoint()
        point.positions = angles
        point.time_from_start = rospy.Duration(dt)
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)
        # self.jta.send_goal(goal)

    def move2(self, angles, dt):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = ARM_JOINTS

        point = JointTrajectoryPoint()
        global states
        for x, y in zip(states, angles):
            point.positions.append(x + y)
        # a = states[0] + angles[0]
        # b = states[1] + angles[1]
        # c = states[2] + angles[2]
        # d = states[3] + angles[3]
        # e = states[4] + angles[4]
        # f = states[5] + angles[5]
        
        point.positions = [a, b, c, d, e, f] 
        point.time_from_start = rospy.Duration(dt)
        goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(goal)

    def is_move_done(self):

        return self.jta.get_state()

    def reset_env(self):
        print("Reset Environment >>>>>>>>>>>>>>>>>>>>")
        global episode, reward_per_ep, step, done, reward
        episode += 1
        reward_per_ep = 0
        step = 0
        done = False
        reward = 0

        self.move1(INITIAL_STATES, 1)
        while(self.is_move_done() == 0):
            rospy.spinOnce()
            sleep(0.05)

        print("<<<<<<<<<<<<<<<<<<< Completed environment reset")

    def step(self, action, dt):
        move2(action, dt)

        global endPosition
        end_position = endPosition
        reward, done = self.get_reward(end_effetor, action)

        return end_position, reward, done

    def get_reward(self, end_position, action):
        distance = np.sqrt((self.target[0]-end_position[0])**2 + (self.target[1]-end_position[1])**2
                            + (self.target[2]-end_position[2])**2)

        reward = -1*distance - 0.1*np.linalg.norm(action)

        if distance < 0.1:
            done = True
            reward = 10
        else:
            done = False

        return reward, done

# =============================================================================
# ====================== CALLBACK FUNC. =======================================
def arm_state_callback(data):
    global states
    states = data.actual.positions
    # print(states)

def link_state_callback(data):
    global endPosition
    endPosition = data.pose[7].position
    # print(endPosition)