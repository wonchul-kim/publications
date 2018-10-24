#!/usr/bin/env python
#title echo.py

import rospy
import roslib
from control_msgs.msg import JointTrajectoryControllerState

def callback(data):
    print(data.actual.positions)


def listener():
    rospy.init_node('listener')
    rospy.Subscriber('arm_controller/state', JointTrajectoryControllerState, callback)
    rospy.spin()

listener()
