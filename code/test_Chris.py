from kinematic import *
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *

import importlib
import sys
import time
import numpy as np
import yarp
yarp.Network.init()

import sys
import os

iCubMotor = importlib.import_module(params.iCub_joint_names)

myT = fSetTiming()

# Create list of CPG objects
myCont = fnewMLMPcpg(params.number_cpg)
# Instantiate the CPG list with iCub robot data
myCont = fSetCPGNet(myCont, params.my_iCub_limits, params.positive_angle_dir)

# Initiate PF and RG patterns for the joints
joint1 = iCubMotor.RShoulderPitch
joint2 = iCubMotor.RShoulderRoll
joint3 = iCubMotor.RShoulderYaw
joint4 = iCubMotor.RElbow

joints = [joint1, joint2, joint3, joint4]

AllJointList = joints
num_joints = 4
angles = np.zeros(params.number_cpg)

# Update CPG initial position (reference position)
for i in range(0, len(myCont)):
    myCont[i].fUpdateInitPos(angles[i])
# Update all joints CPG, it is important to update all joints
# at least one time, otherwise, non used joints will be set to
# the default init position in the CPG which is 0
for i in range(0, len(myCont)):
    myCont[i].fUpdateLocomotionNetwork(myT, angles[i])

angles[iCubMotor.RShoulderPitch] = -10.
angles[iCubMotor.RShoulderRoll] = 5.
angles[iCubMotor.RShoulderYaw] = -30.
angles[iCubMotor.RElbow] = 95.
#angles = np.radians(angles)


initial_position = wrist_position_icub(np.radians(angles[joints]))[0:3]

print(initial_position)
