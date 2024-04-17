from kinematic import *
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *
from train_BG_calibration_todorov import *

import importlib
import sys
import time
import numpy as np
import yarp
#import yarp.math as ym --> not available for python only in C++
import copy
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
goal = copy.deepcopy(initial_position)
goal[0] = goal[0] - 0.1

print("Initial position:", initial_position)
print("Goal:", goal)

runs = 10
parameter_history = []
s = 0
pf = ''

for i in range(runs):
    parameter_history.append(np.load(f"results_todorov/calibration_todorov/frequency_1/run_{i}/parameter_history_bg_adapt.npy")) 

    final_pos = execute_movement(parameter_history[i][0],s,pf)

    distance = np.linalg.norm(final_pos-goal)
    print(f"Distance run{i}:", distance)


print("Parameter values:", parameter_history)

yarp.Network.fini()

"""
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
goal = copy.deepcopy(initial_position)
goal[0] = goal[0] - 0.1

print(initial_position)

yarp.Network.fini()
"""