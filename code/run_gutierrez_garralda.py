#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Code for the paper:

Baladron, J., Vitay, J., Fietzek, T. and Hamker, F. H.
The contribution of the basal ganglia and cerebellum to motor learning: a neuro-computational approach.

Copyright the Authors (License MIT)

Script for the adaptation task.

> python run_adaptation.py
"""

# Imports
import importlib
import sys
import time
import numpy as np
from numpy import cross, eye, dot
from scipy.linalg import expm, norm
from monitoring import Con_Monitor
from pathlib import Path

#TODO: change num_actions in the Basalganglia to 100?
#TODO: change number of trials to correct amounts --> DONE
# Parameters
num_goals = 2 # Number of goals
num_baseline_trials = 525 # Number of trials for baseline phases + 500 training trials
num_rotation_trials = 25 # Number of rotation trials for each visuomotor adaptation
num_test_trials = 25 # Number of test trials at the end to finish
strategy = 0 # Set to 1 to simulate a condition which includes an explicit instruction
rotation = int(sys.argv[3]) # Set to 1 to simulate a condition in which the 30-degree shift perturbation is included and to 0 for the x-reflection perturbation

# Import ANNarchy
from ANNarchy import *
setup(num_threads=2)

# Model
from reservoir import *
from kinematic import *
from train_BG_gutierrez_garralda import *

# CPG
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *

# update frequeny, amplitude and learnrate of the reservoir
Wrec.eta = 0.8
pop.f = float(sys.argv[2])
pop.A = 20.

# Prepare save directory
sub_folder = "/gutierrez_garralda/frequency_" + sys.argv[2] + "_rotation_" + sys.argv[3] +  "/run_" + sys.argv[1] + "/"
folder_net = "./results" + sub_folder
Path(folder_net).mkdir(parents=True, exist_ok=True)

# get the random state of numpy
randomstate = np.random.get_state
with open(folder_net + "randomstate", "wb") as f:
    pickle.dump(randomstate, f)

# Compile the network
directory_ann = "./annarchy" + sub_folder
Path(directory_ann).mkdir(parents=True, exist_ok=True)
compile(directory=directory_ann)

# init monitor for tracking weight changes
con_monitor = Con_Monitor([Wi, Wrec])
con_monitor.extract_weights()

# Initialize robot connection
sys.path.append('../../CPG_lib/MLMPCPG')
sys.path.append('../../CPG_lib/icubPlot')
iCubMotor = importlib.import_module(params.iCub_joint_names)
myT = fSetTiming()

# Create list of CPG objects
myCont = fnewMLMPcpg(params.number_cpg)
# Instantiate the CPG list with iCub robot data
myCont = fSetCPGNet(myCont, params.my_iCub_limits, params.positive_angle_dir)

"""
    NeckPitch, NeckRoll, NeckYaw, EyesTilt, EyesVersion, EyesVergence, TorsoYaw, TorsoRoll, TorsoPitch, RShoulderPitch, RShoulderRoll, \
    RShoulderYaw, RElbow, RWristProsup, RWristPitch, RWristYaw, RHandFinger, RThumbOppose, RThumbProximal, RThumbDistal, RIndexProximal, \
    RIndexDistal, RMiddleProximal, RMiddleDistal, RPinky, RHipPitch, RHipRoll, RHipYaw, RKnee, RAnklePitch, RAnkleRoll, \
    LShoulderPitch, LShoulderRoll, LShoulderYaw, LElbow, LWristProsup, LWristPitch, LWristYaw, LHandFinger, LThumbOppose, LThumbProximal, \
    LThumbDistal, LIndexProximal, LIndexDistal, LMiddleProximal, LMiddleDistal, LPinky, LHipPitch, LHipRoll, LHipYaw, LKnee, \
    LAnklePitch, LAnkleRoll
"""

# Initiate PF and RG patterns for the joints
joint1 = iCubMotor.RShoulderPitch
joint2 = iCubMotor.RShoulderRoll
joint3 = iCubMotor.RShoulderYaw
joint4 = iCubMotor.RElbow

joints = [joint1, joint2, joint3, joint4]

AllJointList = joints
num_joints = 4
#TODO: change angles to initial position --> DONE
angles = np.zeros(params.number_cpg)

angles[iCubMotor.RShoulderPitch] = -10.
angles[iCubMotor.RShoulderRoll] = 5.
angles[iCubMotor.RShoulderYaw] = -30.
angles[iCubMotor.RElbow] = 95.
#angles = np.radians(angles)

# Update CPG initial position (reference position)
for i in range(0, len(myCont)):
    myCont[i].fUpdateInitPos(angles[i])
# Update all joints CPG, it is important to update all joints
# at least one time, otherwise, non used joints will be set to
# the default init position in the CPG which is 0
for i in range(0, len(myCont)):
    myCont[i].fUpdateLocomotionNetwork(myT, angles[i])


for ff in range(num_joints):
    VelRG_Pat1[ff].disable_learning()
    VelRG_Pat2[ff].disable_learning()
    VelRG_Pat3[ff].disable_learning()
    VelRG_Pat4[ff].disable_learning()
    VelPF_Pat1[ff].disable_learning()
    VelPF_Pat2[ff].disable_learning()
    VelInjCurr[ff].disable_learning()

RG_Pat1.factor_exc = 1.0
RG_Pat2.factor_exc = 1.0
RG_Pat3.factor_exc = 1.0
RG_Pat4.factor_exc = 1.0
PF_Pat1.factor_exc = 1.0
PF_Pat2.factor_exc = 1.0
Inj_Curr.factor_exc = 1.0


def gaussian_input(x,mu,sig):
    return np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2)))


#TODO:is it right that way?
hc_goals = [ [0.3286242/2.,  0.33601961/2., 0.55/2.], [-0.8713758/2.,   0.3360196/2.,  0.55/2.]] 

max_angle = 0
num_tests = 0
a = [0,0,0]

pop.enable()

#TODO: Number of total visuomotor adaptation trials --> DONE
num_trials = num_rotation_trials

error_history = np.zeros(num_baseline_trials+num_trials+num_test_trials)
angle_history3 = np.zeros(num_baseline_trials+num_trials+num_test_trials)
#error_historyP = np.zeros(num_trials+10)

#TODO:is it right that way?
dh = np.zeros(num_trials) 

###################
# BG controller
###################
print('Training BG')
goal_history, parameter_history = train_bg(num_goals)


###################
# Reservoir
###################
print('Training reservoir')

def M(axis, theta):
    return expm(cross(eye(3), axis/norm(axis)*theta))


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def normalize(x):
    return x/ np.linalg.norm(x)

def project_onto_plane(x, n):
    d = np.dot(x, n) / np.linalg.norm(n)
    p = d * normalize(n)
    return x-p

StrD1SNc_put.disable_learning()

#TODO: shift-perturbation (translation), in our case by 30 degrees
perpendicular_vector = np.cross(initial_position, goal_history[0])
perpendicular_normalized = perpendicular_vector/np.linalg.norm(perpendicular_vector)
rot30 = rotation_matrix( perpendicular_vector  ,np.radians(-30))

def angle_in_plane(v1,v2,n):
    dot = np.dot(v1,v2)
    det = v1[0]*v2[1]*n[2] + v2[0]*n[1]*v1[2] + n[0]*v1[1]*v2[2]  - v1[2]*v2[1]*n[0] - v2[2]*n[1]*v1[0] - n[2]*v1[1]*v2[0]
    return np.arctan2(det,dot)

cerror = np.zeros(num_baseline_trials+num_trials+num_test_trials)

# Compute the mean reward per trial
R_mean = np.zeros(num_goals)
alpha = 0.33 #0.75 0.33

for t in range(num_baseline_trials+num_trials+num_test_trials):

    # Select goal
    #TODO: select correct goal --> DONE
    if (t<num_baseline_trials):
        goal_id = 0
    elif(t>num_baseline_trials and t<(num_baseline_trials+num_trials)):
        goal_id = 1
    elif(t>num_baseline_trials+num_trials): 
        goal_id = 0
    current_goal = goal_history[goal_id]

    # Reinitialize reservoir
    pop.x = Uniform(-0.01, 0.01).get_values(N)
    pop.r = np.tanh(pop.x)
    pop[1].r = np.tanh(1.0)
    pop[10].r = np.tanh(1.0)
    pop[11].r = np.tanh(-1.0)


    # Set input
    inp[goal_id].r = 1.0
    simulate(200)

    # ISI
    inp.r  = 0.0
    simulate(200)

    rec = m.get()
    output = rec['r'][-200:,-24:]
    output = np.mean(output,axis=0) * 2.0

    current_params =  np.copy(parameter_history[goal_id])

    # Turn this on for simulations with strategy
    # we only learn in the cerebellum for the adaptation
    #TODO: We don't need the if-statement for the strategy, since we only have one goal and strategy is always 0 --> DONE

    current_params+=output.reshape((4,6))

    s = 0
    pf = ''
    if(t>(num_baseline_trials-3)):
        s = 1
        pf = str(t)
    final_pos = execute_movement(current_params,s,pf)

    #Turn this on for simulations with perturbation
    #TODO: We need the if-statement for the rotation, since rotation defines which perturbation we apply--> DONE
    #TODO: Adapt to the new task of Gutierrez-Garralda mit shift-perturbation oder x-reflection perturbation
    if (rotation==1): #shift-perturbation
        if(t>=(num_baseline_trials) and t<(num_baseline_trials+num_rotation_trials) ):
            final_pos = np.dot(rot30,final_pos) 
    else: #x-reflection perturbation
        if(t>=(num_baseline_trials) and t<(num_baseline_trials+num_rotation_trials) ):
            final_pos = np.dot(rot30,final_pos) 
            final_pos[1] = -final_pos[1]


    distance = np.linalg.norm(final_pos-current_goal)

    # Activate this for simulations with strategy
    #TODO: We don't need the if-statement for the strategy, since strategy is always 0 --> DONE

    error = distance

    # Plasticity
    if(t>10):
        # Apply the learning rule
        Wrec.learning_phase = 1.0
        Wrec.error = error
        Wrec.mean_error = R_mean[goal_id]

        # Learn for one step
        step()

        # Reset the traces
        Wrec.learning_phase = 0.0
        Wrec.trace = 0.0
        _ = m.get()

    # Update mean error
    R_mean[goal_id] = alpha * R_mean[goal_id] + (1.- alpha) * error
    error_history[t] = error

    rotated_proj = project_onto_plane(final_pos,perpendicular_vector)
    angle_history3[t] = np.degrees( angle_in_plane(rotated_proj,current_goal,perpendicular_normalized) )
    cerror[t] = error


np.save(folder_net + 'angle3.npy', angle_history3) # Directional error
np.save(folder_net + 'cerror.npy', cerror) # Aiming error


# save goals
np.save(folder_net + 'goals.npy', goal_history)

# extract and save weight after learning
con_monitor.extract_weights()
con_monitor.save_cons(folder=folder_net)

# # save network connectivity
# for proj in projections():
#     proj.save_connectivity(filename=folder_net + 'weights_' + proj.name + '.npz')
