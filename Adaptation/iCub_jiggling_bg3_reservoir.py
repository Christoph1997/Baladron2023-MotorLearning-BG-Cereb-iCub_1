	
from ANNarchy import *
#from motor_jiggling_intermediate import *
#from bg_loop3 import *
from reservoir_bg3 import *
#from reservoir_prediction import *
from kinematic import *
from iCub_jiggling_bg3 import *

#import yarp
#import CPG_lib.iCub_connect.iCub_connect as robot_connect
import CPG_lib.parameter as params
from CPG_lib.MLMPCPG.MLMPCPG import *
from CPG_lib.MLMPCPG.myPloting import *
from CPG_lib.MLMPCPG.SetTiming import *

import importlib
import sys
import time
import numpy as np

from numpy import cross, eye, dot
from scipy.linalg import expm, norm

sim = sys.argv[1]
print(sim)

compile()
setup(num_threads=2)
#setup(dt=0.66)

#initialize robot connection
sys.path.append('../../CPG_lib/MLMPCPG')
sys.path.append('../../CPG_lib/icubPlot')
iCubMotor = importlib.import_module(params.iCub_joint_names)
global All_Command
global All_Joints_Sensor
global myCont, angles, myT
All_Command = []
All_Joints_Sensor = []
RG_Layer_E = []
RG_Layer_F = []
PF_Layer_E = []
PF_Layer_F = []
MN_Layer_E = []
MN_Layer_F = []
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
# Initiate PF and RG patterns for the joints
joint1 = iCubMotor.LShoulderRoll
joint2 = iCubMotor.LElbow
joint3 = iCubMotor.LShoulderPitch
joint4 = iCubMotor.LShoulderYaw
joints = [joint4,joint3,joint1,joint2]
AllJointList = joints
num_joints = 4
angles = np.zeros(params.number_cpg)


angles[iCubMotor.LShoulderPitch] = 40
angles[iCubMotor.LElbow] = -10
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
#VelInter.disable_learning()

#Hand_velocity.disable()



RG_Pat1.factor_exc = 1.0
RG_Pat2.factor_exc = 1.0
RG_Pat3.factor_exc = 1.0
RG_Pat4.factor_exc = 1.0
PF_Pat1.factor_exc = 1.0
PF_Pat2.factor_exc = 1.0
Inj_Curr.factor_exc = 1.0


#VelInter.transmission = False

def gaussian_input(x,mu,sig):
             return np.exp(-np.power(x-mu,2.)/(2*np.power(sig,2)))


num_trials_test = 100

distance_history = np.zeros(num_trials_test)
goal_history= np.zeros((num_trials_test,3))
parameter_history = np.zeros((num_trials_test,4,6))
final_pos_history = np.zeros((num_trials_test,3))


hc_goals = [ [0.3286242/2.,  0.33601961/2., 0.55/2.], [-0.8713758/2.,   0.3360196/2.,  0.55/2.]]






max_angle = 0
num_tests = 0
a = [0,0,0]


#train_bg(0)
#StrD1SNr_putamen.disable_learning()
#StrD1SNc_put.disable_learning()




pop.enable()

num_goals = 2

num_trials = num_goals*300 #600

num_rotation_trials = 200
num_test_trials = 200

error_history = np.zeros(num_trials+num_rotation_trials+num_test_trials)
angle_history = np.zeros(num_trials+num_rotation_trials+num_test_trials)
angle_history2 = np.zeros(num_trials+num_rotation_trials+num_test_trials)
angle_history3 = np.zeros(num_trials+num_rotation_trials+num_test_trials)
#error_historyP = np.zeros(num_trials+10)


dh = np.zeros(num_trials)


simulation_type = 0 #0 = with bg / 1 = direct parameters


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])





#BG controller
goal_history, parameter_history = preproc(num_goals)

StrD1SNr_putamen.disable_learning()
StrD1SNc_put.disable_learning()



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


perpendicular_vector = np.cross(goal_history[0],goal_history[1])
perpendicular_normalized = perpendicular_vector/np.linalg.norm(perpendicular_vector)
rot = rotation_matrix( perpendicular_vector  ,np.radians(45))

def angle_in_plane(v1,v2,n):
    dot = np.dot(v1,v2)
    det = v1[0]*v2[1]*n[2] + v2[0]*n[1]*v1[2] + n[0]*v1[1]*v2[2]  - v1[2]*v2[1]*n[0] - v2[2]*n[1]*v1[0] - n[2]*v1[1]*v2[0]
    return np.arctan2(det,dot)

cerror = np.zeros(num_trials+num_rotation_trials+num_test_trials)

for t in range(num_trials+num_rotation_trials+num_test_trials):

    goal_id = t%num_goals
    if(t>num_trials):
        goal_id = 0
    current_goal =  goal_history[goal_id]   
    

    pop.x = Uniform(-0.01, 0.01).get_values(N)
    pop.r = np.tanh(pop.x)
    pop[1].r = np.tanh(1.0)
    pop[10].r = np.tanh(1.0)
    pop[11].r = np.tanh(-1.0)



    inp[goal_id].r = 1.0
    
    simulate(200)

    inp.r  = 0.0

    simulate(200)

    rec = m.get()
    
    output = rec['r'][-200:,-24:]
    output = np.mean(output,axis=0)


    if(simulation_type==0):
        output=output*2


    current_parms = np.zeros((4,6))
    if(simulation_type == 0):
        current_parms =  np.copy(parameter_history[goal_id])

    #Turn this on for simulations with strategy
    if(t>(num_trials+2) and t<(num_trials+num_rotation_trials-10) ):
        current_parms = np.copy(parameter_history[2])

    if(t>-1):
        current_parms+=output.reshape((4,6))        

    
    s = 0
    pf = ''
    if(t>(num_trials-3)):
        s = 1
        pf = str(t)
    final_pos = execute_movement(current_parms,s,pf)

    #Turn this on for simulations with perturbation
    if(t>num_trials and t<(num_trials+num_rotation_trials) ):
        final_pos = np.dot(rot,final_pos) 


    distance = np.linalg.norm(final_pos-current_goal)
    #Activate this for simulations with strategy 
    if(t>(num_trials) and t<(num_trials+num_rotation_trials)):
        distance = np.linalg.norm(final_pos-goal_history[2]) 
    error = distance 


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


    
    R_mean[goal_id] = alpha * R_mean[goal_id] + (1.- alpha) * error
    error_history[t] = error
    rotated_proj = project_onto_plane(final_pos,perpendicular_vector)
    angle_history[t]  = np.degrees( angle_between(rotated_proj, current_goal))
    angle_history2[t] = np.degrees( angle_between(current_goal, initial_position) - angle_between(rotated_proj,initial_position) )
    angle_history3[t] = np.degrees( angle_in_plane(rotated_proj,current_goal,perpendicular_normalized) )
    cerror[t] = error

np.save(sim+'angle3.npy',angle_history3) #Directional error
np.save(sim+'cerror.npy',cerror) #Aiming error


