#!/usr/bin/env python2
# Test the Inverse Kinematic Model Estimator

import numpy as np
import math
import intera_interface
import rospy
import roslib
import tensorflow
from keras.models import load_model

# Initialize ROS/robot 
rospy.init_node('inverse_model_testing')
limb=intera_interface.Limb('right')

joint_names = ['right_j0','right_j1','right_j3','right_j5','right_j6']
#p_cube = np.array([[0,-0.6,-0.07]])
#p_cube = np.array([0.1, -0.1,-0.07])
p_cube = np.array([0.4,-0.5,-0.07])

forward_model_file='weights/ForwardModel.h5'
inverse_model_file='weights/InverseModelCombined.h5'

ForwardModel = load_model(forward_model_file)
InverseModel = load_model(inverse_model_file)

ForwardModel.summary()
InverseModel.summary()

# Robot has fixed starting location
angles = dict()
angles['right_j0']=math.radians(0)
angles['right_j1']=math.radians(-50)
angles['right_j2']=math.radians(0)
angles['right_j3']=math.radians(120)
angles['right_j4']=math.radians(0)
angles['right_j5']=math.radians(0)
angles['right_j6']=math.radians(0)
limb.move_to_joint_positions(angles)

# Define the state for the first time
prev_state = None
prev_q= None
is_first_run = True
velocities=dict()

# Velocity scale
dt=5

while not rospy.is_shutdown():

	# Grab the position of the joints
	joint_positions=limb.joint_angles()

	# Convert them to an np array
	q=np.array([[float(joint_positions[i]) for i in joint_names]])

	# Predict EE position based on q
	prediction=ForwardModel.predict(q)
        
        if is_first_run:
            is_first_run = False
            prev_state = prediction
            prev_q = q
	
        prediction_combined = np.hstack((prev_state - p_cube, prediction - p_cube))
	joint_estimation = InverseModel.predict(prediction_combined)
        
	#Push the velocities to the robot
	velocities['right_j0'] = (joint_estimation[0][0]-prev_q[0][0])/dt
	velocities['right_j1'] = (joint_estimation[0][1]-prev_q[0][1])/dt
	velocities['right_j2'] = 0
	velocities['right_j3'] = (joint_estimation[0][2]-prev_q[0][2])/dt
	velocities['right_j4'] = 0
	velocities['right_j5'] = (joint_estimation[0][3]-prev_q[0][3])/dt
	velocities['right_j6'] = (joint_estimation[0][4]-prev_q[0][4])/dt

	limb.set_joint_velocities(velocities)
	
        print '-----------------'
	print prev_state.shape
	print joint_estimation.shape
        print velocities
        
	# The current prediction becomes the previous state during the next iteration
	prev_state = prediction
	prev_q = q

