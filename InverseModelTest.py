#!/usr/bin/env python

# Script Test the Inverse Kineamtic Model Estimator

# import libraries
import numpy as np
import math
import intera_interface
import rospy
import roslib
import tensorflow
from keras.models import load_model

# Initialize a ROS node 
rospy.init_node('inverse_model_testing')

# Initialize the Limb object
limb=intera_interface.Limb('right')

# Array to store the joint names
joint_names = ['right_j6','right_j5','right_j4','right_j3','right_j2','right_j1','right_j0']
#p_cube=np.array([[0.187,-0.462,-0.07]])
p_cube=np.array([[0,-0.6,-0.07]])


# Models location
forward_model_file='weights/ForwardModel.h5'
inverse_model_file='weights/InverseModelCombined.h5'

# Load the model
forwardModel = load_model(forward_model_file)
InverseModel = load_model(inverse_model_file)

# Review of the Model
forwardModel.summary()
InverseModel.summary()

# Move the Robot to the starting location
angles = dict()
angles['right_j0']=math.radians(0)
angles['right_j1']=math.radians(-50)
angles['right_j2']=math.radians(0)
angles['right_j3']=math.radians(120)
angles['right_j4']=math.radians(0)
angles['right_j5']=math.radians(0)
angles['right_j6']=math.radians(0)
limb.move_to_joint_positions(angles)

# Grab the position of the joints
joint_positions=limb.joint_angles()

# Convert them to an np array
joint_names = ['right_j0','right_j1','right_j3','right_j5','right_j6']
q=np.array([[float(joint_positions[i]) for i in joint_names]])

# Perform a prediction
prediction=forwardModel.predict(q)

# Define the state for the first time
prev_state = prediction
prev_q=q

# Define the velocity dictionary
velocities=dict()

# Define dt
dt=0.4

while not rospy.is_shutdown():

	# Grab the position of the joints
	joint_positions=limb.joint_angles()

	# Convert them to an np array
	q=np.array([[float(joint_positions[i]) for i in joint_names]])

	#print q

	# Perform a prediction

	prediction=forwardModel.predict(q)
	prediction_combined = np.hstack((prev_state - p_cube, prediction - p_cube))
	joint_estimation = InverseModel.predict(prediction_combined)

	#print joint_estimation
	print '-----------------'

	#print prev_state.shape
	#print joint_estimation.shape
	#Push the velocities to the robot
	velocities['right_j0']=(joint_estimation[0][0]-prev_q[0][0])/dt
	velocities['right_j1']=(joint_estimation[0][1]-prev_q[0][1])/dt
	velocities['right_j2']=0
	velocities['right_j3']=(joint_estimation[0][2]-prev_q[0][2])/dt
	velocities['right_j4']=0
	velocities['right_j5']=(joint_estimation[0][3]-prev_q[0][3])/dt
	velocities['right_j6']=(joint_estimation[0][4]-prev_q[0][4])/dt

        print velocities
        
	limb.set_joint_velocities(velocities)


	# The current prediction becomes the previous state during the next iteration
	prev_state = prediction
	prev_q=q

