#!/usr/bin/env python

# Script to Test Keras

# import libraries
import os
import glob
import numpy as np
import sympy as sp
import math
#import rospy
#import roslib
import tensorflow
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense
from keras import optimizers
from keras.models import load_model
from sympy import lambdify
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def get_position(matrix):

	#print matrix

	return [matrix[0][3],matrix[1][3],matrix[2][3]]


# Forward Model location
model_file='weights/ForwardModel.h5'

# Load the model
forwardModel = load_model(model_file)

# Define the paths of the data
#training_file='/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/InverseModelData/data5.txt'
#training_path = '/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/InverseModelData/data_*'
#training_files = glob.glob(training_path)
#training_files = ['/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/InverseModelData/data_5', 
#					'/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/InverseModelData/data_6', 
#					'/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/InverseModelData/data_7', 
#					'/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/InverseModelData/data_8']
training_files = ['data/data_5', 'data/data_6','data/data_7', 'data/data_8']  

# Define where you want the Inverse model to be stored
#inverse_model_file='/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/InverseModel/InverseModelCombined.h5'
inverse_model_file='weights/InverseModelCombined.h5'


# Create the training dataset
#training_data= np.loadtxt(training_files[0],delimiter=',',skiprows=1)

training_data = np.array([np.loadtxt(file,delimiter=',',skiprows=1) for file in training_files])


# Dataset to learn the Inverse kinematics of the Sawyer Robot.

# Declate some empty array lists
input_training_data_r=np.empty((4,0,3),float)
input_training_data_n= [] #np.empty((0,3),float)

# Position of the cube
#p_cube=np.array([[0.187,-0.462,-0.07]])

p_cube1 = np.array([0.18529411764705872, -0.46270270270270264,-0.07])
p_cube2 = np.array([-0.17867647058823533, -0.47783783783783784,-0.07])
p_cube3 = np.array([0.1874999999999999, -0.6983783783783784,-0.07])
p_cube4 = np.array([-0.17647058823529416, -0.7027027027027026,-0.07])
p_cubes = np.array([p_cube1, p_cube3, p_cube2, p_cube4])

out_tr_dat = [np.delete(data,[0,3,7,8],axis=1) for data in training_data]
input_training_data_n = np.array([np.array([ forwardModel.predict(np.array([i])) -p_cubes[j] for i in out_tr_dat[j]]) for j in range(len(training_data))])


print(training_data[0])
"""
for index, data in enumerate(training_data):
	# Create the labels of the training data
	output_training_data=np.delete(data,[0,3,7,8],axis=1)

	for i in output_training_data:

		# Get the correct joint angles
		#angles=robot.JointOffset([i[0],i[1],0,i[2],i[3],0,0])

		# Estimate the real position
		#pos_r=get_position(Te(angles[0],angles[1],angles[2],angles[3],angles[4],angles[5],angles[6]))

		# Append the real estimation
		#input_training_data_r=np.append(input_training_data_r, np.array([pos_r]), axis=0)

		# Estimate the prediction
		prediction=forwardModel.predict(np.array([i]))

		# Append the network estimation
		print index
		print input_training_data_n[index]
		input_training_data_n=np.append(input_training_data_n[index], prediction-p_cubes[index], axis=0)
"""
#print input_training_data_n

#input_training_data_n = np.concatenate(input_training_data_n)
print input_training_data_n[0].shape
# Plot the trajectories
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
X = input_training_data_n[:,[0]]
Y = input_training_data_n[:,[1]]
Z = input_training_data_n[:,[2]]

# Make the scatter plot
surf = ax.scatter(X, Y, Z)

# Show the scatter plot
# plt.show()


# Define the Inverse Model
InverseModel=Sequential()

# Add the input Layer and the 1rst Hidden Layer
InverseModel.add(Dense(200,input_shape=(6,),init="uniform",activation="sigmoid"))

# Add a second Hidden Layer
InverseModel.add(Dense(200,init="uniform",activation="sigmoid"))

# Add the Output Layer
InverseModel.add(Dense(5))

# Define the Training Process
InverseModel.compile(optimizer='adam',loss='mse',metrics=['accuracy'])

# Define the final input and output data
input_training_data=input_training_data_n[:-1]
output_training_data=output_training_data[1:]
input_training_data_t = np.vstack((input_training_data_n[0], input_training_data_n[:-2]))
input_training_combined = np.hstack((input_training_data, input_training_data_t))


# Train the Model
InverseModel.fit(input_training_combined,output_training_data,batch_size=1)

# Save the Model
InverseModel.save(inverse_model_file)




	 
	 
