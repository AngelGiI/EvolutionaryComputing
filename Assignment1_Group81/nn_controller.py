###############################################################################
# EvoMan FrameWork - Assignment 1			                                  #
# Neuroevolution - Genetic Algorithm  neural network.                         #
# Author: Group 81 			                                                  #
###############################################################################
from controller import Controller
import numpy as np


def sigmoid(x):
	return 1./(1.+np.exp(-x))


# 3-layer fully-connected neural network with 10 hidden neurons: 20 inputs -> 10 neurons -> 10 neurons -> 5 outputs

class player_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0] 
			weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))
			# Outputs activation first layer.
			output1 = sigmoid(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + self.n_hidden[0]].reshape(1,self.n_hidden[0]) 
			weights2_slice = weights1_slice + self.n_hidden[0] 
			weights2 = controller[weights2_slice : weights2_slice + (self.n_hidden[0] *self.n_hidden[0]) ].reshape((self.n_hidden[0],self.n_hidden[0]))

			# Outputting activated second layer. Each entry in the output is an action
			output2 = sigmoid(output1.dot(weights2)+ bias2)[0]

			# Preparing the weights and biases from the controller of layer 3
			bias3_slice =weights2_slice+(self.n_hidden[0] *self.n_hidden[0])
			bias3 = controller[bias3_slice:bias3_slice + 5].reshape(1,5)
			weights3 = controller[bias3_slice + 5:].reshape((self.n_hidden[0],5))

			# Outputting activated third layer. Each entry in the output is an action
			output = sigmoid(output2.dot(weights3)+ bias3)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid(inputs.dot(weights) + bias)[0]

		# Decision making
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]

