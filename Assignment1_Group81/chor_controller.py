from controller import Controller
import numpy as np
import copy

# An individual is represented by a sequence of steps to perform: each step has an integer encoding from 0 to 5 
# 0 -> stay still
# 1 -> left
# 2 -> right
# 3 -> jump
# 4 -> shoot
# 5 -> release
# The individual can make only a move at a time and does not take into consideration any sensory signal
class player_controller(Controller):
	def __init__(self):
		self.step=0 # array index of the action to be performed
		self.previous="" # previous individual

	def control(self, inputs, controller):
		
		if not np.array_equal(self.previous, controller): # check if the current player is same as previous
			self.step=0
			self.previous=copy.deepcopy(controller)

		action=0 # encoding of the move
		if len(controller)>self.step:
			action=controller[self.step] # retrieval of the move
			
		self.step+=1 # move to next action

		left=0
		right=0
		jump=0
		shoot=0
		release=0

		# Decoding actions
		if(action==1): left=1
		if(action==2): right=1
		if(action==3): jump=1
		if(action==4): shoot=1
		if(action==5): release=1

		return [left, right, jump, shoot, release]

