#!/usr/bin/python
import re
import Tokenizer
import math

# Removes the indexes from state and transition descriptions
def strip_indexes(token):	
	if(token.isdigit()):
		return token
	param = token		
	if(re.search("[0-9]+x[0-9]+", token) != None and token.startswith("--")):
		param = param[:-3]
	elif (re.search("[0-9]+", token) != None and token.startswith("--")):
			param = param[:-1]		
				
	return param
	
# returns a list with the parameters of a state or a transition 
def parse_parameters(tokenizer):
		parameters = [strip_indexes(tokenizer.next_token())]     # --sK or --nKxW 
		parameters.append(strip_indexes(tokenizer.next_token())) # type
		while(tokenizer.has_more_tokens() and (re.search("--n[0-9]+x[0-9]+", tokenizer.peek()) == None and re.search("--s[0-9]+", tokenizer.peek()) == None)):
			parameters.append(strip_indexes(tokenizer.next_token()))
		return parameters
		
# This class models a state of a FSM 
class AutoMoDeFSMState:
	def __init__(self, id, tokenizer, states_map):
		self.id = id                    # the state id (0,1,2...)
		self.original_id = id
		self.counter = 0                # this variable counts how many times the state is active
		self.loop = False
		self.state_paramas = parse_parameters(tokenizer) # parsing the state parameters
		self.transition_counters = []   # counters for tracing how many times each transition is activated
		self.transition_definition = [] # definition and parameters of each transition
		self.transition_active = []     # A transition is active if the destination state is in the FSM
		self.transition_destination = []# The state where the transition lands
		self.transition_p = []          # The probability that the transition will be taken if verified
		self.transition_type = []
		self.transition_w = []          
		self.states_mapping = states_map# a list defining the id of all states
		self.active = True
		
		while(tokenizer.has_more_tokens() and re.search("--s[0-9]+", tokenizer.peek()) == None):
			self.transition_counters.append(0)
			self.transition_active.append(True)
			transition_parameters = parse_parameters(tokenizer)
			self.transition_definition.append(transition_parameters)
			self.transition_destination.append(int(transition_parameters[1]))
			self.transition_w.append(0)
			#print("Transition {0} -> {1}".format(len(self.transition_counters), transition_parameters))
			for ind,param in enumerate(transition_parameters):				
				if(param.startswith("--p")):
					pparam = float(transition_parameters[ind+1])
					self.transition_p.append(pparam)
				elif(param.startswith("--c")):
					ttype = int(transition_parameters[ind+1])
					self.transition_type.append(ttype)
				elif(param.startswith("--w")):
					wparam = float(transition_parameters[ind+1])
					self.transition_w[-1] = wparam				
	
	def clone(self, fsmstate_to_clone):
		self.id = fsmstate_to_clone.id  # the state id (0,1,2...)
		self.counter = fsmstate_to_clone.counter  # this variable counts how many times the state is active
		self.state_paramas = fsmstate_to_clone.state_params # parsing the state parameters
		self.transition_counters = fsmstae_to_clone.counters   # counters for tracing how many times each transition is activated
		self.transition_definition = fsmstate_to_clone.definition # definition and parameters of each transition
		self.states_mapping = fsmstate_to_clone.states_mapping# a list defining the id of all states
	
	def increase_counter_state(self):
		self.counter += 1
	
	def increase_counter_transition(self, index):
		self.transition_counters[index] += 1
	
	# this methods returns a string representing the state 
	# as a series of parameters for automode.
	# Transitions that have never been active according to the log will be removed
	# and the others will receive a new index if needed.
	def __str__(self):
		state_description = ""
		new_transation_index = 0
		transform = False
		trs_number = False
		#if(self.counter > 0):			
		for par in self.state_paramas:	
			if(trs_number):
				state_description += " "+str(self.active_transitions())
				trs_number = False
			else:
				state_description += " "+par			
			
			if(par.startswith("--")):
				state_description += str(self.id)
				if(par.startswith("--n")):
					trs_number = True				
				
		for idx in range(0,len(self.transition_counters)):
			if(self.transition_active[idx]):
				for trs_par in self.transition_definition[idx]:
					if(not(transform)):
						state_description += " " + trs_par
						if(trs_par.startswith("--")):
							state_description += "{0}x{1}".format(self.id,new_transation_index)
							if(trs_par.startswith("--n")):
								transform = True
					else:
						state_description += " {0}".format(self.states_mapping[int(trs_par)])
						transform = False
					
				new_transation_index += 1
		
		return state_description	
		
	def __repr__(self):
		state_description = str(self.state_paramas)
		for idx,trs in enumerate(self.transition_counters):			
				state_description += " " + str(self.transition_definition[idx])
		
		return state_description	
		
	def get_counter(self):
		return self.counter
	
	def get_transition_counters(self):
		return self.transition_counters
	
	def get_id(self):
		return self.id	
	
	def set_id(self,new_id):
		self.id = new_id
		
	def deactivate_transition_to_state(self, state):
		for idx,trs in self.transition_destination:
			if(trs == state):
				self.transition_active[idx] = False
	
	def deactivate_transition_to_states(self, states, keep_connection=False):
		#print("State {0}".format(self.id))
		#print("States to deactivate {0}".format(states))
		#print("Active transitions {0}".format(self.transition_active))
		#print("Transition destinations {0}".format(self.transition_destination))
		for state in states:
			toremove = state
			if(state > self.id):
				toremove = state-1
			for idx,trs in enumerate(self.transition_destination):
				if(trs == toremove):
					self.transition_active[idx] = False
					
		#print("Active transitions after deactivation {0}".format(self.transition_active))
		oops = True
		for trs in self.transition_active:
			if(trs):
				oops = False
				break
		
		if(oops and keep_connection):
			#print("Warning: state {0} has no active transitions".format(self.id))
			#print("A transition will be reactivated in order to have a functioning FSM")
			self.transition_active[0] = True
	
	def active_transitions(self):
		atrs = 0
		for idx,trs in enumerate(self.transition_active):
			if(trs):# and self.transition_counters[idx] > 0):
				atrs += 1
			
		return atrs
	
	def num_of_transitions_to(self, state):
		num = 0
		smap = state
		if(state > self.id):
			smap = state-1
		
		for destination in self.transition_destination:
			if(destination == smap):
				num +=1
			
		return num
	
	def get_transition_probability(self, transition, num_neighbors=0, ground_sensor=-1):
		type = self.transition_type[transition]
		prob = self.transition_p[transition]
		blackGroundThreshold = 0.1
		whiteGroundThreshold = 0.95		
		type_name = ["BlackFloor", "GrayFloor", "WhiteFloor", "NeighborsCount", "InvertedNeighborsCount", "FixedProbability"]	
		if type == 0 and ( ground_sensor >= blackGroundThreshold or ground_sensor < 0) :
			prob = 0.0
			#print("State {0} Transition {1} type 0 -> prob {2}".format(self.id, transition,prob))
		elif type == 1 and ( ground_sensor >= whiteGroundThreshold or ground_sensor < blackGroundThreshold or ground_sensor < 0):
			prob = 0.0
			#print("State {0} Transition {1} type 1 -> prob {2}".format(self.id, transition,prob))
		elif type == 2 and ( ground_sensor < whiteGroundThreshold or ground_sensor < 0) :	
			prob = 0.0
			#print("State {0} Transition {1} type 2 -> prob {2}".format(self.id, transition,prob))q
		elif type == 3 :
			prob = 1.0/(1.0 + math.exp(self.transition_w[transition]*(prob-num_neighbors)))
			#print("State {0} Transition {1} type 3 -> prob {2}".format(self.id, transition,prob))
		elif type == 4:
			prob = 1.0 - 1.0/(1.0 + math.exp(self.transition_w[transition]*(prob-num_neighbors)))
			#if(self.id == 0):
			#print("State {0} Transition {1} type 4 -> prob {2} [ p {3} | w {4} | n {5}]".format(self.id, transition,prob,self.transition_p[transition],self.transition_w[transition],num_neighbors))		
		#if prob == 0:# and transition == 0:
			#print("State {0} Transition {1} type {2} ground {3} neighbors {4} P {5} -> prob {6}".format(self.id, transition,type_name[type],ground_sensor,num_neighbors,self.transition_p[transition],prob))
		return prob 
	
	def prob_of_reaching_state(self, target, states, num_neighbors=0, ground_sensor=-1):		
		if(self.loop):
			self.loop = False
			return 0.0
			
		active_transitions = []
		
		for idx,active in enumerate(self.transition_active):
			if(active):
				active_transitions.append(idx)
		
		num = len(active_transitions)#Number of transition for the state
		if(num == 0):
			num +=1.0
		if(self.original_id == target):	
				return 1.0/(num)
		
		#smap = target 
		prob = 0.0
		smap = 0
		for st in range(0,len(self.states_mapping)):
			if(st == target):
				break
			elif(st != self.id):
				smap +=1
				
		#if(target > self.id):			
		#	smap = self.states_mapping[target-1] #Set smap to the correct transition target for target
		
		prob = 0 #Start with 0 since there can be more than one transition to target
		#for idx,destination in enumerate(self.transition_destination):
		for idx in active_transitions:
			destination = self.transition_destination[idx]
			#print("State {0} Transition {1} to {2} - target {3} {4} ".format(self.id,self.transition_type[idx],destination,target, states[target].original_id))
			if destination == smap : #if state has a transition to target							
				tprob = self.get_transition_probability(idx,num_neighbors,ground_sensor)
				prob += tprob/float(num) #Add the probability of taking that transition
		
		#if(self.id == 0):
		#	print("State {0} probability of reaching directly state {1} : {2}".format(self.id, target, prob))
			
		if prob == 0 : #There is no direct transition to target
			#for idx,destination in enumerate(self.transition_destination) :			
			for idx in active_transitions:
				true_destination = self.transition_destination[idx]				
				if(true_destination >= self.id):
					true_destination = destination+1
					#print("State {5} target {4} -> true {0} vs real {1} : {2} destinations {3}".format(true_destination, states[true_destination].id,self.states_mapping,self.transition_destination,target,self.original_id))				
				self.loop = True
				nprob = states[true_destination].prob_of_reaching_state(target,states,num_neighbors,ground_sensor)
				#print("State {5} target {4} -> true {0} vs real {1} : {2} destinations {3} = {6}".format(true_destination, states[true_destination].id,states[true_destination].states_mapping,states[true_destination].transition_destination,target,self.original_id,nprob))
				#nprob *= (self.transition_p[idx]/float(num))
				nprob *= self.get_transition_probability(idx,num_neighbors,ground_sensor)/float(num)
				if(nprob > prob):
					prob = nprob
				#print("State {0} probability of reaching indirectly state {1} : {2}".format(self.id, target, prob))
		if prob == 0:
			prob = 0.0
			#print("WARNING: from state {0} it is impossible to reach state {1}!".format(self.id, target))
			
		return prob
						
	
	def update_states_map(self,new_states_map):
		#print("S{1} old state mapping : {0} ".format(self.states_mapping,self.id))
		self.states_mapping = new_states_map #updates the state mappings		
		#print("S{1} new state mapping : {0} ".format(self.states_mapping,self.id))
		#print("S{1} transition destinations before update : {0} ".format(self.transition_destination,self.id))
		#for idx in range(0, len(self.transition_destination)): # updates the destinations for the transitions
		#	current_destination = self.transition_destination[idx]			
		#	self.transition_destination[idx] = self.states_mapping[current_destination]
		#print("S{1} transition destinations after update  : {0} ".format(self.transition_destination,self.id))			
		
	def is_transition_active(self, transition):			
		return self.transition_active[transition]		
		