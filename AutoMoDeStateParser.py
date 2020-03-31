#!/usr/bin/python
import re
import Tokenizer

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
		self.counter = 0                # this variable counts how many times the state is active
		self.state_paramas = parse_parameters(tokenizer) # parsing the state parameters
		self.transition_counters = []   # counters for tracing how many times each transition is activated
		self.transition_definition = [] # definition and parameters of each transition
		self.states_mapping = states_map# a list defining the id of all states
		while(tokenizer.has_more_tokens() and re.search("--s[0-9]+", tokenizer.peek()) == None):
			self.transition_counters.append(0)
			self.transition_definition.append(parse_parameters(tokenizer))
	
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
		if(self.counter > 0):			
			for par in self.state_paramas:
				state_description += " "+par
				if(par.startswith("--")):
					state_description += str(self.id)
			for idx,trs in enumerate(self.transition_counters):
				if(trs > 0):
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
	
	def update_states_map(self,new_states_map):
		self.states_mapping = new_states_map
				
		