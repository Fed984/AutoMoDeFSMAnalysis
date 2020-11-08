import Tokenizer
import gmpy2
from gmpy2 import mpfr
import numpy as np
import sys

def parse_experiment(file_lines, starting_index, fsm):
	exp = AutoMoDeExperiment()	
	logs = []
	idx = starting_index
	cline = file_lines[idx]
	while cline.startswith("[INFO]"):
		idx += 1
		cline = file_lines[idx]
	
	if not(cline.startswith("Score")):
		print("ERROR in line {1} of the log file!!! \n The Score was not after the [INFO] section".format(idx))	
	
	res = float(cline.split()[1])
	
	
	while not(cline.startswith("[INFO]")):
		# parse log line
		idx+=1
		cline = file_lines[idx]
	
		
	return exp
	

# this class models an Experiment keeping track of the result,
# the FSM behavior (the sequence of states the fsm was in and the transitions with their activation probability)
class AutoMoDeExperiment:
	
	def __init__(self):
		self.result = 0.0
		self.logs = [] # contains states, transitions, transitions_probabilities, active_transitions,state_contribution,neighbors
		self.metrics = []
		self.startIdx = 0
		self.endIdx = 0
		self.vpi = []	
		self.objFun = []
		#self.fsm = fsm
	
	def set_result(self, result):
		self.result = result
	
	def set_logs(self, logs):
		self.logs = logs
	
	def append_logs(self, logs):
		self.logs.append(logs)
	
	def set_startIdx(self, idx):
		self.startIdx = idx
	
	def set_endIdx(self, idx):
		self.endIdx = idx
		
	def set_fsm(self, fsm):
		self.fsm = fsm
	
	def set_vpi(self, vpistates):
		self.vpi = vpistates

	def set_objFun(self, objfun):
		self.objFun = objfun

	def calculate_vpi_for_experiment(self):
		vpi = []
		num_of_robots = float(len(self.logs))		
		per_robot_reward = self.result/float(num_of_robots)
		#print("Num of robots {0}, result {1}, reward per robot {2}, states {3}".format(num_of_robots, self.result, per_robot_reward, self.vpi))
		for count in self.vpi:
			vpi.append((count * per_robot_reward)/num_of_robots)
			
		#print("Episode vpi {0}".format(vpi))
		return vpi
	
	# Compute the state values and proportional state values of the original experiment using
	# first visit or every visit and discount factor.
	def calculate_state_values(self, first_visit=True, discounted_factor=0.98):
		timesteps = self.endIdx+1
		overall_state_values = np.zeros(self.number_of_states())
		overall_state_proportional_values = np.zeros(self.number_of_states())
		num_of_robots = float(len(self.logs))
		per_robot_reward = self.result/float(num_of_robots)
		state_proportional_rewards = np.zeros(self.number_of_states())

		for episode in self.logs:
			state_values = np.zeros(self.number_of_states())
			state_values_proportional = np.zeros(self.number_of_states())
			state_number_of_visits = np.zeros(self.number_of_states())
			per_robot_reward = self.result/float(num_of_robots)

			for s in range(0, self.number_of_states()):				
				state_proportional_rewards[s] = (episode[4][s]/float(timesteps) * per_robot_reward)	

			for index, _ in reversed(list(enumerate(episode[0]))):
				if index == 0:
					break
				previous_state = episode[0][index - 1] # previous state
				state_number_of_visits[previous_state] += 1

				if state_number_of_visits[previous_state] == 1 or not first_visit:
					state_values[previous_state] += per_robot_reward
					state_values_proportional[previous_state] += state_proportional_rewards[previous_state]
				
				per_robot_reward *= discounted_factor
				state_proportional_rewards *= discounted_factor

			if not first_visit:
				for s in range(0, self.number_of_states()):
					if state_number_of_visits[s] != 0:
						overall_state_values[s] += state_values[s] / state_number_of_visits[s]
						overall_state_proportional_values[s] += state_values_proportional[s] / state_number_of_visits[s]
			else:
				overall_state_values += state_values
				overall_state_proportional_values += state_values_proportional

		return overall_state_values / len(self.logs), overall_state_proportional_values / len(self.logs)
	
	# Compute the state values and proportional state values of the original experiment using
	# first visit or every visit and discount factor.
	def calculate_state_values_intermediate_rewards(self, first_visit=True, discounted_factor=0.98):
		timesteps = self.endIdx+1
		overall_state_values = np.zeros(self.number_of_states())
		overall_state_proportional_values = np.zeros(self.number_of_states())
		num_of_robots = float(len(self.logs))
		per_robot_reward = self.result/float(num_of_robots)
		state_proportional_rewards = np.zeros(self.number_of_states())
		state_number_of_visits = np.zeros(self.number_of_states())

		# state_values = np.zeros(self.number_of_states())
		# state_values_proportional = np.zeros(self.number_of_states())
		state_values = []
		state_values_proportional = []

		for s in range(0, self.number_of_states()):	
			state_values.append(mpfr(0))
			state_values_proportional.append(mpfr(0))

		for episode in self.logs:
			g = mpfr(0)
			g_prop = [] #np.zeros(self.number_of_states())
			for s in range(0, self.number_of_states()):		
				g_prop.append(mpfr(0))
		
			
			per_robot_reward = self.result/float(num_of_robots)

			for index, state in reversed(list(enumerate(episode[7][:-1]))):
				per_robot_reward = self.objFun[index] / float(num_of_robots) # reward for each robot/episode
				for s in range(0, self.number_of_states()):				
					state_proportional_rewards[s] = (episode[4][s]/float(timesteps) * per_robot_reward)	

				g = mpfr(g * discounted_factor + per_robot_reward)
				g_prop[state] = mpfr(g_prop[state] * discounted_factor + state_proportional_rewards[state])
				
				if per_robot_reward != 0 and (not state in episode[7][0:index] or not first_visit):
					state_number_of_visits[state] += 1
					state_values[state] += g
					state_values_proportional[state] += g_prop[state]
					# print(index, g,  per_robot_reward, state_values / state_number_of_visits)


		for s in range(0, self.number_of_states()):
			if state_number_of_visits[s] != 0:
				overall_state_values[s] = state_values[s] / state_number_of_visits[s]
				overall_state_proportional_values[s] = state_values_proportional[s] / state_number_of_visits[s]

		return overall_state_values, overall_state_proportional_values
	
	def calculate_proportional_vpi_for_experiment(self):
		vpi = [0.0 for i in range(0, len(self.vpi))]
		num_of_robots = float(len(self.logs))		
		per_robot_reward = self.result/float(num_of_robots)
		#print("Num of robots {0}, result {1}, reward per robot {2}, states {3}".format(num_of_robots, self.result, per_robot_reward, self.vpi))		
		timesteps = self.endIdx+1
		for episode in self.logs:
			for idx,stateContrib in enumerate(episode[4]):				
				vpi[idx] += (stateContrib/float(timesteps) * per_robot_reward)
		#		print("State {0} : stateContrib {1} vpi {2}".format(idx,stateContrib,vpi[idx] ))		
		return vpi
	
	def set_state_contribution(self, state_contrib):
		self.stateContrib = state_contrib
	
	def get_state_contribution(self):
		return self.stateContrib
	
	def update_state_counter(self, state):
		self.state_counters[state] += 1
	
	def calculate_is_ratio(self,states):
		accumulated_prob = 0;
		for episode in self.logs:
			in_episode = 1
			for idx,state in enumerate(episode[0]):
				tr_prob = episode[2]
				tr_actives = episode[3]
				if(state in states):
					in_episode *= float(tr_prob[idx]/tr_actives[idx])
					if( idx+1 < len(episode[0])):
						in_episode *= float(tr_prob[idx+1]/tr_actives[idx+1])
				
			accumulated_prob += in_episode
				
		return accumulated_prob
		
	# Calculates the state values of the new FSM using the ordinary importance sampling
	def calculate_ord_is(self,states, new_fsm):		
		number_of_states = len(self.vpi) # save the total number of states
		num_of_robots = float(len(self.logs)) # number of robots or episodes per experiment
		accumulated_prob = [0 for i in range(0,number_of_states)]; # accumulated transition probability of the original fsm
		per_robot_reward = self.result/float(num_of_robots) # reward for each robot/episode
		for episode in self.logs:
			for s in range(0, number_of_states):
				in_episode = 1 # probability for the behavior policy
				in_episode_pi = 1 # probability for the target policy
				for idx,state in enumerate(episode[0]):
					tr_prob = episode[2] # get the measured probabilities for the behavior policy 
					tr_actives = episode[3] # get the measured probabilities for the behavior policy 
					if(state in states): # if state has been removed
						next_prob_b = 1.0 # probability of the transition from state to the next state
						next_prob_new = 1.0 # assuming the prob of taking or not the transition
						if(idx+1 < len(tr_prob)): # if there is a next state
							state_a = episode[0][idx+1] # next state
							next_prob_b = float(tr_prob[idx+1]/tr_actives[idx+1]) 
							if(idx > 0):
								state_b = episode[0][idx-1] # previous state
								next_prob_new = new_fsm[state_b].prob_of_reaching_state(state_a, new_fsm,episode[5][idx]) # probability that the target policy transitions from the previous state to the next state
								
						current_prob_b = float(tr_prob[idx]/tr_actives[idx]) # probability of the transition from the previous state to state
						in_episode *=  current_prob_b * next_prob_b # combine the two
						in_episode_pi *= next_prob_new # compounds the probabilities
				if(not(s in states)):
					accumulated_prob[s] += (in_episode_pi/in_episode) * per_robot_reward # combines the state values per each episode
		
		#accumulated_prob = [i/float(len(self.logs)) for i in accumulated_prob]		
		return accumulated_prob
	
	# Calculates the state values of the new FSM using weighted importance sampling
	def calculate_weighted_is(self,states, new_fsm):		
		number_of_states = len(self.vpi) # save the total number of states
		num_of_robots = float(len(self.logs)) # number of robots or episodes per experiment
		usefull_experience = 0
		accumulated_prob = [] # accumulated transition probability of the original fsm
		accumulated_is = [] 
		first_visit_states = []
		for i in range(0,number_of_states):
			accumulated_prob.append(0.0)
			accumulated_is.append(0.0) 			
		per_robot_reward = self.result/float(num_of_robots) # reward for each robot/episode
		for episode in self.logs:			
			in_episode = 1.0 # probability for the behavior policy
			in_episode_pi = 1.0 # probability for the target policy				
			first_visit_states = [False for i in range(0,number_of_states)]
			for idx,state in enumerate(episode[0]):
				tr_prob = episode[2] # get the measured probabilities for the behavior policy 
				tr_actives = episode[3] # get the active transitions 
				if not(first_visit_states[state]):
					first_visit_states[state] = True
					
				if(state in states): # if state has been removed
					#accumulated_is[state] = 1.0
					next_prob_b = 1.0 # probability of the transition from state to the next state
					next_prob_new = 1.0 # assuming the prob of taking or not the transition
					current_prob_b = float(tr_prob[idx]/tr_actives[idx]) # probability of the transition from the previous state to state
					if(idx+1 < len(tr_prob)): # if there is a next state
						state_a = episode[0][idx+1] # next state
						next_prob_b = float(tr_prob[idx+1]/tr_actives[idx+1]) 
						if(idx > 0):
							state_b = episode[0][idx-1] # previous state
							next_prob_new = new_fsm[state_b].prob_of_reaching_state(state_a, new_fsm,episode[5][idx],episode[6][idx])/tr_actives[idx] # probability that the target policy transitions from the previous state to the next state		
							#print("State {0} to {1} prob {2}, removed states {3} next prob {4} current prob {5}".format(state_b,state_a,next_prob_new,states,next_prob_b,current_prob_b))
							
					old_in_episode = in_episode
					in_episode *=  (current_prob_b * next_prob_b) # combine the two
					in_episode_pi *= next_prob_new#/tr_actives[idx] # compounds the probabilities
					
				else: #Checks if a transtion of an active state has been deleted
					if(idx > 1 ): # if there is a state before the current one
						prev_transition = episode[1][idx] # gets the transition
						prev_state = episode[0][idx-1] # gets the previous state
						if not(new_fsm[prev_state].is_transition_active(prev_transition)):
							#print("State {0} transition {1} to {2} is {3}".format(prev_state, prev_transition, state, new_fsm[prev_state].is_transition_active(prev_transition)))
							current_prob = float(tr_prob[idx]/tr_actives[idx]) # the measured probability of coming to the current state
							prob_new = new_fsm[prev_state].prob_of_reaching_state(state, new_fsm,episode[5][idx],episode[6][idx]) # the probability without the deactivated transition
							in_episode *= current_prob #update mu
							in_episode_pi *= prob_new #update pi					
							
			if in_episode == 0:
				in_episode = 1.0
	
			for s in range(0, number_of_states):				
				if(not(s in states) and first_visit_states[s]):		
					accumulated_prob[s] += (in_episode_pi/in_episode) *  per_robot_reward # combines the state values per each episode
					#if(s==0 and in_episode_pi>0):
					#	print("Con {0} inep {1} ratio {2} total rew {3} \n {4}".format(in_episode_pi,in_episode,(in_episode_pi/in_episode),per_robot_reward,episode[0]))
				accumulated_is[s] += (in_episode_pi/in_episode)
			
			if(in_episode_pi > 0):
				usefull_experience += 1
		
		#print("Episode accumulated_prob {0} / {1} ".format(accumulated_prob,accumulated_is))
		#accumulated_prob = [i/float(len(self.logs)) for i in accumulated_prob]		
		
		return accumulated_prob,accumulated_is,usefull_experience
	
	# Calculates the state values of the new FSM using the ordinary and weighted importance sampling with proportional reward
	def calculate_proportional_weighted_is(self,states, new_fsm):		
		timesteps = self.endIdx+1		
		number_of_states = len(self.vpi) # save the total number of states
		num_of_robots = float(len(self.logs)) # number of robots or episodes per experiment
		accumulated_prob = [0 for i in range(0,number_of_states)]; # accumulated transition probability of the original fsm
		accumulated_is = [0 for i in range(0,number_of_states)];
		per_robot_reward = self.result/float(num_of_robots) # reward for each robot/episode
		for episode in self.logs:			
			in_episode = 1.0 # probability for the behavior policy
			in_episode_pi = 1.0 # probability for the target policy
			first_visit_states = [False for i in range(0,number_of_states)]
			for idx,state in enumerate(episode[0]):
				tr_prob = episode[2] # get the measured probabilities for the behavior policy 
				tr_actives = episode[3] # get the number of active transitions for the behavior policy
				tr_neighbors = episode[5]
				tr_ground = episode[6]				
				if not(first_visit_states[state]):
					first_visit_states[state] = True # if a state does not appear in the history it does not get a reward
					
				if(state in states): # if state has been removed
					#accumulated_is[state] = 1.0
					next_prob_b = 1.0 # probability of the transition from state to the next state
					next_prob_new = 1.0 # assuming the prob of taking or not the transition
					if(idx+1 < len(tr_prob)): # if there is a next state
						state_a = episode[0][idx+1] # next state
						next_prob_b = float(tr_prob[idx+1]/tr_actives[idx+1]) 
						if(idx > 0):
							state_b = episode[0][idx-1] # previous state		
							next_prob_new = new_fsm[state_b].prob_of_reaching_state(state_a, new_fsm,tr_neighbors[idx],tr_ground[idx]) # probability that the target policy transitions from the previous state to the next state	
							#if(state_b != 4 and state_a != state_b):	
							#	print("Calculated prob to jump S{0} from S{1}[{4} {5}] to S{2} to {3}".format(state,state_b,state_a,next_prob_new,new_fsm[state_b].id,new_fsm[state_b].original_id))
					current_prob_b = float(tr_prob[idx]/tr_actives[idx]) # probability of the transition from the previous state to state
					in_episode *=  current_prob_b * next_prob_b # combine the two
					in_episode_pi *= next_prob_new # compounds the probabilities
				else: #Checks if a transtion of an active state has been deleted
					if(idx > 1 ): # if there is a state before the current one
						prev_transition = episode[1][idx] # gets the transition
						prev_state = episode[0][idx-1] # gets the previous state
						if not(new_fsm[prev_state].is_transition_active(prev_transition)):
							#print("State {0} transition {1} to {2} is {3}".format(prev_state, prev_transition, state, new_fsm[prev_state].is_transition_active(prev_transition)))
							current_prob = float(tr_prob[idx]/tr_actives[idx]) # the measured probability of coming to the current state
							prob_new = new_fsm[prev_state].prob_of_reaching_state(state, new_fsm,tr_neighbors[idx],tr_ground[idx]) # the probability without the deactivated transition
							in_episode *= current_prob #update mu
							in_episode_pi *= prob_new #update pi
		
			if in_episode == 0:
				in_episode = 1.0
	
			for s in range(0, number_of_states):	
				if(not(s in states) and first_visit_states[s]):		
					state_reward = (episode[4][s]/float(timesteps) * per_robot_reward)
					episode_val = (in_episode_pi/in_episode) *  state_reward			
					accumulated_prob[s] += episode_val # combines the state values per each episode
					#is_c = in_episode_pi/in_episode
					#if s == 2 and is_c > 0 :
					#	print("Acc {0} : {1} / {2} = {3} {5}\n {4}".format(episode_val, in_episode_pi, in_episode, is_c, episode[0],state_reward))
				accumulated_is[s] += (in_episode_pi/in_episode)		
		#accumulated_prob = [i/float(len(self.logs)) for i in accumulated_prob]		
		return accumulated_prob,accumulated_is
	
	def __repr__(self):
		return self.logs
	
	def __str__(self):
		return "Exp length : {0} #robots : {1} vpi {2}".format(len(self.logs), self.result, self.calculate_vpi_for_experiment())
	
	def number_of_states(self):
		return len(self.vpi)


	def estimateValueStatesUsingImportanceSamplingWithIntermediate(self, old_fsm, new_fsm, number_of_episodes, experiments, first_visit=True, discount_factor=0.98):
		timesteps = self.endIdx+1
		num_of_robots = float(len(self.logs)) # number of robots or episodes per experiment
		weighted_importance_sampling_state_estimation = np.zeros(self.number_of_states())
		ordinary_importance_sampling_state_estimation = np.zeros(self.number_of_states())
		proportional_weighted_importance_sampling_state_estimation = np.zeros(self.number_of_states())
		proportional_ordinary_importance_sampling_state_estimation = np.zeros(self.number_of_states())

		nominator = [] #np.zeros(self.number_of_states())
		weight_denominator = [] #np.zeros(self.number_of_states())
		ordinary_denominator = [] #np.zeros(self.number_of_states())
		proportional_nominator = []
		
		for s in range(0, self.number_of_states()):		
			nominator.append(mpfr(0))
			weight_denominator.append(mpfr(0))
			ordinary_denominator.append(mpfr(0))
			proportional_nominator.append(mpfr(0))

		for episode in self.logs: 
			importance_sampling_coefficient = mpfr(1.0)
			
			state_proportional_rewards = np.zeros(self.number_of_states())

			g = mpfr(0)
			g_prop = [] #np.zeros(self.number_of_states())
			for s in range(0, self.number_of_states()):		
				g_prop.append(mpfr(0))
			
			ordinary_denominator = np.zeros(self.number_of_states())
			visited = np.zeros(self.number_of_states())

			for index, state in reversed(list(enumerate(episode[7][:-1]))):
				per_robot_reward = self.objFun[index] / float(num_of_robots) # reward for each robot/episode
				for s in range(0, self.number_of_states()):				
					state_proportional_rewards[s] = (episode[4][s]/float(timesteps) * per_robot_reward)	

				next_state = episode[7][index + 1] # previous state
				tr_neighbors = episode[9]
				tr_ground = episode[8]

				behavior_prob_transition = old_fsm[state].prob_of_reaching_state(next_state, old_fsm, tr_neighbors[index + 1], tr_ground[index + 1])
				target_prob_transition = new_fsm[state].prob_of_reaching_state(next_state, new_fsm,tr_neighbors[index + 1],tr_ground[index + 1]) # probability that the target policy transitions from the previous state to the current state

				if target_prob_transition == 0:
					target_prob_transition = sys.float_info.epsilon

				if behavior_prob_transition == 0:
					behavior_prob_transition = sys.float_info.epsilon

				importance_sampling_coefficient *= (mpfr(target_prob_transition) / mpfr(behavior_prob_transition))

				g = mpfr(g * discount_factor + importance_sampling_coefficient * per_robot_reward)
				g_prop[state] = mpfr(g_prop[state] * discount_factor + importance_sampling_coefficient * state_proportional_rewards[state])

				if per_robot_reward != 0 and (not state in episode[7][0:index] or not first_visit):
					nominator[state] += g

					weight_denominator[state] += importance_sampling_coefficient
					proportional_nominator[state] += g_prop[state] 
					ordinary_denominator[state] += 1

				visited[state] = 1

		for s in range(0, self.number_of_states()):
			proportional_weighted_importance_sampling_estimation = (proportional_nominator[s] / weight_denominator[s]) if weight_denominator[s] > 0 else 0
			proportional_ordinary_importance_sampling_estimation = (proportional_nominator[s] / ordinary_denominator[s]) if ordinary_denominator[s] > 0 else 0
			weighted_importance_sampling_estimation =  (nominator[s] / weight_denominator[s]) if weight_denominator[s] else 0
			ordinary_importance_sampling_estimation = nominator[s] / ordinary_denominator[s] if ordinary_denominator[s] > 0 else 0

			# if not np.isnan(weighted_importance_sampling_estimation):
			weighted_importance_sampling_state_estimation[s] = weighted_importance_sampling_estimation

			# if not np.isnan(ordinary_importance_sampling_estimation):
			ordinary_importance_sampling_state_estimation[s] = ordinary_importance_sampling_estimation

			# if not np.isnan(proportional_weighted_importance_sampling_estimation):
			proportional_weighted_importance_sampling_state_estimation[s] = proportional_weighted_importance_sampling_estimation

			# if not np.isnan(proportional_ordinary_importance_sampling_estimation):
			proportional_ordinary_importance_sampling_state_estimation[s] = proportional_ordinary_importance_sampling_estimation

		return weighted_importance_sampling_state_estimation, ordinary_importance_sampling_state_estimation, proportional_weighted_importance_sampling_state_estimation, proportional_ordinary_importance_sampling_state_estimation



	def estimateValueStatesUsingImportanceSampling(self, old_fsm, new_fsm, number_of_episodes, experiments, first_visit=True, discount_factor=0.98):
		timesteps = self.endIdx+1
		num_of_robots = float(len(self.logs)) # number of robots or episodes per experiment
		weighted_importance_sampling_state_estimation = np.zeros(self.number_of_states())
		ordinary_importance_sampling_state_estimation = np.zeros(self.number_of_states())
		proportional_weighted_importance_sampling_state_estimation = np.zeros(self.number_of_states())
		proportional_ordinary_importance_sampling_state_estimation = np.zeros(self.number_of_states())

		for episode in self.logs:
			importance_sampling_coefficient = 1.0
			proportional_nominator = np.zeros(self.number_of_states())
			state_proportional_rewards = np.zeros(self.number_of_states())

			nominator = [] #np.zeros(self.number_of_states())
			weight_denominator = [] #np.zeros(self.number_of_states())
			ordinary_denominator = [] #np.zeros(self.number_of_states())
			per_robot_reward = self.result/float(num_of_robots) # reward for each robot/episode
			visited = np.zeros(self.number_of_states())

			for s in range(0, self.number_of_states()):				
				state_proportional_rewards[s] = (episode[4][s]/float(timesteps) * per_robot_reward)	
				nominator[s] = mpfr(0)
				weight_denominator[s] = mpfr(0)
				ordinary_denominator[s] = mpfr(0)

			for index, state in reversed(list(enumerate(episode[0]))):
				if index == 0:
					break

				previous_state = episode[0][index - 1] # previous state
				tr_neighbors = episode[5]
				tr_ground = episode[6]

				behavior_prob_transition = old_fsm[previous_state].prob_of_reaching_state(state, old_fsm, tr_neighbors[index], tr_ground[index])
				target_prob_transition = new_fsm[previous_state].prob_of_reaching_state(state, new_fsm,tr_neighbors[index],tr_ground[index]) # probability that the target policy transitions from the previous state to the current state

				if target_prob_transition == 0:
					target_prob_transition = sys.float_info.epsilon
				
				if behavior_prob_transition == 0:
					behavior_prob_transition = sys.float_info.epsilon
				
				importance_sampling_coefficient *= (mpfr(target_prob_transition) / behavior_prob_transition)

				if visited[previous_state] == 0 or not first_visit:
					nominator[previous_state] += importance_sampling_coefficient * per_robot_reward
					proportional_nominator[previous_state] += importance_sampling_coefficient * state_proportional_rewards[previous_state]
					weight_denominator[previous_state] += importance_sampling_coefficient
					ordinary_denominator[previous_state] += 1

				visited[previous_state] = 1
				per_robot_reward *= discount_factor
				state_proportional_rewards *= discount_factor

			for s in range(0, self.number_of_states()):
				proportional_weighted_importance_sampling_estimation = (proportional_nominator[s] / weight_denominator[s]) if weight_denominator[s] > 0 else 0
				proportional_ordinary_importance_sampling_estimation = (proportional_nominator[s] / ordinary_denominator[s]) if ordinary_denominator[s] > 0 else 0
				weighted_importance_sampling_estimation =  (nominator[s] / weight_denominator[s]) if weight_denominator[s] else 0
				ordinary_importance_sampling_estimation = nominator[s] / ordinary_denominator[s] if ordinary_denominator[s] > 0 else 0

				if not np.isnan(weighted_importance_sampling_estimation):
					weighted_importance_sampling_state_estimation[s] += weighted_importance_sampling_estimation

				if not np.isnan(ordinary_importance_sampling_estimation):
					ordinary_importance_sampling_state_estimation[s] += ordinary_importance_sampling_estimation

				if not np.isnan(proportional_weighted_importance_sampling_estimation):
					proportional_weighted_importance_sampling_state_estimation[s] += proportional_weighted_importance_sampling_estimation
				
				if not np.isnan(proportional_ordinary_importance_sampling_estimation):
					proportional_ordinary_importance_sampling_state_estimation[s] += proportional_ordinary_importance_sampling_estimation

		return weighted_importance_sampling_state_estimation / len(self.logs),  ordinary_importance_sampling_state_estimation / len(self.logs), proportional_weighted_importance_sampling_state_estimation / len(self.logs), proportional_ordinary_importance_sampling_state_estimation / len(self.logs)
				

	#
	# This function computes the importance sampling terms needed for WIS and OIS with proportional and regular 
	# reward calculation for two FSM with identical structure (same states and transitions) but with a different
	# parameter configuration
	##
	def parameters_analysis_importance_sampling(self, old_fsm, new_fsm):
		timesteps = self.endIdx+1		
		number_of_states = len(self.vpi) # save the total number of states
		num_of_robots = float(len(self.logs)) # number of robots or episodes per experiment
		usefull_experience = 0
		per_robot_reward = self.result/float(num_of_robots) # reward for each robot/episode
		wis      = []
		wis_den  = []
		pwis     = []	

		for idx in self.vpi:
			wis.append(0.0)
			wis_den.append(0.0)
			pwis.append(0.0)
		for episode in self.logs:
#			in_episode = 1.0 # probability for the behavior policy
#			in_episode_pi = 1.0 # probability for the target policy
			is_coef = mpfr('1.0',100) # importance sampling coefficient
			first_visit_states = [False for i in range(0,number_of_states)]
			for idx,state in enumerate(episode[0]):
				tr_prob = episode[2] # get the measured probabilities for the behavior policy 
				tr_actives = episode[3] # get the number of active transitions for the behavior policy
				tr_neighbors = episode[5]
				tr_ground = episode[6]		

				# print("tr_neighbors, tr_ground", tr_neighbors, tr_ground, idx)

				if not(first_visit_states[state]):
					first_visit_states[state] = True # if a state does not appear in the history it does not get a reward
					#accumulated_is[state] = 1.0
					#prob_transition = float(tr_prob[idx]/tr_actives[idx])# probability of the transition from the previous state to the current state
				prob_transition_target = 1.0 # same probability for the target FSM
				prob_transition = 1.0
				if(idx > 0):
					previous_state = episode[0][idx-1] # previous state		

					prob_transition = old_fsm[previous_state].prob_of_reaching_state(state, old_fsm,tr_neighbors[idx],tr_ground[idx])

					#prob_transition_p = old_fsm[previous_state].get_transition_probability(episode[1][idx],tr_neighbors[idx],tr_ground[idx])/tr_actives[idx]
					#prob_transition = float(tr_prob[idx]/tr_actives[idx]) # the measured probability of coming to the current state
					# transition probability target FSM
					prob_transition_target = new_fsm[previous_state].prob_of_reaching_state(state, new_fsm,tr_neighbors[idx],tr_ground[idx]) # probability that the target policy transitions from the previous state to the current state
					#prob_transition_target_p = new_fsm[previous_state].get_transition_probability(episode[1][idx],tr_neighbors[idx],tr_ground[idx])/tr_actives[idx]

					# print("Transition from state2", previous_state, "to", state, is_coef, tr_neighbors[idx], tr_ground[idx], new_fsm,tr_neighbors[idx],tr_ground[idx])												
				if prob_transition != prob_transition_target:	
					#print("Transition from {0} to {1} condition {2} : old probability {3} ( {4} ) / new probability  {5} ( {6} )".format(previous_state,state,episode[1][idx],prob_transition,prob_transition_p,prob_transition_target,prob_transition_target_p))				
					is_coef = is_coef * mpfr(prob_transition_target)/(prob_transition)

					
					#print("is_coef : {0}".format(is_coef))			

			for s in range(0, number_of_states):				
				if first_visit_states[s]:		
					state_proportional_reward = (episode[4][s]/float(timesteps) * per_robot_reward)	
					episode_val = is_coef *  state_proportional_reward			
					pwis[s] += episode_val # combines the state values per each episode
					wis[s]  += is_coef *  per_robot_reward # combines the state values per
					#print("Episode end state {0} : wis {1} IS coef {2} ".format(s,wis[s],is_coef))
					
				wis_den[s] += is_coef
		
			#if(in_episode_pi > 0):
			usefull_experience += 1
		
		# print("wis", wis)
		return wis,wis_den,pwis,usefull_experience