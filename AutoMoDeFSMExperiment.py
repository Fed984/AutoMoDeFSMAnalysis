import Tokenizer

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
		self.logs = []
		self.metrics = []
		self.startIdx = 0
		self.endIdx = 0
		self.vpi = []
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
	
	def calculate_vpi_for_experiment(self):
		vpi = []
		num_of_robots = float(len(self.logs))		
		per_robot_reward = self.result/float(num_of_robots)
		#print("Num of robots {0}, result {1}, reward per robot {2}, states {3}".format(num_of_robots, self.result, per_robot_reward, self.vpi))
		for count in self.vpi:
			vpi.append((count * per_robot_reward)/num_of_robots)
		
		return vpi
	
	def calculate_is_ratio(self,states):
		accumulated_prob = 0;
		for episode in self.logs:
			in_episode = 1
			for idx,state in enumerate(episode[0]):
				tr_prob = episode[2]
				tr_actives = episode[3]
				if(state in states):
					in_episode *= float(tr_prob[idx]/tr_actives[idx]) * float(tr_prob[idx+1]/tr_actives[idx+1])
				
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
						next_prob_new = 0.5 # assuming the prob of taking or not the transition
						if(idx+1 < len(tr_prob)): # if there is a next state
							state_a = episode[0][idx+1] # next state
							next_prob_b = float(tr_prob[state_a]/tr_actives[state_a]) 
							if(idx > 0):
								state_b = episode[0][idx-1] # previous state
								next_prob_new = new_fsm[state_b].prob_of_reaching_state(state_a, new_fsm) # probability that the target policy transitions from the previous state to the next state
								
						current_prob_b = float(tr_prob[idx]/tr_actives[idx]) # probability of the transition from the previous state to state
						in_episode *=  current_prob_b * next_prob_b # combine the two
						in_episode_pi *= next_prob_new # compounds the probabilities
				if(not(s in states)):
					accumulated_prob[s] += (in_episode_pi/in_episode) * per_robot_reward # combines the state values per each episode
		
		#accumulated_prob = [i/float(len(self.logs)) for i in accumulated_prob]		
		return accumulated_prob
	
	# Calculates the state values of the new FSM using the ordinary importance sampling
	def calculate_weighted_is(self,states, new_fsm):		
		number_of_states = len(self.vpi) # save the total number of states
		num_of_robots = float(len(self.logs)) # number of robots or episodes per experiment
		accumulated_prob = [0 for i in range(0,number_of_states)]; # accumulated transition probability of the original fsm
		accumulated_is = [0 for i in range(0,number_of_states)];
		per_robot_reward = self.result/float(num_of_robots) # reward for each robot/episode
		for episode in self.logs:
			for s in range(0, number_of_states):
				in_episode = 1 # probability for the behavior policy
				in_episode_pi = 1 # probability for the target policy
				for idx,state in enumerate(episode[0]):
					tr_prob = episode[2] # get the measured probabilities for the behavior policy 
					tr_actives = episode[3] # get the measured probabilities for the behavior policy 
					if(state in states): # if state has been removed
						accumulated_is[state] = 1.0
						next_prob_b = 1.0 # probability of the transition from state to the next state
						next_prob_new = 0.5 # assuming the prob of taking or not the transition
						if(idx+1 < len(tr_prob)): # if there is a next state
							state_a = episode[0][idx+1] # next state
							next_prob_b = float(tr_prob[state_a]/tr_actives[state_a]) 
							if(idx > 0):
								state_b = episode[0][idx-1] # previous state
								next_prob_new = new_fsm[state_b].prob_of_reaching_state(state_a, new_fsm) # probability that the target policy transitions from the previous state to the next state			
						current_prob_b = float(tr_prob[idx]/tr_actives[idx]) # probability of the transition from the previous state to state
						in_episode *=  current_prob_b * next_prob_b # combine the two
						in_episode_pi *= next_prob_new # compounds the probabilities
				if(not(s in states)):
					accumulated_prob[s] += (in_episode_pi/in_episode) * per_robot_reward # combines the state values per each episode
					accumulated_is[s] += (in_episode_pi/in_episode)
			
		#accumulated_prob = [i/float(len(self.logs)) for i in accumulated_prob]		
		return accumulated_prob,accumulated_is
	
	def __repr__(self):
		return self.logs
	
	def __str__(self):
		return "Exp length : {0} #robots : {1} vpi {2}".format(len(self.logs), self.result, self.calculate_vpi_for_experiment())
	
		