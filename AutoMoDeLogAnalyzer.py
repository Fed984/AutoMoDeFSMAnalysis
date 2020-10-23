#!/usr/bin/python

# FSM grammar
# starting token -> --fsm-config --states <integer>
# state token -> --s<integer>
# transition token -> --n<integer>x<integer> --c<integer>x<integer> [parameters of transition]
# <FSM> ::= <S0>
# <S0> ::= <State> <S0> | emptyset
# <State> ::= --sK <integer> <Parameters> <Transitions>
# <Parameters> ::= <Parameter> <Parameters> | emptyset
# <Parameter> ::= --*[A-z]K <integer> 
# <Transitions> ::= <Transition> <Transitions> | emptyset
# <Transition> ::= --nKxY W --cKxY <Parameters>

# Log data
# --t iteration 
# --sK <integer> 
# K <- index, for states is the state number ( from zero to nstates) 
# <integer> <-- the state type
# --cK <integer> <boolean> conditions that led to the new current state
# K <- index, same as for states but for conditions it is specific to the state
# <integer> <-- condition type
# <boolean> <-- if the condition is verified or not

#Scenario file
#trainInstancesDir = "./experiments-folder" <-- directory with the instances
#trainInstanceFile = "...." <-- file with the instance list

# Creation of the Behaviour object
#			case 0: cNewBehaviour = new AutoMoDeBehaviourExploration();
#			case 1: cNewBehaviour = new AutoMoDeBehaviourStop();
#			case 2: cNewBehaviour = new AutoMoDeBehaviourPhototaxis();
#			case 3: cNewBehaviour = new AutoMoDeBehaviourAntiPhototaxis();
#			case 4:	cNewBehaviour = new AutoMoDeBehaviourAttraction();
#			case 5:	cNewBehaviour = new AutoMoDeBehaviourRepulsion();
# Transition conditions
#			case 0: cNewCondition = new AutoMoDeConditionBlackFloor();
#			case 1: cNewCondition = new AutoMoDeConditionGrayFloor();
#			case 2: cNewCondition = new AutoMoDeConditionWhiteFloor();
#			case 3: cNewCondition = new AutoMoDeConditionNeighborsCount();
#			case 4: cNewCondition = new AutoMoDeConditionInvertedNeighborsCount();
#			case 5: cNewCondition = new AutoMoDeConditionFixedProbability();

import Tokenizer		
import re
import sys
import AutoMoDeStateParser
import os
import subprocess
import random
from scipy.stats import wilcoxon
import AutoMoDeFSMExperiment
import copy
from tqdm import tqdm

#
#  Presentation code
###################################################### 
commandline_separator = "-------------------------------------------------------------------------------------"	
# print how to use the script
def command_usage():
	print("Usage   : \n\t AutoMoDeLogAnalyzer historyfile_path [options] --fsm-config <finite state machine description>".format(sys.argv[0]))
	print("Options : ")
	print("\t --threshold VALUE or -t VALUE")
	print("\t\t Set the threshold used to decide wether to delete a state.")
	print("\t\t The value represent a percentage that should go from 0 (the default value,")
	print("\t\t a state is delete if it is never accessed) to 1 (all the states are deleted).\n")	
	print("\t --test or -te")	
	print("\t\t When used the script will run RUN experiments with the original and pruned FSM.")
	print("\t\t The results are compared using the Wilcoxon paired statistical test.\n")
	print("\t --scenario FILE or -sc FILE")
	print("\t\t The script will run the experiments using the instances specified in the irace.")
	print("\t\t scenario file FILE.\n")
	print("\t --targetrunner FILE or -ta FILE") 
	print("\t\t The script will use the target-runner FILE to run the experiments.\n")
	print("\t --runs RUN or -ru RUN")
	print("\t\t The script will run RUN experiments before the statistical test.\n")
	print("\t --rseed SEED or -rs SEED")
	print("\t\t The random seed used to initialize the random seed generator.\n")
	print("\t --keep-transitions or -kt")
	print("\t\t The transitions towards a removed state are kepts and rerouted.\n")
	print("\t --no-is-analysis or -na")
	print("\t\t The performance estimation based on importance sampling is deactivated.\n")
	print("\t --pruning or -p")
	print("\t\t The states of the FSM that are active for less than threshold will be eliminated.\n")
	print("\t --all-state-analysis or -as")
	print("\t\t The importance of each state will be estimated using the importance sampling analysis.\n")
	print("\t --param-analysis or -pa --newfsm-config NEWFSM")
	print("\t\t This option will compare NEWFSM with the FSM specified later with --fsm-config or in the log file.")
	print("\t\t NEWFSM has different parameters but the same states and transitions ")
	print("\t\t as the FSM that generated the log.\n")
	print("\t --help or -h")
	print("\t\t Prints this help.\n")


#
# This function returns the string representation of a finite state machine as
# the set of parameters accepted from AutoMoDe
## 
def print_fsm(fsm_log_counter):
	cfsm = ""
	for state in fsm_log_counter:
		if(state.active):					
			cfsm += str(state)+" "

	cfsm = "--nstates {0}".format(len(fsm_log_counter))+" "+cfsm
	
	return cfsm

#
#  Analysis functions
###################################################### 
	
# the states_map represents the transition map for the states
# this function updates the list when a state is removed so that
# the transitions points to the right states
def update_states_map(states_map, nstates, remove_state):
	new_idx = 0
	new_nstates = nstates-1
	max_idx = nstates-2
	for i in range(0,len(states_map)):			
		states_map[i] = new_idx
		if(i != remove_state and new_idx+1 < max_idx):		
			new_idx = new_idx + 1
			
	return states_map	
	
# Reads the scenario.txt file to get the training instances
# that will be used to test the prunedFSM	
def read_scenario_file(scenario_txt):
	print("Opening scenario...")
	f = open(scenario_txt,"r")
	instances=[]
	lsplit = ""
	lines = f.readlines();
	for line in lines: # Goes through the file to get the path to the instance/instances
		if(line.startswith("trainInstancesDir")): # if the instances are in a directory
			lsplit = line.split("=")
			path=lsplit[1].split("\"")[1]
			if(path.startswith(".")):
				base_path = os.path.dirname(os.path.abspath(scenario_txt))				
				path = os.path.join(base_path, path)
			# Do something here
			for r, d, f in os.walk(path):
				for file in f:
					instances.append(os.path.join(r, file))					
					
			break
		elif(line.startswith("trainInstancesFile")): # if the instances are specified in a file
			lsplit = line.split("=").split("\"")[1]
			path = lsplit[1]
			if(path.startswith(".")):
				base_path = os.path.dirname(os.path.abspath(scenario_txt))				
				path = os.path.join(base_path, path)

			ifile = open(path)
			iflines = ifile.readlines()
			for ins in iflines:
				instances.append(ins)
			# Do something else			
			break
			
	if(len(instances)==0):
		print("no instances found in {0}".format(lsplit[1]))
	else:
		print("{0} instances found in {1}".format(len(instances),lsplit[1]))
		
	return instances
	
# Run the experiments for the naive comparison and compares the results with the wilcoxon pairwise test 
# the function returns the pvalue
def execute_experiments(max_runs, instances, original_fsm, pruned_fsm, default_target_runner, rseed=1):
	instances_num = len(instances)         # get the number of instances
	random_seeds = int(max_runs/instances_num)  # number of random seeds to generate for max_runs
	results_original = []
	results_pruned = []
	random.seed(rseed)
	avg_res_original = 0.0
	avg_res_pruned = 0.0
	for ins in instances:
		for r in range(0, random_seeds):			
			seed = str(random.randint(0,pow(10,9))) #generate random seed
			print("Running {0} with seed {1}".format(ins, seed))
			ro = float(subprocess.check_output([default_target_runner,"0","0",seed,ins,original_fsm]))
			avg_res_original += ro
			results_original.append(ro)
			rp = float(subprocess.check_output([default_target_runner,"0","0",seed,ins,pruned_fsm]))
			avg_res_pruned += rp
			results_pruned.append(rp)
	
	avg_res_original = avg_res_original/max_runs*-1
	avg_res_pruned = avg_res_pruned/max_runs*-1	
	print("Results original        : {0}".format(results_original))
	print("Average result original : {0}".format(avg_res_original))
	print("Results pruned          : {0}".format(results_pruned))	
	print("Average result pruned   : {0}".format(avg_res_pruned))
	#Check that the results are not exactly the same
	test=False
	for idx,val in enumerate(results_original):
		if(val != results_pruned[idx]):
			test=True
			break
			
	#Calculate stat test
	if(test):
		print("\n Executing Wilcoxon test")
		print(commandline_separator)
		stat, p = wilcoxon(results_original, results_pruned)
		print('stat=%.3f, p=%.3f' % (stat, p))
		if p < 0.05:
			print('There is a significant difference')
		else:
			print('There is no significant difference')
			
	else:
		p=0
		print("The two FSM reported exactly the same results")
			
	return p,results_original,results_pruned
		
# Reads and analyzes the FSM history file 
# returns the total number of time ticks read
def analyze_logfile(history_file, fsm_log_counter, experiments):
	number_of_ticks = 0
	number_of_episodes= 0
	number_of_states = len(fsm_log_counter)
	print("\n Opening log file")
	print(commandline_separator)
	f = open(history_file,"r")
	Lines = f.readlines();
	print("Number of lines      : {0}".format(len(Lines)))
	previous_state = 0
	cstate = 0
	#print("Analyzing log file   :")	
	recording_exp = False	
	exp_states = [0] 
	exp_transitions = [0]
	exp_transitions_probabilities = [1]
	exp_active_transitions = [1]
	exp_state_contribution = [0 for i in range(0,number_of_states)]
	exp_ground = [0]
	exp_neighbors = [0]
	cscore = 0
	r_recording = False
	vpi_overall = [0.0 for i in range(0,number_of_states)]
	#state_contribution = [0.0 for i in range(0,number_of_states)]
	vpi_states = [0 for i in range(0,number_of_states)]	
	episode_tick = 0
	for line in tqdm(Lines,desc="Analyzing log file   "):
		if(not(recording_exp) and line.startswith("Score ")):	  # Starting of the experiment		
					cscore = float(line.split()[1])   # Getting the score 
					exp = AutoMoDeFSMExperiment.AutoMoDeExperiment() #Initializing the experiment object	
					exp.set_result(cscore)
					recording_exp = True              # Activating the recording
		elif(recording_exp and not(line.startswith("[INFO]"))):	  # if reading the experiment traces
			t = Tokenizer.Tokenizer(line) # splitting the traces in tokens
			previous_state = cstate # saving the previous state
			while t.has_more_tokens():  
				ctoken = t.next_token()						
				if(ctoken.startswith("--t")):   #if token is a clock tick, update the clock ticks counter
					ctick = int(t.next_token())
					number_of_ticks += 1
					episode_tick += 1
					#len(exp_states) > 1
					if(ctick == 0 and episode_tick > 1):	#Saving the experience for each robot
						episode_tick = 0
						#print("Saving EPISODE ROBOT {0} OF EXPERIMENT {1} vpi {2} # {3}".format(number_of_episodes,len(experiments),vpi_overall,len(exp_states)))
						exp.set_startIdx(ctick)	# saving start tick (not usefull since ticks reset)
						exp_log = [exp_states, exp_transitions, exp_transitions_probabilities, exp_active_transitions,exp_state_contribution,exp_neighbors,exp_ground]		
						exp.append_logs(exp_log) # save logs of the robot
						vpi_states = [0]
						exp_state_contribution = [0]
						for i in range(1,number_of_states):
							vpi_states.append(0) #reinitialize
							exp_state_contribution.append(0)
							
						exp_states = [0]                    # reinitialize
						exp_transitions = [0]               # reinitialize
						exp_transitions_probabilities = [1] # reinitialize
						exp_active_transitions = [1]        # reinitialize
						exp_neighbors = [0]                 # reinitialize	
						exp_ground = [0]                    # reinitialize	 				
					if(ctick == 0): #Updating the total number of robots
						number_of_episodes += 1						
				else:	
					#match = re.search("--s[0-9]+", ctoken)
					#if(match != None):				
					if(ctoken.startswith("--s")):  #if token is a state description log the state execution
						match = re.search("[0-9]+", ctoken)
						cstate = int(match.group(0));
						fsm_log_counter[cstate].increase_counter_state() #Updating the counter state
						exp_state_contribution[cstate] += 1
						if( vpi_states[cstate] == 0): #If this state was not visited by this robot
							vpi_states[cstate] = 1
							vpi_overall[cstate] += 1 # update the Experiment First Visit count 
						#print("Current State {} of type {}".format(cstate,t.next_token()))
			
					#match = re.search("--c[0-9]+", ctoken)		
					#if(match != None):
					elif(ctoken.startswith("--c")): #if token is a condition, log only if the condition is true						
						match = re.search("[0-9]+", ctoken)
						condition = int(match.group(0))
						type = t.next_token(); # type of the transition
						value = int(t.next_token()) # value (with the new code is always 1)
						probability = float(t.next_token()) # probability that the transition was active
						active_transitions = 1
						neighbors = 0
						gsensor = -1
						if(t.peek().startswith("--a")): # if the active states log is present
							t.next_token()
							active_transitions = int(t.next_token()) #number of active transitions
						if(t.peek().startswith("--n")): # if the number of neighbors is logged
							t.next_token()
							neighbors = int(t.next_token())
						if(t.peek().startswith("--f")): # if the ground sensor is logged
							t.next_token()
							gsensor = t.getFloat()
							
						if(value == 1): #if the condition is true
							fsm_log_counter[previous_state].increase_counter_transition(condition)
							exp_states.append(cstate) #update the states log
							exp_transitions.append(condition) #update the transitions log
							exp_transitions_probabilities.append(probability)# probabilities
							exp_active_transitions.append(active_transitions)# active transitions
							exp_neighbors.append(neighbors)
							exp_ground.append(gsensor)
							#print("Condition {} of type {} has value {} belongs to state {}".format(condition,type,value,previous_state))	
		elif(recording_exp and line.startswith("[INFO]")): # Beginning of a new experiment 
			recording_exp = False    # stop recording
			exp_log = [exp_states, exp_transitions, exp_transitions_probabilities, exp_active_transitions, exp_state_contribution,exp_neighbors,exp_ground]
			exp.append_logs(exp_log) # save experiment log
			exp.set_vpi(vpi_overall) # save overall vpi
			experiments.append(exp)	 # save experiment
			exp.set_endIdx(ctick)	 # saving final time tick (not usefull since ticks reset per each robot)
			vpi_overall = [0.0]
			vpi_states = [0]
			exp_state_contribution = [0]
			episode_tick = 0
			for i in range(1,number_of_states):
				vpi_overall.append(0.0)          # reinitialize 				
				vpi_states.append(0)             # reinitialize
				exp_state_contribution.append(0)
			
			exp_states = [0] 			 # reinitialize
			exp_transitions = [0] 			  # reinitialize
			exp_transitions_probabilities = [1] 	  # reinitialize
			exp_active_transitions = [1] 		  # reinitialize	
			exp_neighbors = [0]
			exp_ground = [0]
			cscore = 0 				  # reinitialize
	
	if(len(exp_states) > 0): # Save the final experiment
		exp_log = [exp_states, exp_transitions, exp_transitions_probabilities, exp_active_transitions, exp_state_contribution,exp_neighbors,exp_ground]
		exp.append_logs(exp_log)  # save experiment log
		exp.set_vpi(vpi_overall)  # save overall vpi
		#print("Saving EPISODE ROBOT {0} OF EXPERIMENT {1}".format(number_of_robots,len(experiments)))
		exp.set_endIdx(ctick)     # saving final time tick (not usefull since ticks reset per each robot)
		experiments.append(exp)   # save experiment
		
	f.close()
	print("Number of episodes   : {0}".format(number_of_episodes))	
	return number_of_ticks,number_of_episodes

def compute_state_values(originalFSM, number_of_episodes, experiments):
	number_of_states = len(originalFSM)
	vpi_all = [0.0] # the value for each state as estimated using the first visit method 
	vpi_proportional = [0.0] # In this case the reward for each state is proportional to the percentage of time each state was executed
	for i in range(1, number_of_states): # Set to 0 for each state
		vpi_all.append(0.0)
		vpi_proportional.append(0.0)
		
	for idx,ex in enumerate(experiments): #Go through each Experiment to calculate and collect the values
		vpi_part = ex.calculate_vpi_for_experiment()
		vpi_prop = ex.calculate_proportional_vpi_for_experiment()		
		for i in range(0,number_of_states): # Update the values for each state
			vpi_all[i]    += vpi_part[i]
			vpi_proportional[i] += vpi_prop[i]
	
	for i in range(0,number_of_states):		
		vpi_all[i] = vpi_all[i]/float(len(experiments))
		vpi_proportional[i] = vpi_proportional[i]/float(number_of_episodes)				
	
	return vpi_all, vpi_proportional

def parameters_comparison( originalFSM, newFSM, number_of_episodes, experiments):
	number_of_states = len(originalFSM)
	ois = [] # Contains the ordinary importance sampling for each state
	ois_proportional = []
	wis = [0.0] # Contains the sum of the numerator of the weighted importance sampling for each state
	wis_den = [0.0] # Contains the denominator of the wrighted importance sampling  for each state
	vpi_all = [0.0] # the value for each state as estimated using the first visit method 
	vpi_proportional = [0.0] # In this case the reward for each state is proportional to the percentage of time each state was executed
	wis_proportional = [0.0] #weighted importance sampling using the proportional reward
	usefull_exp = 0.0 # number of episodes contributing to the importance sampling	
	for i in range(1, number_of_states): # Set to 0 for each state
		wis.append(0.0)     
		wis_den.append(0.0)
		vpi_all.append(0.0)
		vpi_proportional.append(0.0)
		wis_proportional.append(0.0)
		
	for idx,ex in enumerate(experiments): #Go through each Experiment to calculate and collect the values
		pwis,pwis_den,ppwis,ue = ex.parameters_analysis_importance_sampling(originalFSM, newFSM)
		usefull_exp += ue
		vpi_part = ex.calculate_vpi_for_experiment()
		vpi_prop = ex.calculate_proportional_vpi_for_experiment()		
		for i in range(0,number_of_states): # Update the values for each state
			wis[i]     += pwis[i]
			wis_den[i] += pwis_den[i]
			vpi_all[i]    += vpi_part[i]
			vpi_proportional[i] += vpi_prop[i]
			wis_proportional[i] += ppwis[i]
	
	for i in range(0,number_of_states):
		ois.append(wis[i]/float(number_of_episodes))
		vpi_all[i] = vpi_all[i]/float(len(experiments))
		vpi_proportional[i] = vpi_proportional[i]/float(number_of_episodes)
		ois_proportional.append(wis_proportional[i]/float(number_of_episodes))
		den = wis_den[i]
		if( den == 0):
			den = 1.0
		wis[i] = wis[i]/den
		wis_proportional[i] = wis_proportional[i]/den
	
	return vpi_all, ois, wis, vpi_proportional, wis_proportional, usefull_exp, ois_proportional

# Computes and collects the importance sampling	analysis over all the experiments
# registered in the log file for two FSM where one was derived from the first
# by removing one or more states and/or transitions
def importance_sampling_analysis(fsm_log_counter, removed_states, number_of_episodes, experiments):
	number_of_states = len(fsm_log_counter) # the number of states in the original FSM
	is_ratio = 0.0 # collects the is ratio as calculated by the calculate_is_ratio method of the Experiment class
	ord_is = [] # Contains the ordinary importance sampling for each state
	wei_is = [0.0] # Contains the sum of the numerator of the weighted importance sampling for each state
	wei_is_den = [0.0] # Contains the denominator of the wrighted importance sampling  for each state
	vpi_all = [0.0] # the value for each state as estimated using the first visit method 
	vpi_proportional = [0.0] # In this case the reward for each state is proportional to the percentage of time each state was executed
	wei_is_proportional = [0.0] #weighted importance sampling using the proportional reward
	wei_is_den_proportional = [0.0]#denominator of the weighted importance sampling using the proportional reward
	usefull_exp = 0.0 # number of episodes contributing to the importance sampling
	ord_is_proportional = []
	for i in range(1, number_of_states): # Set to 0 for each state
		wei_is.append(0.0)     
		wei_is_den.append(0.0)
		vpi_all.append(0.0)
		vpi_proportional.append(0.0)
		wei_is_proportional.append(0.0)
		wei_is_den_proportional.append(0.0)
	
	for idx,ex in enumerate(experiments): #Go through each Experiment to calculate and collect the values
		#print("Experiment {0}".format(idx))
		is_ratio += ex.calculate_is_ratio(removed_states)
		#partial_ord_is = ex.calculate_ord_is(removed_states,fsm_log_counter)
		partial_p_wei_is,partial_p_wei_is_den = ex.calculate_proportional_weighted_is(removed_states, fsm_log_counter)
		partial_wei_is,partial_wei_is_den,ue = ex.calculate_weighted_is(removed_states, fsm_log_counter)
		usefull_exp += ue
		vpi_part = ex.calculate_vpi_for_experiment()
		vpi_prop = ex.calculate_proportional_vpi_for_experiment()		
		#ord_is = [ord_is[i]+partial_ord_is[i] for i in range(0,number_of_states)]
		#print("Vpi {0} Wei {1} Prop {2} Den {3} PropDen {4}".format(vpi_part,partial_wei_is,partial_p_wei_is,partial_wei_is_den,partial_p_wei_is_den))		
		for i in range(0,number_of_states): # Update the values for each state
			wei_is[i]     += partial_wei_is[i]
			wei_is_den[i] += partial_wei_is_den[i]
			vpi_all[i]    += vpi_part[i]
			vpi_proportional[i] += vpi_prop[i]
			wei_is_proportional[i] += partial_p_wei_is[i]
			wei_is_den_proportional[i] += partial_p_wei_is_den[i]
		
#	is_ratio = is_ratio/float(number_of_episodes)
#	ord_is = [ord_is[i]/float(number_of_episodes) for i in range(0,number_of_states)]
	
	for i in range(0,number_of_states):
		ord_is.append(wei_is[i]/float(number_of_episodes))
		vpi_all[i] = vpi_all[i]/float(len(experiments))
		vpi_proportional[i] = vpi_proportional[i]/float(number_of_episodes)
		ord_is_proportional.append(wei_is_proportional[i]/float(number_of_episodes))
		den = wei_is_den[i]
		den2 = wei_is_den_proportional[i]
		if( den == 0):
			den = 1.0
			
		wei_is[i] = wei_is[i]/den
		if( den2 == 0 ):
			den2 = 1.0
		wei_is_proportional[i] = wei_is_proportional[i]/den2
		
	return vpi_all, ord_is, wei_is, vpi_proportional, wei_is_proportional, usefull_exp, ord_is_proportional

def evaluate_state_removal(original_fsm, state_to_remove, number_of_episodes, experiments, control=5):
	nstates = len(original_fsm)
	states_map = list(range(0,nstates-1))
	print("\nOriginal FSM : ")
	orfsm = ""
	for state in original_fsm:	
		orfsm += str(state)+" "
		
	orfsm = "--nstates {0}".format(nstates)+" "+orfsm
	print(orfsm)
	cfsm = ""
	new_number_of_states = nstates
	current_state = 0
	removed_states = []
	fsm_log_counter = copy.deepcopy(original_fsm);	

	#updates state ids and updates the states_map
	for idx,state in enumerate(fsm_log_counter):				
		if(idx != state_to_remove):
			old_id = state.get_id()
			state.set_id(current_state)
			current_state += 1				
		else:		
			states_map = update_states_map(states_map, new_number_of_states, state.get_id())
			removed_states.append(state.get_id())
			new_number_of_states -= 1

	for idx,state in enumerate(fsm_log_counter):	
		if(idx != state_to_remove):					
			state.deactivate_transition_to_states(removed_states)				
			state.update_states_map(states_map)			
			cfsm += str(state)+" "

	cfsm = "--nstates {0}".format(new_number_of_states)+" "+cfsm

	print("\nPruned FSM   : ")
	print(cfsm)		

	if(len(removed_states) > 0 ):
		vpi_all, ord_is, wei_is, vpi_proportional, wei_is_proportional,usefull_exp,ord_is_proportional = importance_sampling_analysis(fsm_log_counter,removed_states, number_of_episodes, experiments)
		#vpi_all, ord_is, wei_is, vpi_proportional, wei_is_proportional,usefull_exp,ord_is_proportional = parameters_comparison(fsm_log_counter,fsm_log_counter,number_of_episodes,experiments)
		print("\n Off-policy analysis of the pruned FSM")
		print(commandline_separator)	
		print("State values of the original FSM                             : {0}".format([round(i,4) for i in vpi_all]))	
		print("States removed by pruning                                    : {0}".format(removed_states))
		print("Number of episodes contributing to the analysis              : {0}/{1}".format(usefull_exp,number_of_episodes))
		print("State values after pruning with ordinary importance sampling : {0}".format([round(i,4) for i in ord_is]))
		print("State values after pruning with weighted importance sampling : {0}".format([round(i,4) for i in wei_is]))
		print("State values using proportional reward calculation           : {0}".format([round(i,4) for i in vpi_proportional]))
		print("State values after pruning with weighted importance sampling : {0}".format([round(i,4) for i in wei_is_proportional]))
		
		average_original_reward = vpi_all[0] * float(number_of_episodes/len(experiments))
		
		average_wei_reward = 0.0
		check = -1	
		for s in range(0,len(wei_is)):
			if(s != state_to_remove):				
				check = s
				state_contribution = 1.0
				if vpi_all[s] != 0.0 :
					state_contribution =  wei_is[s]/vpi_all[s]
					
				average_wei_reward += average_original_reward * state_contribution
				#break
		average_wei_reward = average_wei_reward/(len(wei_is) - 1)		
		average_prop_reward = 0.0
		for s in wei_is_proportional:
			average_prop_reward += s
			
		average_prop_reward *= float(number_of_episodes/len(experiments))	
		
		average_ord_prop_reward = 0.0
		for s in ord_is_proportional:
			average_ord_prop_reward += s
		
		average_ord_prop_reward *= float(number_of_episodes/len(experiments))	
			
		average_ord_reward = 0.0
		for s in range(0,len(ord_is)):
			if(s != state_to_remove):
				state_contribution = 1.0
				if vpi_all[s] != 0.0 :
					state_contribution =  ord_is[s]/vpi_all[s]
				
				average_ord_reward +=  average_original_reward * state_contribution
		
		average_ord_reward = average_ord_reward/(len(ord_is) - 1)
		
		if check >= 0 :						
			print("\n Performance estimation")
			print(commandline_separator)
			print("Average performance of the original FSM                              : {0}".format(round(average_original_reward,3)))
			print("WIS Expected average performance of the pruned FSM                   : {0}".format(round(average_wei_reward,3)))
			print("OIS Expected average performance of the pruned FSM                   : {0}".format(round(average_ord_reward,3)))
			print("WIS Expected average performance with the proportional reward        : {0}".format(round(average_prop_reward,3)))
			print("OIS Expected average performance with the proportional reward        : {0}".format(round(average_ord_prop_reward,3)))
			
			if(usefull_exp > 0 and usefull_exp < 40):
				print("\nWARNING : These results are based on a very small fraction of the total experience and they may not be reliable!")
				
			if(usefull_exp == 0):
				print("\nWARNING : State {0} seems to be a key component of the FSM. \n          Removing it may cause the FSM to be disconnected.".format(state_to_remove))
		
		return average_original_reward,average_wei_reward,average_prop_reward

def evaluate_all_states(or_fsm, number_of_episodes, experiments):
	print("\n Analysis of states contribution")
	results = []
	for state in or_fsm:
		print("Evaluating effectiveness of the FSM without state : {0}".format(state.id))
		
		original,wis,wisprop = evaluate_state_removal(or_fsm, state.id, number_of_episodes, experiments)
		results.append([original, wis, wisprop])
		print("\n")
	
	print("\n Results summary")
	print("State \t\t Original \t\t WIS \t\t Proportional WIS \t\t Average WIS")
	for idx,res in enumerate(results):
		print("   {0}   \t\t {1} \t\t\t {2}  \t\t{3} \t\t {4}".format(idx, round(res[0],3), round(res[1],3), round(res[2],3), round(((res[1]+res[2])/2.0),3)))

def bool_to_string(bool_option):
	if(bool_option):
		return "Active"
	else:
		return "No"	

def evaluate_different_parameters(originalFSM, newFSM, number_of_episodes, experiments):
	
	print("new FSM :\n{0}".format(print_fsm(newFSM)))
	
	vpi_all, ord_is, wei_is, vpi_proportional, wei_is_proportional,usefull_exp,ord_is_proportional = parameters_comparison(originalFSM,newFSM,number_of_episodes,experiments)
	#print("Values from parameters_comparison vpi_all {0} ord_is {1} wei_is {2} vpi_proportional {3}wei_is_proportional {4} usefull_exp {5} ord_is_proportional {6}".format(vpi_all, ord_is, wei_is, vpi_proportional, wei_is_proportional,usefull_exp,ord_is_proportional))
	
	print("\n Off-policy analysis of the new FSM")
	print(commandline_separator)	
	print("State values of the original FSM                             : {0}".format([round(i,4) for i in vpi_all]))		
	print("Number of episodes contributing to the analysis              : {0}/{1}".format(usefull_exp,number_of_episodes))
	print("State values after pruning with ordinary importance sampling : {0}".format([round(i,4) for i in ord_is]))
	print("State values after pruning with weighted importance sampling : {0}".format([round(i,4) for i in wei_is]))
	print("State values using proportional reward calculation           : {0}".format([round(i,4) for i in vpi_proportional]))
	print("State values after pruning with weighted importance sampling : {0}".format([round(i,4) for i in wei_is_proportional]))
	print("State values after pruning with ordinary importance sampling : {0}".format([round(i,4) for i in ord_is_proportional]))
	
	average_original_reward = vpi_all[0] * float(number_of_episodes/len(experiments))
	
	average_wei_reward = 0.0
	check = -1	
	for s in range(0,len(wei_is)):
		state_contribution = 1.0
		if vpi_all[s] != 0.0 :			
			state_contribution =  wei_is[s]/vpi_all[s] # old test vpi_all[s]/wei_is[s]
			average_wei_reward += average_original_reward * state_contribution
			#break
	average_wei_reward = average_wei_reward/(len(wei_is))		
	average_prop_reward = 0.0
	for s in wei_is_proportional:
		average_prop_reward += s
		
	average_prop_reward *= float(number_of_episodes/len(experiments))	
	
	average_ord_prop_reward = 0.0
	for s in ord_is_proportional:
		average_ord_prop_reward += s
	
	average_ord_prop_reward *= float(number_of_episodes/len(experiments))	
		
	average_ord_reward = 0.0
	for s in range(0,len(ord_is)):
		state_contribution = 1.0
		if vpi_all[s] != 0.0 :
			state_contribution =  ord_is[s]/vpi_all[s]
		average_ord_reward +=  average_original_reward * state_contribution
	
	average_ord_reward = average_ord_reward/(len(ord_is))
	
	print("\n Performance estimation")
	print(commandline_separator)
	print("Average performance of the original FSM                              : {0}".format(round(average_original_reward,3)))
	print("WIS Expected average performance of the pruned FSM                   : {0}".format(round(average_wei_reward,3)))
	print("OIS Expected average performance of the pruned FSM                   : {0}".format(round(average_ord_reward,3)))
	print("WIS Expected average performance with the proportional reward        : {0}".format(round(average_prop_reward,3)))
	print("OIS Expected average performance with the proportional reward        : {0}".format(round(average_ord_prop_reward,3)))
	
	if(usefull_exp < 40):
		print("\nWARNING : These results are based on a very small fraction of the total experience and they may not be reliable!")
	
	return average_original_reward,average_wei_reward,average_prop_reward		

def load_FSM(fsm_tokenizer, param_name="--fsm-config"):
	if(fsm_tokenizer.peek() == param_name):
		fsm_tokenizer.next_token() #--fsm-config
		fsm_tokenizer.next_token() #--nstates
	else:
		raise(SyntaxError("Could not find a FSM specification"))
				
	nstates = int(fsm_tokenizer.next_token())
	# initialize log and parse each states
	fsm_log_counter = []
	states_map = list(range(0,nstates-1))
	for idx in range(0,nstates):
		fsm_log_counter.append(AutoMoDeStateParser.AutoMoDeFSMState(idx, fsm_tokenizer, states_map))
	
	return fsm_log_counter


#
#  Main code
###################################################### 

#check that all the arguments are there	
if(len(sys.argv) < 2):
	command_usage()
	raise(SyntaxError("Insert all the required parameters"))
	params = False

history_file = sys.argv[1]
# initialize tokenizer to read the FSM description
fsm_tokenizer = Tokenizer.Tokenizer(sys.argv)
default_scenario = "./scenario.txt"
default_target_runner = "./target-runner"
cut_thresh = 0.0
testPrunedFSM = False
deactivateTransitions = True
all_state_analysis = False
max_runs = 10
randseed=1
is_active = False
pruning = False
remove_state = False
parameter_analysis = False 
rounding = True
state_to_remove = -1
fsm_tokenizer.next_token() # token 0 "AutoMoDeLogAnalyzer.py"
if not(fsm_tokenizer.peek().startswith("-")):
	fsm_tokenizer.next_token() # history file

#Checks if a value for the threshold has been provided,
# otherwise it uses the default one 0
params=True
while(params and fsm_tokenizer.has_more_tokens()):
	tok = fsm_tokenizer.peek()
	if( tok == "--threshold" or tok == "-t"):		
		fsm_tokenizer.next_token()
		cut_thresh = fsm_tokenizer.getFloat()		
	elif(tok == "--help" or tok == "-h"):
		command_usage()
		exit(0)
	elif(tok == "--scenario" or tok == "-sc"):
		fsm_tokenizer.next_token()
		default_scenario = fsm_tokenizer.next_token()		
	elif(tok == "--targetrunner" or tok == "-ta"):
		fsm_tokenizer.next_token()
		default_target_runner = fsm_tokenizer.next_token()
	elif(tok == "--runs" or tok == "-ru"):
		fsm_tokenizer.next_token()
		max_runs = fsm_tokenizer.getInt()
	elif(tok == "--rseed" or tok == "-rs"):
		fsm_tokenizer.next_token()
		randseed = fsm_tokenizer.getInt()
	elif(tok == "--keep-transitions" or tok == "-kt"):
		fsm_tokenizer.next_token()
		deactivateTransitions = False		
	elif(tok == "--test" or tok == "-te"):
		fsm_tokenizer.next_token()
		testPrunedFSM = True
		pruning = True
	elif(tok == "--fsm-config"):
		params=False
	elif(tok == "--no-is-analysis" or tok == "-na"):
		fsm_tokenizer.next_token()
		is_active = False
	elif(tok == "--all-state-analysis" or tok == "-as"):
		fsm_tokenizer.next_token()
		all_state_analysis = True
	elif(tok == "--pruning" or tok == "-p"):
		fsm_tokenizer.next_token()
		pruning = True
		is_active = True
	elif(tok == "--remove-state" or tok == "-ds"):
		fsm_tokenizer.next_token()
		remove_state = True
		state_to_remove = fsm_tokenizer.getInt()
	elif(tok == "--param-analysis" or tok == "-pa"):
		fsm_tokenizer.next_token()
		newfsm = load_FSM(fsm_tokenizer,"--newfsm-config")
		parameter_analysis = True
	elif(tok == "--no-rounding" or tok == "-nr"):
		fsm_tokenizer.next_token()
		rounding = False
	else:
		fsm_tokenizer.next_token()

# move the current token to the start of the FSM
pos = fsm_tokenizer.seek("--fsm-config")
# if the FSM description is not found
original_fsm = ""
original_fsm_list = []
if(pos<0):
	f = open(history_file,"r")
	fline = f.readline()
	fsm_tokenizer = Tokenizer.Tokenizer(fline)
	f.close()
	pos = fsm_tokenizer.seek("--fsm-config")
	if(pos < 0):
		command_usage()
		raise(SyntaxError("Finite state machine description not found"))
	else:
		print("\n Reading the FSM configuration from the log file.")	
		original_fsm_list = fsm_tokenizer.tokens	
		pos += 1
		fsm_tokenizer.next_token() #needed because when added to the log file the FSM description has also the configuration number
else:
	original_fsm_list = sys.argv
	
for arg in range(pos+1, len(original_fsm_list)):
	original_fsm += original_fsm_list[arg] + " "
	
fsm_tokenizer.next_token() #this token is --fsm-config
fsm_tokenizer.next_token() #this token is --nstates

print(" Configuration")
print(commandline_separator)
print("Threshold value for state pruning          : {0}".format(cut_thresh))
print("Deactivating transitions to removed states : {0}".format(bool_to_string(deactivateTransitions)))
print("Performance estimation of the pruned FSM   : {0}".format(bool_to_string(is_active)))
print("Analysis of the contribution of all states : {0}".format(bool_to_string(all_state_analysis)))
print("Estimating state removal                   : {0}".format(bool_to_string(remove_state)))
print("Parameter analysis                         : {0}".format(bool_to_string(parameter_analysis)))
if(testPrunedFSM):
	print("FSM pruning                                : Active")
	print("Evaluation of the pruned FSM               : Active")
	print("Scenario file                              : {0}".format(default_scenario))
	print("Target runner                              : {0}".format(default_target_runner))	
	print("Number of tests                            : {0}".format(max_runs))	
	print("Random seed                                : {0}".format(randseed))
else:
	print("Evaluation of the pruned FSM               : No")
	print("FSM pruning                                : {0}".format(bool_to_string(pruning)))
if remove_state:
	print("State to remove                            : {0}".format(state_to_remove))
	
nstates = int(fsm_tokenizer.next_token())

# initialize log and parse each states
fsm_log_counter = []
states_map = list(range(0,nstates-1))
for idx in range(0,nstates):
	fsm_log_counter.append(AutoMoDeStateParser.AutoMoDeFSMState(idx, fsm_tokenizer, states_map))
	
	

#print("FSM : \n"+str(fsm_log_counter))
experiments = []
number_of_ticks,number_of_episodes = analyze_logfile(history_file, fsm_log_counter, experiments)
print("\nFSM log ")
print(commandline_separator)
print("Total number of experiments : {0}".format(len(experiments)))
print("Total number of ticks       : {0}".format(number_of_ticks))
print("Total number of states      : {0}".format(nstates))
for state in fsm_log_counter:
	state_load = round(float(state.get_counter())/float(number_of_ticks)*100,2)
	print("Sate {0} active for           : {1} ticks or {2}% ".format(state.get_id(),state.get_counter(),state_load)+str(state.get_transition_counters()))

vpi,vpip = compute_state_values(fsm_log_counter, number_of_episodes, experiments)

if(rounding):
	vpi = [round(i,4) for i in vpi]
	vpip = [round(i,4) for i in vpip]

print("State values                : {0}".format(vpi))
print("State values proportional   : {0}".format(vpip))	
#for idx,e in enumerate(experiments):
#	print("Exp {0} #episodes : {1}".format(idx, len(e.logs)) )

print("\nOriginal FSM : ")
print(original_fsm)
removed_states = []
or_fsm = copy.deepcopy(fsm_log_counter);	

if(pruning):
	cfsm = ""
	new_number_of_states = nstates
	current_state = 0		
	#updates state ids and updates the states_map
	for idx,state in enumerate(fsm_log_counter):				
		if(state.get_counter()/float(number_of_ticks) > cut_thresh):
			old_id = state.get_id()
			state.set_id(current_state)
			current_state += 1				
		else:		
			states_map = update_states_map(states_map, new_number_of_states, state.get_id())
			removed_states.append(state.get_id())
			new_number_of_states -= 1

	for state in fsm_log_counter:	
		if((state.get_counter()/float(number_of_ticks)) > cut_thresh):		
			if(deactivateTransitions):
				state.deactivate_transition_to_states(removed_states)
				
			state.update_states_map(states_map)			
			cfsm += str(state)+" "

	cfsm = "--nstates {0}".format(new_number_of_states)+" "+cfsm

	print("\nPruned FSM   : ")
	print(cfsm)		

if(is_active and len(removed_states) > 0 ):
	vpi_all, ord_is, wei_is, vpi_proportional, wei_is_proportional,ord_is_proportional = importance_sampling_analysis(fsm_log_counter,removed_states, number_of_episodes, experiments)
	print("\n Off-policy analysis of the pruned FSM")
	print(commandline_separator)	
	print("State values of the original FSM                             : {0}".format([round(i,4) for i in vpi_all]))	
	print("States removed                                               : {0}".format(removed_states))
	print("State values after pruning with ordinary importance sampling : {0}".format([round(i,4) for i in ord_is]))
	print("State values after pruning with weighted importance sampling : {0}".format([round(i,4) for i in wei_is]))
	print("State values using proportional reward calculation           : {0}".format([round(i,4) for i in vpi_proportional]))
	print("State values after pruning with weighted importance sampling : {0}".format([round(i,4) for i in wei_is_proportional]))
	
	average_original_reward = vpi_all[0] * float(number_of_episodes/len(experiments))
		
	for s in range(0,len(wei_is)):
		if(wei_is[s] != 0):
			average_wei_reward = wei_is[s] * float(number_of_episodes/len(experiments))
			break
	
	average_prop_reward = 0.0
	for s in wei_is_proportional:
		average_prop_reward += s
			
	average_prop_reward *= float(number_of_episodes/len(experiments))
	
	print("\n Performance estimation")
	print(commandline_separator)
	print("Average performance of the original FSM           : {0}".format(round(average_original_reward,3)))
	print("Expected average performance of the pruned FSM    : {0}".format(round(average_wei_reward,3)))
	print("Expected performance with proportional estimation : {0}".format(average_prop_reward))
	
if(all_state_analysis):
	evaluate_all_states(or_fsm, number_of_episodes, experiments)

if(remove_state):
	evaluate_state_removal(or_fsm, state_to_remove, number_of_episodes, experiments)
	
if(parameter_analysis):
	evaluate_different_parameters(or_fsm, newfsm, number_of_episodes, experiments)

if(testPrunedFSM):
	print("\n Performance evaluation")
	print(commandline_separator)
	# read scenario file
	print("Reading scenario file ")
	instances = read_scenario_file(default_scenario)
	#print(instances)	
	# run tests with default_target_runner
	print("Running experiments ")
	execute_experiments(max_runs, instances, original_fsm, cfsm, default_target_runner, randseed)
	# compare results with statistical test
	#print("Comparing results ")	
	# output
	
