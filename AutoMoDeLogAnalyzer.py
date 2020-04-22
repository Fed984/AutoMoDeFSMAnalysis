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
import scipy.stats
import AutoMoDeFSMExperiment
import copy

commandline_separator = "-------------------------------------------------------------------------------------"	
# print how to use the script
def command_usage():
	print("Usage   : \n\t AutoMoDeLogAnalyzer historyfile_path [options] --fsm-config <finite state machine description>".format(sys.argv[0]))
	print("Options : ")
	print("\t --threshold VALUE")
	print("\t\t Set the threshold used to decide wether to delete a state.")
	print("\t\t The value represent a percentage that should go from 0 (the default value,")
	print("\t\t a state is delete if it is never accessed) to 1 (all the states are deleted).\n")	
	print("\t --test")	
	print("\t\t When used the script will run RUN experiments with the original and pruned FSM.")
	print("\t\t The results are compared using the Wilcoxon paired statistical test.\n")
	print("\t --scenario FILE")
	print("\t\t The script will run the experiments using the instances specified in the irace")
	print("\t\t scenario file FILE.\n")
	print("\t --targetrunner FILE") 
	print("\t\t The script will use the target-runner FILE to run the experiments.\n")
	print("\t --runs RUN")
	print("\t\t The script will run RUN experiments before the statistical test.\n")
	print("\t --rseed SEED")
	print("\t\t The random seed used to initialize the random seed generator\n")
	print("\t --keep-transitions")
	print("\t\t The transitions towards a removed state are kepts and rerouted\n")
	print("\t --help")
	print("\t\t prints this help.\n")
	
	
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
	for line in lines:
		if(line.startswith("trainInstancesDir")):
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
		elif(line.startswith("trainInstancesFile")):
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
	for ins in instances:
		for r in range(0, random_seeds):			
			seed = str(random.randint(0,pow(10,9))) #generate random seed
			print("Running {0} with seed {1}".format(ins, seed))
			ro = float(subprocess.check_output([default_target_runner,"0","0",seed,ins,original_fsm]))
			results_original.append(ro)
			rp = float(subprocess.check_output([default_target_runner,"0","0",seed,ins,pruned_fsm]))
			results_pruned.append(rp)
	
	print("Results original : {0}".format(results_original))
	print("Results pruned   : {0}".format(results_pruned))
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
		stat, p = scipy.stats.wilcoxon(results_original, results_pruned)
		print('stat=%.3f, p=%.3f' % (stat, p))
		if p > 0.05:
			print('There is no significant difference')
		else:
			print('There is a significant difference')
	else:
		p=0
		print("The two FSM reported exactly the same results")
			
	return p
		
# Reads and analyzes the FSM history file 
# returns the total number of time ticks read
def analyze_logfile(history_file, fsm_log_counter, experiments):
	number_of_ticks = 0
	number_of_episodes= 0
	print("Opening log file...")
	f = open(history_file,"r")
	Lines = f.readlines();
	print("Number of lines {0}".format(len(Lines)))
	previous_state = 0
	cstate = 0
	print("Analyzing log file...")	
	recording_exp = False	
	exp_states = [0]
	exp_transitions = [0]
	exp_transitions_probabilities = [1]
	exp_active_transitions = [1]
	cscore = 0
	r_recording = False
	vpi_overall = [0.0 for i in range(0,len(fsm_log_counter))]
	vpi_states = [0 for i in range(0,len(fsm_log_counter))]	
	for line in Lines:
		if(not(recording_exp) and line.startswith("Score ")):	# Starting of the experiment		
					cscore = float(line.split()[1]) # Getting the score 
					exp = AutoMoDeFSMExperiment.AutoMoDeExperiment() #Initializing the experiment object	
					exp.set_result(cscore)
					recording_exp = True # Activating the recording
		elif(recording_exp and not(line.startswith("[INFO]"))):	# Parsing the experiment traces		
			t = Tokenizer.Tokenizer(line) # splitting the traces in tokens
			previous_state = cstate # saving the previous state
			while t.has_more_tokens():  
				ctoken = t.next_token()						
				if(ctoken.startswith("--t")):   #if token is a clock tick, update the clock ticks counter
					ctick = int(t.next_token())
					number_of_ticks += 1
					if(ctick == 0 and len(exp_states) > 1):	#Saving the experience for each robot
						#print("Saving EPISODE ROBOT {0} OF EXPERIMENT {1} vpi {2} # {3}".format(number_of_robots,len(experiments),vpi_overall,len(exp_states)))
						exp.set_startIdx(ctick)	# saving start tick (not usefull since ticks reset)
						exp_log = [exp_states, exp_transitions, exp_transitions_probabilities, exp_active_transitions] 
						exp.append_logs(exp_log) # save logs of the robot
						vpi_states = [0 for i in range(0,len(fsm_log_counter))] #reinitialize
						exp_states = [0]                    # reinitialize
						exp_transitions = [0]               # reinitialize
						exp_transitions_probabilities = [1] # reinitialize
						exp_active_transitions = [1]        # reinitialize
				
					if(ctick == 0): #Updating the total number of robots
						number_of_episodes += 1
						
				else:
					match = re.search("--s[0-9]+", ctoken)	
					if(match != None):  #if token is a state description log the state execution
						match = re.search("[0-9]+", ctoken)
						cstate = int(match.group(0));
						fsm_log_counter[cstate].increase_counter_state() #Updating the counter state
						if( vpi_states[cstate] == 0): #If this state was not visited by this robot
							vpi_states[cstate] = 1
							vpi_overall[cstate] += 1 # update the Experiment First Visit count 
						#print("Current State {} of type {}".format(cstate,t.next_token()))
			
					match = re.search("--c[0-9]+", ctoken)		
					if(match != None): #if token is a condition, log only if the condition is true
						match = re.search("[0-9]+", ctoken)
						condition = int(match.group(0))
						type = t.next_token(); # type of the transition
						value = int(t.next_token()) # value (with the new code is always 1)
						probability = float(t.next_token()) # probability that the transition was active
						active_transitions = 1
						if(t.peek().startswith("--a")): # if the active states log is present
							t.next_token()
							active_transitions = int(t.next_token()) #number of active transitions
							
						if(value == 1):
							fsm_log_counter[previous_state].increase_counter_transition(condition)
							#exp_log.append("t {0} {1} {2}".format(condition, probability, active_transitions))
							#exp_log.append("s {0}".format(cstate))	
							exp_states.append(cstate) #update the states log
							exp_transitions.append(condition) #update the transitions log
							exp_transitions_probabilities.append(probability)# probabilities
							exp_active_transitions.append(active_transitions)# active transitions
							#print("Condition {} of type {} has value {} belongs to state {}".format(condition,type,value,previous_state))
		elif(recording_exp and line.startswith("[INFO]")): # Beginning of a new experiment 
			recording_exp = False    # stop recording
			exp_log = [exp_states, exp_transitions, exp_transitions_probabilities, exp_active_transitions]
			exp.append_logs(exp_log) # save experiment log
			exp.set_vpi(vpi_overall) # save overall vpi
			experiments.append(exp)	 # save experiment
			exp.set_endIdx(ctick)	 # saving final time tick (not usefull since ticks reset per each robot)
			vpi_overall = [0.0 for i in range(0,len(fsm_log_counter))] # reinitialize 				
			vpi_states = [0 for i in range(0,len(fsm_log_counter))]    # reinitialize
			exp_states = [0] 					   # reinitialize
			exp_transitions = [0] 					   # reinitialize
			exp_transitions_probabilities = [1] 			   # reinitialize
			exp_active_transitions = [1] 				   # reinitialize			
			cscore = 0 						   # reinitialize
	
	if(len(exp_states) > 0): # Save final experiment
		exp_log = [exp_states, exp_transitions, exp_transitions_probabilities, exp_active_transitions]
		exp.append_logs(exp_log)  # save experiment log
		exp.set_vpi(vpi_overall)  # save overall vpi
		#print("Saving EPISODE ROBOT {0} OF EXPERIMENT {1}".format(number_of_robots,len(experiments)))
		exp.set_endIdx(ctick)     # saving final time tick (not usefull since ticks reset per each robot)
		experiments.append(exp)   # save experiment
		
	f.close()
	print("number of episodes {0}".format(number_of_episodes))	
	return number_of_ticks,number_of_episodes
	

#check that all the arguments are there	
if(len(sys.argv) < 4):
	command_usage()
	raise(SyntaxError("Insert all the required parameters"))

history_file = sys.argv[1]
# initialize tokenizer to read the FSM description
fsm_tokenizer = Tokenizer.Tokenizer(sys.argv)
default_scenario = "./scenario.txt"
default_target_runner = "./target-runner"
cut_thresh = 0.0
testPrunedFSM = False
deactivateTransitions = True
max_runs = 10
randseed=1
fsm_tokenizer.next_token() # token 0 "AutoMoDeLogAnalyzer.py"
fsm_tokenizer.next_token() # history file

#Checks if a value for the threshold has been provided,
# otherwise it uses the default one 0
params=True
while(params and fsm_tokenizer.has_more_tokens()):
	if(fsm_tokenizer.peek() == "--threshold"):		
		fsm_tokenizer.next_token()
		cut_thresh = fsm_tokenizer.getFloat()		
	elif(fsm_tokenizer.peek() == "--help"):
		command_usage()
		exit(0)
	elif(fsm_tokenizer.peek() == "--scenario"):
		fsm_tokenizer.next_token()
		default_scenario = fsm_tokenizer.next_token()		
	elif(fsm_tokenizer.peek() == "--targetrunner"):
		fsm_tokenizer.next_token()
		default_target_runner = fsm_tokenizer.next_token()
	elif(fsm_tokenizer.peek() == "--runs"):
		fsm_tokenizer.next_token()
		max_runs = fsm_tokenizer.getInt()
	elif(fsm_tokenizer.peek() == "--rseed"):
		fsm_tokenizer.next_token()
		randseed = fsm_tokenizer.getInt()
	elif(fsm_tokenizer.peek() == "--keep-transitions"):
		fsm_tokenizer.next_token()
		deactivateTransitions = False		
	elif(fsm_tokenizer.peek() == "--test"):
		fsm_tokenizer.next_token()
		testPrunedFSM = True
	elif(fsm_tokenizer.peek() == "--fsm-config"):
		params=False
	else:
		fsm_tokenizer.next_token()

# move the current token to the start of the FSM
pos = fsm_tokenizer.seek("--fsm-config")
# if the FSM description is not found
if(pos<0):
	command_usage()
	raise(SyntaxError("Finite state machine description not found"))
	
fsm_tokenizer.next_token() #this token is --fsm-config
original_fsm = ""
for arg in range(pos+1, len(sys.argv)):
	original_fsm += sys.argv[arg] + " "

fsm_tokenizer.next_token() #this token is --nstates

print("Configuration")
print(commandline_separator)
print("Threshold value for state pruning     : {0}".format(cut_thresh))
if(testPrunedFSM):
	print("Evaluation of the pruned FSM          : Active")
	print("Scenario file                         : {0}".format(default_scenario))
	print("Target runner                         : {0}".format(default_target_runner))	
	print("Number of tests                       : {0}".format(max_runs))
	print("Keeping transitions to removed states : {0}".format(deactivateTransitions))
else:
	print("Evaluation fo the pruned FSM          : No")
	
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
	print("Sate {0} active for {1} ticks or {2}% ".format(state.get_id(),state.get_counter(),state_load)+str(state.get_transition_counters()))
	


print("\nOriginal FSM : ")
print(original_fsm)

print("\nPruned FSM   : ")
cfsm = ""
new_number_of_states = nstates
current_state = 0
removed_states = []
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
		
or_fsm = []		
for state in fsm_log_counter:
	or_fsm.append(copy.copy(state))	
	if((state.get_counter()/float(number_of_ticks)) > cut_thresh):
		state.update_states_map(states_map)
		if(deactivateTransitions):
			state.deactivate_transition_to_states(removed_states)
			
		cfsm += str(state)+" "
	

cfsm = "--nstates {0}".format(new_number_of_states)+" "+cfsm
print(cfsm)		

number_of_states = len(fsm_log_counter)
is_ratio = 0.0
ord_is = [0.0 for i in range(0, number_of_states)]
wei_is = [0.0 for i in range(0, number_of_states)]
wei_is_den = [0.0 for i in range(0, number_of_states)]
vpi_all = [0.0 for i in range(0, number_of_states)]
for ex in experiments:
	#print(ex)
	is_ratio += ex.calculate_is_ratio(removed_states)
	partial_ord_is = ex.calculate_ord_is(removed_states,or_fsm)
	partial_wei_is,partial_wei_is_den = ex.calculate_weighted_is(removed_states, or_fsm)
	vpi_part = ex.calculate_vpi_for_experiment()
	ord_is = [ord_is[i]+partial_ord_is[i] for i in range(0,number_of_states)]
	wei_is = [wei_is[i]+partial_wei_is[i] for i in range(0,number_of_states)]
	wei_is_den = [wei_is_den[i]+partial_wei_is_den[i] for i in range(0,number_of_states)]
	vpi_all = [vpi_all[i]+vpi_part[i] for i in range(0,number_of_states)]
	
is_ratio = is_ratio/float(number_of_episodes)
ord_is = [ord_is[i]/float(number_of_episodes) for i in range(0,number_of_states)]
vpi_all = [vpi_all[i]/float(len(experiments)) for i in range(0,number_of_states)]
wei_is = [wei_is[i]/wei_is_den[i] for i in range(0,number_of_states)]

average_original_reward = vpi_all[0] * float(number_of_episodes/len(experiments))
average_wei_reward = wei_is[0] * float(number_of_episodes/len(experiments))

print("\n Off-policy analysis of the pruned FSM")
print(commandline_separator)
print("State values of the original FSM                             : {0}".format(vpi_all))
print("State values after pruning with ordinary importance sampling : {0}".format(ord_is))
print("State values after pruning with weighted importance sampling : {0}".format(wei_is))
#print("is ratio due to the removal of state {0} {1}".format(removed_states, is_ratio))

print("\n Performance estimation")
print(commandline_separator)
print("Average performance of the original FSM        : {0}".format(average_original_reward))
print("Expected average performance of the pruned FSM : {0}".format(average_wei_reward))


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
	