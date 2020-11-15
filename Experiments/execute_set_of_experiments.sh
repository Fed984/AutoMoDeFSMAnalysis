#!/bin/bash

AUTOMODE_EXEC=~/ARGoS3-AutoMoDe/bin/automode_main
AUTOMODE_ANALYZER=../AutoMoDeLogAnalyzer.py
PYTHON=python3.8

# Create the executions traces for the original finite state machines
input="./original_finite_state_machines.txt"
while IFS= read -r line
do
  set -- $line
  rm ./traces/$2-fsmlog
  touch ./traces/$2-fsmlog
  echo $line >> ./traces/$2-fsmlog
  echo "Get execution traces for $2"
  for i in {1..30}; do $AUTOMODE_EXEC -t -n -c foraging.argos --seed RNDSEED $line >> ./traces/$2-fsmlog; done
done < "$input"

# Run the analyzer
input="./params_final_choice.txt"

echo "average_performance_original_FSM, discounted_WIS, discounted_OIS, proportional_discounted_WIS, proportional_discounted_OIS, WIS, OIS, proportional_WIS, proportional_OIS, average_simulation" >> results.csv

while IFS= read -r line
do
  set -- $line
  experiment_name=$2
  echo "Experiment $experiment_name results:"

  execution_number=$(echo "$2" | sed 's/[^0-9]*//g')
  echo "./traces/$execution_number-fsmlog"

  arguments=$(echo "$line" | sed 's/[^ ]* *[^ ]* *//')
  output="$($PYTHON $AUTOMODE_ANALYZER ./traces/$execution_number-fsmlog -pa --newfsm-config $arguments)"

  # Extract the data with discount factor from the output
  average_performance_original_FSM=$(echo "$output" | grep "Average performance of the original FSM " | sed 's/[0-9]*.[^0-9]*//')
  discounted_WIS=$(echo "$output" | grep "Discounted WIS Expected average performance of the pruned FSM " | sed 's/[0-9]*.[^0-9]*//')
  discounted_OIS=$(echo "$output" | grep "Discounted OIS Expected average performance of the pruned FSM " | sed 's/[0-9]*.[^0-9]*//')
  proportional_discounted_WIS=$(echo "$output" | grep "Discounted WIS Expected average performance with the proportional reward " | sed 's/[0-9]*.[^0-9]*//')
  proportional_discounted_OIS=$(echo "$output" | grep "Discounted OIS Expected average performance with the proportional reward " | sed 's/[0-9]*.[^0-9]*//')
  
  # Extract the data without discount factor from the output
  WIS=$(echo "$output" | grep "WIS Expected average performance of the pruned FSM" | sed 's/[0-9]*.[^0-9]*//')
  OIS=$(echo "$output" | grep "OIS Expected average performance of the pruned FSM" | sed 's/[0-9]*.[^0-9]*//')
  proportional_WIS=$(echo "$output" | grep "WIS Expected average performance with the proportional reward " | sed 's/[0-9]*.[^0-9]*//')
  proportional_OIS=$(echo "$output" | grep "OIS Expected average performance with the proportional reward" | sed 's/[0-9]*.[^0-9]*//')

  # Perform multiple automode simulations in order to estimate the average performance of the modified FSM in order to compute the mean squared error metric
  average_simulation="$(for i in {1..30}; do $AUTOMODE_EXEC -t -n -c foraging.argos --seed RNDSEED --fsm-config $arguments | grep Score | sed 's/[^0-9]*//'; done | awk '{ total += $1; count++ } END { print total/count }')"
  echo "$average_performance_original_FSM, $discounted_WIS, $discounted_OIS, $proportional_discounted_WIS, $proportional_discounted_OIS, $WIS, $OIS, $proportional_WIS, $proportional_OIS, $average_simulation" >> results.csv
done < "$input"

# Make the plots
Rscript make_plots.R
