#!/bin/bash

input="./params_final_choice.txt"
while IFS= read -r line
do
  set -- $line
  experiment_name=$2
  echo "Experiment $experiment_name results:"

  execution_number=$(echo "$2" | sed 's/[^0-9]*//g')
  #echo "./traces/$execution_number-fsmlog"

  arguments=$(echo "$line" | sed 's/[^ ]* *[^ ]* *//')
  output="$(python3.7 ../AutoMoDeLogAnalyzer.py ./traces/$execution_number-fsmlog -pa --newfsm-config $arguments)"

  average_performance_original_FSM=$(echo "$output" | grep "Discounted Average performance of the original FSM " | sed 's/[0-9]*.[^0-9]*//')
  discounted_OIS=$(echo "$output" | grep "Discounted OIS Expected average performance of the pruned FSM " | sed 's/[0-9]*.[^0-9]*//')
  proportional_discounted_WIS=$(echo "$output" | grep "Discounted WIS Expected average performance with the proportional reward " | sed 's/[0-9]*.[^0-9]*//')
  proportional_discounted_OIS=$(echo "$output" | grep "Discounted OIS Expected average performance with the proportional reward " | sed 's/[0-9]*.[^0-9]*//')
  
  WIS=$(echo "$output" | grep "WIS Expected average performance of the pruned FSM" | sed 's/[0-9]*.[^0-9]*//')
  OIS=$(echo "$output" | grep "OIS Expected average performance of the pruned FSM" | sed 's/[0-9]*.[^0-9]*//')
  proportional_WIS=$(echo "$output" | grep "WIS Expected average performance with the proportional reward " | sed 's/[0-9]*.[^0-9]*//')
  proportional_OIS=$(echo "$output" | grep "OIS Expected average performance with the proportional reward" | sed 's/[0-9]*.[^0-9]*//')

  average_simulation"$(for i in {1..30}; do ./automode_main -t -n -c foraging.argos --seed RNDSEED --fsm-config $arguments | grep Score | sed 's/[^0-9]*//'; done | awk '{ total += $1; count++ } END { print total/count }')"
  echo $average_simulation
done < "$input"