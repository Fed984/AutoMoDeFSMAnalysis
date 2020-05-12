#!/bin/bash

function print_guide_and_quit
{

  echo "AutoMoDe $0"
  echo "Usage:     "
  echo "$0 irace_stdout destination_path [canfiguration]" 
  exit -1

}

if [ $# -lt 2 ]
then
   print_guide_and_quit
fi

IRACE_EXECUTION_OUTPUT=$1
DESTINATION_DIR=$2

if [ $DESTINATION_DIR == "." ]
then
    DESTINATION_DIR=$PWD/
fi


if [ -f $IRACE_EXECUTION_OUTPUT ]
then 
  NODES=`cat $IRACE_EXECUTION_OUTPUT | grep compute | grep -v "catch" | uniq `
  TMPDIR=`echo $IRACE_EXECUTION_OUTPUT | awk -F/ '{print $NF}' | cut -d'.' -f1`
else
  echo "specify a valid irace output file"
  print_guide_and_quit
fi

if [ $# -eq 3 ]
then
   CANDIDATE=$3
else
  CANDIDATE=`cat $IRACE_EXECUTION_OUTPUT | grep "Best configurations as commandlines" -A1 | tail -n 1 | awk '{print \$1}' `
fi

CONFIGURATION=`cat $IRACE_EXECUTION_OUTPUT | grep "$CANDIDATE" | grep "nstates" `

echo "Configuration         : $CANDIDATE"
echo "Destination directory : $DESTINATION_DIR"
echo "Nodes: "
echo $NODES 

if [ -e $DESTINATION_DIR ]
then
     for i in $NODES
     do
        ssh $i cp /tmp/$TMPDIR/fsm-$CANDIDATE-* $DESTINATION_DIR
        #ssh $i rm -f /tmp/$TMPDIR/fsm-$CANDIDATE-*
     done
else
    echo "Specify a valid destination directory"
    print_guide_and_quit
fi

echo "--fsm-config $CONFIGURATION" >> $CANDIDATE-fsmlog
pushd $DESTINATION_DIR
for i in `ls fsm-$CANDIDATE-*`
do
  cat $i >> $CANDIDATE-fsmlog
  rm -f $i
done

