#!/bin/bash

SOLNS2OUT_EXEC=/home/natalia/Installs/MiniZincIDE-2.1.4-bundle-linux-x86_64/solns2out

echo "Run all"

for i in $1/*.fzn
do 
	f=${i##*/}
	echo "<><><><><><><><><><>"
	echo "RUNNING BENCHMARK" $i 
	$BRANCH/fzn_chuffed -verbosity=2 -time_out=3600 -lazy=true -steinerlp=true $i | $SOLNS2OUT_EXEC $1/${f%.*}.ozn; done;


