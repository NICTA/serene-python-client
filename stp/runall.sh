#!/bin/bash

echo "Run all"

for i in $1/*.fzn
do 
	f=${i##*/}
	echo "<><><><><><><><><><>"
	echo "RUNNING BENCHMARK" $i 
	$BRANCH/fzn_chuffed -verbosity=2 -time_out=3600 -lazy=true -steinerlp=true $i | solns2out $1/${f%.*}.ozn; done;


