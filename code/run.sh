#!/usr/bin/env bash

TEST=./preprocessed_test.csv
TRAIN=./preprocessed_train.csv
IDEAL=./preprocessed_ideal_40.csv

clear
if [ -f $TEST ]; then
	rm $TEST 
	printf "Removed $TEST\n"
fi
if [ -f $TRAIN ]; then
	rm $TRAIN
	printf "Removed $TRAIN\n"
fi
if [ -f $IDEAL ]; then
	rm $IDEAL
	printf "Removed $IDEAL\n"
fi

time python Preprocesser.py
printf "\n\n\nDone with the proprocessing\nTraining ML model now.\n\n\n"
time python Model.py
