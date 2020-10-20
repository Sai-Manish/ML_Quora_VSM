#!/usr/bin/env bash

TEST=./preprocessed_test.csv
TRAIN=./preprocessed_train.csv

clear
if [ -f $TEST ]; then
	rm preprocessed_test.csv
	printf "Removed $TEST\n"
fi
if [ -f $TRAIN ]; then
	rm preprocessed_train.csv
	printf "Removed $TRAIN\n"
fi

time python Preprocesser.py
printf "\n\n\nDone with the proprocessing\nTraining ML model now.\n\n\n"
time python Model.py
