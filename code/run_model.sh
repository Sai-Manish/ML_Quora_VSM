#!/usr/bin/env bash

printf "\nTraining Bernoulli Naive Bayes model.\nShould take about 2-3 minutes.\n"
time python BernoulliNB.py

printf "\nTraining Linear support vector classifier.\nShould take about 6-8 minutes.\n"
time python LSVC.py

printf "\nTraining logistic regression model, using stochastic gradient descent.\nShould take about 4-5 minutes.\n"
time python SGD.py

printf "\nTraining and ensemble of the above 3 models.\nShould take about 8-9 minutes.\n"
time python Ensemble.py
