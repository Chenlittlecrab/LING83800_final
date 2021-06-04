# Final project: Gender classification

This project is based on the qualifying paper I am working on. The study in the paper collected vowel and fricative data from 74 American English speakers (38f, 36m) in NYC. The goal of the paper is to investigate whether gender identity, measured by self-rated femininity scores, are indexed in American English speakers' speech. 

This final project is an extension from the qualifying paper. It explores whether descriminative classifiers are able to classify the gender of the speakers based on the following acoustic measurements analyzed using Praat:

1. vowel formants: F1-F3, all transferred in bark scale;
2. four spectral moments of fricatives, including Center of Gravity (cog), standard deviation (stdev), skewness, and kurtosis. 

The project will use logistic regression and Support Vector Machines (SVM) to train and test the data to see if those two types of discriminative classifiers can classify the gender of speakers for each measurement accurately. 

## Part 1: preparation

split_clean_data.py is used for cleaning vowel (vowel_data.txt) and fricative (fricative_data.txt) data and for splitting them into train and test data. 

The output of this part are training and testing txt files for vowels and fricatives seperately. 

## Part 2: train and evaluation

gender_classification.py then takes the training and testing txt files as the input and calculate the accuracy of the model built with sklearn using logistic regression and SVM. 