# MealRecognitionClassifier
Repository for an academic project focused on training a machine learning model to distinguish between meal and no meal time series data using Python. The project involves feature extraction, model training, and evaluation through k-fold cross-validation for accurate meal detection.

## Purpose
In this project, you will use a training dataset to train and test a machine model. The purpose is to distinguish between meal and no meal time series data. 

## Objectives
Learners will be able to: 
1. Develop code to train a machine model. 
2. Assess the accuracy of a machine model. 

## Technology Requirements
1. Python 3.6 to 3.8 (do not use 3.9). 
2. scikit-learn==0.21.2 
3. pandas==0.25.1 
4. Python pickle 

## Directions
Meal data can be extracted as follows: 

1. From the InsulinData.csv file, search the column Y for a non NAN non zero value. This time indicates the start of meal consumption time tm. Meal data comprises a 2hr 30 min stretch of CGM data that  starts from tm-30min and extends to tm+2hrs. 
2. No meal data comprises 2 hrs of raw data that does not have meal intake. 

## Extraction: Meal data
Start of a meal can be obtained from InsulinData.csv. Search column Y for a non NAN non zero value. This time indicates the start of a meal. There can be three conditions: 
1. There is no meal from time tm to time tm+2hrs. Then use this stretch as meal data 
2. There is a meal at some time tp in between tp>tm and tp< tm+2hrs. Ignore the meal data at time tm and consider the meal at time tp instead. 
3. There is a meal at time tm+2hrs, then consider the stretch from tm+1hr 30min to tm+4hrs  as meal data. 

## Extraction: No Meal data 
Start of no meal is at time tm+2hrs where tm is the start of some meal. We need to obtain a 2 hr stretch of no meal time. So you need to find all 2 hr stretches in a day that have no meal and do not fall  within 2 hrs of the start of a meal. 

## Handling missing data: 
You have to carefully handle missing data. This is an important data mining step that is required for many applications. Here there are several approaches: 
1. Ignore the meal or no meal data stretch if the number of missing data points in that stretch is greater than a certain threshold.
2. Use linear interpolation  (not a good idea for meal data but maybe for no meal data)
3. Use polynomial regression to fill up missing data (untested in this domain). 

Choose wisely. 

## Feature Extraction and Selection: 
You have to carefully select features from the meal time series that are discriminatory between meal and no meal classes. 

## Test Data: 
The test data will be a matrix of size N×24, where N is the total number of tests and 24 is the size of the CGM time series. N will have some distribution of meal and no meal data. 

Note here that for meal data you are asked to obtain a 2 hr 30 min time series data, while for no meal  you are taking 2 hr. However, a machine will not take data with different lengths. Hence, in the feature extraction step, you have to ensure that features extracted from both meal and no meal data have the same length. 

## Output format: 
You have to output an N×1 vector of 1s and 0s, where if a row is determined to be meal data, then the corresponding entry will be 1, and if determined to be no meal, the corresponding entry will be 0. 

This vector should be saved in a **“Result.csv”** file. 

Given: 
1. Meal Data and No Meal Data of subjects 1 and 2 
2. Ground truth labels of Meal and No Meal for subjects 1 and 2 

Using Python, train a machine model to recognize whether a sample in the training data set represents a  person who has eaten (Meal), or not eaten (No Meal). The training data set contains ground truth labels of Meal and No Meal for 5 subjects. 

You will need to perform the following tasks: 
1. Extract features from Meal and No Meal training data set. 
2. Make sure that the features are discriminatory. 
3. Train a machine to recognize Meal or No Meal data. 
4. Use k fold cross validation on the training data to evaluate your recognition system. 
5. Write a function that takes a single test sample as input, and outputs 1 if it predicts the test sample as meal or 0 if it predicts test sample as No meal. 
