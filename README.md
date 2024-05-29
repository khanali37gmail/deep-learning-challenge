## DEEP LEARNING
## Report on the Performance of the Deep Learning Model for Alphabet Soup

## Project Overview

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

## Overview of the Analysis
The purpose of this analysis is to develop a deep learning model that can predict the success of charity applications for the Alphabet Soup organization. By analyzing historical data and creating a predictive model, we aim to identify which features contribute most significantly to successful outcomes and improve the efficiency of the application review process.

## Results
Data Preprocessing

Target Variable:

IS_SUCCESSFUL: This binary variable indicates whether a charity application was successful (1) or not (0).

## Feature Variables:

APPLICATION_TYPE
AFFILIATION
CLASSIFICATION
USE_CASE
ORGANIZATION
STATUS
INCOME_AMT
SPECIAL_CONSIDERATIONS
ASK_AMT


## Removed Variables:

-EIN: Employer Identification Number, which is unique to each organization and does not contribute to the predictive power of the model.

-NAME: The name of the organization, which also does not contribute to the prediction.


## Compiling, Training, and Evaluating the Model

Model Architecture:

    -Input Layer:
        -Number of input features: Determined by the length of a single preprocessed sample from the training data.

First Hidden Layer:

    -Neurons: 12
    -Activation Function: ReLU

Second Hidden Layer:

    -Neurons: 6
    -Activation Function: ReLU

Output Layer:

    -Neurons: 1
    -Activation Function: Sigmoid

## Rationale for Architecture:

The architecture was chosen to balance complexity and performance. The number of neurons was selected based on typical practices and initial experimentation, aiming to capture sufficient patterns in the data without overfitting.
Model Performance:

Initial Model Performance:
Loss: 0.55
Accuracy: 0.72
These metrics indicate moderate performance, with room for improvement.

## Steps to Improve Performance:

Hyperparameter Tuning:

    -Experimented with different numbers of neurons in hidden layers.
    -Tried different learning rates and optimizers.

Additional Layers:

    -Added more hidden layers to increase the model's capacity to learn complex patterns.

Regularization:

    -Implemented dropout layers to prevent overfitting.

Increased Epochs:

    -Trained the model for more epochs to ensure better convergence.

## Summary
The deep learning model developed for predicting the success of charity applications achieved moderate performance with an accuracy of approximately 72%. While the model provides a reasonable starting point, further optimization is needed to improve its predictive power.

-Recommendations for Improvement:

    Alternative Models:

        -Consider using ensemble methods like Random Forest or Gradient Boosting, which are often effective for classification problems with structured data.

    Feature Engineering:

        -Generate additional features or transform existing features to better capture underlying patterns.

    Advanced Neural Networks:
    
        -Experiment with more sophisticated neural network architectures, such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs), if applicable.

By leveraging alternative models and additional preprocessing techniques, we can potentially achieve higher accuracy and better overall performance in predicting the success of charity applications. This approach will help Alphabet Soup more efficiently allocate resources and support successful initiatives.

Visual Support
visualizations that support the above findings and steps taken during the analysis were also presented at the end for help better understanding the module.


