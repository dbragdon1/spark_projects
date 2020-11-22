# About
This is a set of examples of collaborative filtering and recommender systems with Amazon product reviews

# Files

## `collaborative_filtering_base.py`

## About

This file implements a base collaborative filtering model using Apache Spark's ALS class.

Given a large list of training samples of user-product-review tuples, we will create a model that will try to guess the review that a user will give a product that they have not yet used. 

This file does not impliment any cross-validation or strenuous hyperparameter tuning. It is simply a base-model to contrast later results with. 

## Results:

| Train RMSE | Test RMSE |
|:--------------:|:-------------:|
|      0.286     |     1.699     |

## Interpretation:

On unseen data, we can expect to be off by 1.699 "stars" when trying to predict a user's rating of a product. Train RMSE is much larger than TEST RMSE, meaning we probably overfit model to the training set. 
