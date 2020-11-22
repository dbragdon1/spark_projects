# About
</br>

This is a set of examples of classification on Amazon Product reviews. 

</br>

# Files
</br>

## `logistic_regression_sentiment.py`

This file builds a simple logistic regression model using a BoW approach. 

The dataset label set ranges from 1.0 to 5.0. For the purpose of this project, I removed neutrally labelled points (3.0) set any point below 3.0 as 0, and any point above 3.0 as 1. 

</br>

## Usage

</br>

Navigate to the directory of the file and run like so:

``` bash
python logistic_regression_sentiment.py
```

**WARNING:** The resulting RDD is very large and the program will not run if you do not have enough memory to allocate. For this reason you can select the amount of points you'd like to train the model on like this:

``` bash
python logistic_regression_sentiment.py 1000
```

This will train the model using 1000 samples. 

**Note:** This file tries to automatically rebalance the classes, since the transformed labelset if very imbalanced in favor of positive (1) labels. If there are many points missing after imbalancing, it is because the model attempted to undersample the data.

Not specifying any limit will let the model train on the entire dataset. 

</br>

## Results

</br>

| Area Under ROC | Area Under PR |
|:--------------:|:-------------:|
|      0.796     |     0.754     |

</br>

## `logistic_regression_sentiment_weighted.py`

</br>
### About

This file builds on the previous file, and implements class-weighting. The method for factoring in class weights (in `scikit-learn` fashion), was found from [this awesome blog post](https://danvatterott.com/blog/2019/11/18/balancing-model-weights-in-pyspark/). This resulted in a dramatic increase in performance from the unweighted model, as displayed in the results section below. The code for implementing class-weighting is found in the `get_class_weights()` method in the `helper_functions.py` file. 

</br>

## Usage

</br>

This file can be used in the same way that `logistic_regression_sentiment.py` is used, but it also implements class weights instead of undersampling the highest class. 

</br>

## Results

</br>

| Area Under ROC | Area Under PR |
|:--------------:|:-------------:|
|      0.841    |     0.955     |

