# Purpose
These are some pyspark demonstrations for NLP purposes.

All models are saved to the `\models` directory. 
# Files

`logistic_regression_sentiment.py`

This file builds a simple logistic regression model using a BoW approach. 

The dataset is collected from [Professor Julian McAuley's Amazon product dataset](https://jmcauley.ucsd.edu/data/amazon/). This specific subset is titled "Cell Phones and Accessories". 

The dataset label set ranges from 1.0 to 5.0. For the purpose of this project, I removed neutrally labelled points (3.0) set any point below 3.0 as 0, and any point above 3.0 as 1. 

## Usage

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



