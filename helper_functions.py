import pyspark.sql.functions as F
from itertools import chain
import numpy as np
import json
import urllib.request
import gzip

#method for calculating class weights for classifier
def get_class_weights(dataframe, labelCol):
    label_collect = dataframe.select(labelCol).groupby(labelCol).count().collect()
    unique_labels = [x['label'] for x in label_collect]
    bin_count = [x['count'] for x in label_collect]
    total_labels = sum(bin_count)
    unique_label_count = len(label_collect)
    class_weights = {i: ii for i, ii in zip(unique_labels, total_labels / (unique_label_count * np.array(bin_count)))}
    mapping_expr = F.create_map([F.lit(x) for x in chain(*class_weights.items())])
    dataframe = dataframe.withColumn('weight', mapping_expr.getItem(F.col(labelCol)))
    return dataframe

#method for loading amazon cellphone dataset from source
def load_amazon_cellphones(num_examples):
    link = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz"
    stream = urllib.request.urlopen(link)
    file = gzip.open(stream)
    lines = []
    if num_examples == -1:
        for i, line in enumerate(file):
            lines.append(json.loads(line))
    else:
        for i, line in enumerate(file):
            lines.append(json.loads(line))
            if i == num_examples - 1:
                break
    return lines



