# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 15:08:20 2025

@author: Serkan
"""

import tensorflow as tf
import pandas as pd

import pickle

def preparation(all_metad_df,dataset_name):

    lids=pd.factorize(all_metad_df['label'])
    
    all_metad_df['label_id']=lids[0]
    
    unique_labels=sorted(lids[1])

    with open(f"output_classification_preparation/{dataset_name}_unique_labels.pkl", "wb") as f:
        pickle.dump(unique_labels, f)
    
    
    all_metad_ds = tf.data.Dataset.from_tensor_slices(dict(all_metad_df))
    
    
    
    zipped=all_metad_ds.map(lambda a: (tf.one_hot(a["label_id"], depth=len(unique_labels)),
                                a)
                     )
    
    return zipped,unique_labels