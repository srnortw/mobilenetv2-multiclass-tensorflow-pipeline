# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 02:51:47 2025

@author: Serkan
"""

import c_sql_d

import pdb
import numpy as np
import cv2 as cv
# from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# from concurrent.futures import ThreadPoolExecutor

import argparse

from dotenv import load_dotenv
import os

parser = argparse.ArgumentParser(
    description='preparation_and_training')

parser.add_argument(
    '-dn',
    '--dataset_name',
    default='eurosat')  # eurosat,satelimgslocs,sports

parser.add_argument(
    '-res', '--resolution',
    type=int,
    default=64)

# parser.add_argument("--mode", default='client') when you run directly python console,uncomment this


inputs = parser.parse_args()

dataset_name = inputs.dataset_name
res = inputs.resolution

load_dotenv()
sql_d_o = c_sql_d.c_sql_d_c(os.getenv('DB_HOST'), os.getenv('DB_PORT'), os.getenv('DB_NAME'), os.getenv('DB_USER'),os.getenv('DB_PASSWORD'))

sdaq_com = f'''

SELECT source,COUNT(*) FROM {dataset_name}

GROUP BY source

'''

sdaq_df = sql_d_o.query(sdaq_com)

all_meta_d_com = f'''
SELECT id,source,loc,label FROM {dataset_name}
'''

all_metad_df = sql_d_o.query(all_meta_d_com)

sql_d_o.close_connection()


# from output_classification_preparation import output_classification_preparation
#
# zipped,unique_labels = output_classification_preparation.preparation(all_metad_df, dataset_name)

import pickle
import pandas as pd

lids = pd.factorize(all_metad_df['label'])

all_metad_df['label_id'] = lids[0]

unique_labels = list(lids[1])

with open(f"unique_labels_folder/{dataset_name}_unique_labels.pkl", "wb") as f:
    pickle.dump(unique_labels, f)

all_metad_ds = tf.data.Dataset.from_tensor_slices(dict(all_metad_df))

zipped = all_metad_ds.map(lambda a: (tf.one_hot(a["label_id"], depth=len(unique_labels)),
                                     a)
                          )


# def shp(f, shape):
#     f.set_shape(shape)
#     return f

batch_sz=1000
resh = res
resw = res

import input_preparation
# from input_preparation import inp_prep_f

zipped=input_preparation.inp_prep_f0(zipped,resh,resw,sdaq_df,all_metad_df['loc'])


initial_state=tf.zeros(len(unique_labels),dtype=tf.int32)

# Reduce function
def count_fn(state,inp):
    #indx=md['label_id'].numpy().decode('utf-8')
    md=inp[2]
    return tf.tensor_scatter_nd_add(state, [[md['label_id']]], [1])

# Apply take and reduce
class_counts = zipped.reduce(initial_state=initial_state, reduce_func=count_fn)

ratio=tf.cast(class_counts/tf.reduce_sum(class_counts),tf.float32)
print("Before resampling:", ratio.numpy())

# eq=1/len(unique_labels)
#
# eq=np.zeros(len(unique_labels))+eq

eq = tf.constant([1.0 / len(unique_labels)] * len(unique_labels), dtype=tf.float32)

def class_func(a,b,md):
    return tf.argmax(b,axis=-1)

zipped = (
    zipped
    .rejection_resample(class_func, target_dist=eq,initial_dist=ratio)
    .map(lambda extra_label, features_and_label: features_and_label)).cache()

# for i,j in enumerate(resample_ds.take(50)):
#     print(i)

# Apply take and reduce
resample_counts = zipped.reduce(initial_state=initial_state, reduce_func=count_fn)
resample_counts=tf.cast(resample_counts,tf.int64)

counts=tf.reduce_sum(resample_counts)

ratio1=tf.cast(resample_counts/counts,tf.float32)

print("After resampling :", ratio1.numpy())

zipped=zipped.apply(tf.data.experimental.assert_cardinality(counts))


zipped,q,d_input_shape=input_preparation.inp_prep_f(zipped,batch_sz)

zipped=zipped.cache()

from input_preparation import shp

samq = float(q)#q_rs

perc = 2/100

cv_test_size = int(perc * 2 * samq)


zipped=zipped.map(lambda a,b,c,d,e: (tf.cast(a,tf.float32),b,c,d,e))

zipped=zipped.map(lambda a,b,c,d,e:(a,b,c,d,
                             tf.argsort(tf.norm(d - e[1], axis=1))))


# def xy(a,e):
#     an=tf.gather(a,e)
#     an.set_shape([None] + a.shape[1:])
#     return an

zipped=zipped.map(lambda a,b,c,d,e:(shp(tf.gather(a,e),a.shape),
                             shp(tf.gather(b,e),b.shape),
                             {k: shp(tf.gather(v,e),v.shape) for k, v in c.items()},
                             shp(tf.gather(d,e),d.shape)
                             )
           )



print('order')



print('zipped')


zipped=zipped.unbatch()

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

zipped =zipped.map(lambda a, b, c, d: (a, b, c))#x



zipped=zipped.map(lambda d,y,k:(preprocess_input(d),y,k),num_parallel_calls=tf.data.AUTOTUNE)#x








# zipped_train_traincv = (zippedx.enumerate()
#                         .filter(lambda idx, row: tf.reduce_any(tf.equal(idx, other_indices_train_traincv)))
#                         .map(lambda idx, row: row)
#                         )
#
# zipped_cv_test = (zippedx.enumerate()
#                   .filter(lambda idx, row: tf.reduce_any(tf.equal(idx, closest_indices_cvtest)))
#                   .map(lambda idx, row: row)
#                   )

def func(n,batchsize,stepsize,packs,laststepsize):

    p=n//batchsize

    if p == packs:
        stepsize=laststepsize

    if (n>=p*batchsize) and (n<(p*batchsize)+stepsize):
        return True
    else:
        return False


packs=q//batch_sz#math.ceil(q/batch_sz)

stepsize=tf.cast(batch_sz*2*perc,tf.int64)

par=tf.cast(q/batch_sz,tf.float32)-tf.cast(q//batch_sz,tf.float32)

laststepsize=tf.cast(par*batch_sz*2*perc,tf.int64)


zipped=zipped.enumerate().map(lambda n,x:(x[0],x[1],x[2],
                                       func(n,batch_sz,stepsize,packs,laststepsize)))

zipped_cv_test=zipped.filter(lambda a0,b0,c0,d0:d0).map(lambda a,b,c,d:(a,b,c))
zipped_train_traincv=zipped.filter(lambda a1,b1,c1,d1:~d1).map(lambda a,b,c,d:(a,b,c))


# t=0
# for i in zipped_cv_test:
#     print(t)
#     t+=1

# zipped_train_traincv=zipped.skip(cv_test_size)
# zipped_cv_test=zipped.take(cv_test_size)


q_tr_trcv=int(samq)-cv_test_size
#q_tr_trcv = other_indices_train_traincv.shape[1]  # zipped_train_traincv.reduce(np.int32(0), lambda x, _: x + 1).numpy()

train_traincv_d = zipped_train_traincv.shuffle(q_tr_trcv, seed=42)  # train_traincv_d.cardinality()



# Shuffle and split the dataset
traincv_perc = 1 * perc / (1 - (perc * 2))  # 2 is for cv and test
train_size = int((1 - traincv_perc) * float(q_tr_trcv))

# Split the dataset
train_dataset_d = train_traincv_d.take(train_size)#.cache()

traincv_dataset_d = train_traincv_d.skip(train_size)#.cache()

q_cv_test = cv_test_size  # zipped_cv_test.reduce(np.int32(0), lambda x, _: x + 1).numpy()

cv_test_d = zipped_cv_test.shuffle(q_cv_test, seed=42)

# Shuffle and split the dataset
test_perc = 0.5
cv_size = int((1 - test_perc) *float(q_cv_test))

# Split the dataset
cv_dataset_d = cv_test_d.take(cv_size)#.cache()

test_dataset_d = cv_test_d.skip(cv_size)#.cache()




print('we are starting to create tfrecord file.')
import json


def convert_md_to_json_serializable(md):
    md_serializable = {}
    for key, value in md.items():
        if isinstance(value, tf.Tensor):
            # Convert scalar tensor to native Python type
            value = value.numpy()
            if isinstance(value, bytes):
                value = value.decode('utf-8')  # For string tensors
            elif hasattr(value, 'item'):
                value = value.item()  # For int64, float32 scalars
        md_serializable[key] = value
    return md_serializable


def serialize_example(x, y, md, dataset_name):
    md_serializable = convert_md_to_json_serializable(md)

    feature = {
        'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])),
        'md': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(md_serializable).encode()])),
        # md is dictionary
        'dataset': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dataset_name.encode()]))  # Store dataset name
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


options1 = tf.io.TFRecordOptions(compression_type="GZIP")

os.makedirs('processed_datasets', exist_ok=True)
# Write TFRecord
with tf.io.TFRecordWriter(f"processed_datasets/{dataset_name}{res}_images_and_labels.tfrecord",
                          options=options1) as writer:
    # Process train dataset
    for x, y, md in train_dataset_d:
        serialized = serialize_example(x, y, md, "train")
        writer.write(serialized)

    # Process traincv dataset
    for x, y, md in traincv_dataset_d:
        serialized = serialize_example(x, y, md, "traincv")
        writer.write(serialized)

    # Process cv dataset
    for x, y, md in cv_dataset_d:
        serialized = serialize_example(x, y, md, "cv")
        writer.write(serialized)

    # Process test dataset
    for x, y, md in test_dataset_d:
        serialized = serialize_example(x, y, md, "test")
        writer.write(serialized)