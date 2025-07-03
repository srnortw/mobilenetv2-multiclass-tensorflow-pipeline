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

unique_labels = sorted(lids[1])

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
zipped,q,d_input_shape=input_preparation.inp_prep_f(zipped,resh,resw,sdaq_df,all_metad_df['loc'],batch_sz)

os.makedirs('processed_datasets', exist_ok=True)

zipped=zipped.cache(f"processed_datasets/{dataset_name}{res}")

zipped=zipped.map(lambda a,b,c,d,e: (tf.cast(a,tf.float32),b,c,d,e))

from input_preparation import shp

samq = float(q)#q_rs

perc = 2/100

cv_test_size = int(perc * 2 * samq)


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
train_dataset_d = train_traincv_d.take(train_size).cache()

traincv_dataset_d = train_traincv_d.skip(train_size).cache()

q_cv_test = cv_test_size  # zipped_cv_test.reduce(np.int32(0), lambda x, _: x + 1).numpy()

cv_test_d = zipped_cv_test.shuffle(q_cv_test, seed=42)

# Shuffle and split the dataset
test_perc = 0.5
cv_size = int((1 - test_perc) *float(q_cv_test))

# Split the dataset
cv_dataset_d = cv_test_d.take(cv_size).cache()

test_dataset_d = cv_test_d.skip(cv_size).cache()


import tensorflow.keras.layers as layers
from tensorflow.keras.applications import MobileNetV2,InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

batch_size=64

train_dataset_d=train_dataset_d.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

traincv_dataset_d=traincv_dataset_d.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)#q_tr_trcv-train_size

cv_dataset_d=cv_dataset_d.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset_d=test_dataset_d.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)




# Load MobileNet with pre-trained weights (ImageNet)
# Set `include_top=False` to exclude the final classification layer
base_model = MobileNetV2(alpha=1,weights='imagenet', include_top=False, input_shape=d_input_shape)


# Optional: Freeze the base model layers to use it as a feature extractor
for layer in base_model.layers:
    layer.trainable = False

#print(base_model.layers[-3])

# for layer in base_model.layers[-10]
base_model.layers[-3].trainable=True

# Add custom layers on top of MobileNet

#x=base_model.get_layer("block_16_project_BN").output
x=base_model.output# base_model.get_layer("block_16_project_BN").output


x = layers.GlobalAveragePooling2D()(x)  # Global average pooling

x = Dense(1024, activation='relu',kernel_initializer=tf.keras.initializers.HeNormal())(x)  # Add a fully connected layer

#x=layers.Dropout(0.5)(x)

# x = Dense(256, activation='relu',kernel_initializer=tf.keras.initializers.HeNormal())(x)

# #x=layers.Dropout(0.25)(x)


predictions = Dense(len(unique_labels), activation='linear',kernel_initializer=tf.keras.initializers.GlorotNormal(),name='last_unit')(x)  # Output layer for class quantity




# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)


# Print the model summary
model.summary()


import datetime

import shutil
# # Load the TensorBoard notebook extension
# %reload_ext tensorboard

# # Clear any logs from previous runs
log_dir = "./logs/fit/"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


optimizer1=tf.keras.optimizers.Adam(learning_rate=5e-6)#1e-5
loss1=tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer1, loss=loss1, metrics=['accuracy'])


# sports64 5e-5 epoch 20,satelimgslocs64 1e-6 epoch 25 maybe less ,eurosat64 5e-6 epoch 25,
train_dataset1=train_dataset_d.map(lambda x,y,k:(x,y))
traincv_dataset1=traincv_dataset_d.map(lambda x,y,k:(x,y))
#test_dataset1=test_dataset.map(lambda x,y,k:(x,y))
#train_dataset_d= train_dataset_d.repeat()


h=model.fit(train_dataset1,validation_data=traincv_dataset1, epochs=20,callbacks=[tensorboard_callback])#25
# to see tensorboard try to enter tensorboard --logdir logs/fit inside of command prompt.


x=np.arange(len(h.history['loss']))
y_train=h.history['loss']
y_val=h.history['val_loss']

import matplotlib.pyplot as plt


plt.plot(x,y_train,color='red')
plt.plot(x,y_val,color='blue')
plt.show()


h1=model.evaluate(cv_dataset_d.map(lambda x,y,k:(x,y)))

h2=model.evaluate(test_dataset_d.map(lambda x,y,k:(x,y)))


from sklearn.metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay
import seaborn as sns

import pandas as pd

datasets=[traincv_dataset_d,cv_dataset_d,test_dataset_d]



# def confusion(pred,des,unique_labels):
#
#
#
#     ok=sorted(list(set(pred) | set(des)))
#
#     oky= [unique_labels[i][:2] for i in ok]
#
#     print(len(des))
#     print(len(pred))
#     #print(len(f))
#     cm = confusion_matrix(des,pred)
#
#     # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=oky)
#
#     # disp.plot(cmap=plt.cm.Blues)
#
#     # plt.show()
#
#
#     cm_df = pd.DataFrame(cm, index=oky, columns=oky)
#     sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
#     plt.xlabel("Predicted")
#     plt.ylabel("Actual")
#     plt.title("Confusion Matrix")
#     plt.show()
#
#     return cm_df


# test_dataset_d1=test_dataset_d.cache()
# cv_dataset_d1=cv_dataset_d.cache()

for dataset in datasets:

    results=dataset.map(lambda x,y,k:(x,y,k,
                                    tf.argmax(y, axis=-1),
                                    tf.argmax(tf.nn.softmax(model(x,training=False)),axis=-1),
                        )).unbatch()#.batch(q)


    # results=results.map(lambda a,b,c,d,e:(a,b,c,d,e,tf.py_function(confusion,[d,e,unique_labels],tf.string))).unbatch()


    des=list(results.map(lambda a,b,c,d,e:d).as_numpy_iterator())
    pred=list(results.map(lambda a, b, c, d, e:e).as_numpy_iterator())

    ok=sorted(list(set(pred) | set(des)))

    oky= [unique_labels[i][:2] for i in ok]

    print(len(des))
    print(len(pred))
    #print(len(f))
    cm = confusion_matrix(des,pred)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=oky)

    # disp.plot(cmap=plt.cm.Blues)

    # plt.show()


    cm_df = pd.DataFrame(cm, index=oky, columns=oky)
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    mists=results.filter(lambda x, y, k, d, p: ~tf.equal(d, p))

    for mist in mists.take(3):

        image=tf.cast((mist[0] + 1.0) * 127.5,tf.uint8)
        plt.imshow(image)
        plt.show()


        print(f'prediction:{unique_labels[mist[4]]}')
        print(f'desired:{unique_labels[mist[3]]}')
        print(f"md:{mist[2]}")


# Save the entire model as a `.keras` zip archive.
model.save(f'models/my_model_{dataset_name}_res{res}.keras')


#
#
#
# print('we are starting to create tfrecord file.')
# import json
#
#
# def convert_md_to_json_serializable(md):
#     md_serializable = {}
#     for key, value in md.items():
#         if isinstance(value, tf.Tensor):
#             # Convert scalar tensor to native Python type
#             value = value.numpy()
#             if isinstance(value, bytes):
#                 value = value.decode('utf-8')  # For string tensors
#             elif hasattr(value, 'item'):
#                 value = value.item()  # For int64, float32 scalars
#         md_serializable[key] = value
#     return md_serializable
#
#
# def serialize_example(x, y, md, dataset_name):
#     md_serializable = convert_md_to_json_serializable(md)
#
#     feature = {
#         'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
#         'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])),
#         'md': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(md_serializable).encode()])),
#         # md is dictionary
#         'dataset': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dataset_name.encode()]))  # Store dataset name
#     }
#
#     example = tf.train.Example(features=tf.train.Features(feature=feature))
#     return example.SerializeToString()
#
#
# options1 = tf.io.TFRecordOptions(compression_type="GZIP")
#
# # Write TFRecord
# with tf.io.TFRecordWriter(f"processed_datasets/{dataset_name}{res}1_images_and_labels.tfrecord",
#                           options=options1) as writer:
#     # Process train dataset
#     for x, y, md in train_dataset_d:
#         serialized = serialize_example(x, y, md, "train")
#         writer.write(serialized)
#
#     # Process traincv dataset
#     for x, y, md in traincv_dataset_d:
#         serialized = serialize_example(x, y, md, "traincv")
#         writer.write(serialized)
#
#     # Process cv dataset
#     for x, y, md in cv_dataset_d:
#         serialized = serialize_example(x, y, md, "cv")
#         writer.write(serialized)
#
#     # Process test dataset
#     for x, y, md in test_dataset_d:
#         serialized = serialize_example(x, y, md, "test")
#         writer.write(serialized)