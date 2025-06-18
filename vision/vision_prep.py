# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 02:51:47 2025

@author: Serkan
"""

import c_sql_d
from prep import get_all_image_samples
from prep import images_properties
from prep import filtering_images
from prep import images_relationships


import pdb
import numpy as np
import cv2 as cv
#from sklearn.preprocessing import StandardScaler
import tensorflow as tf
#from concurrent.futures import ThreadPoolExecutor

import argparse



parser = argparse.ArgumentParser(
    description='preparation')

parser.add_argument(
    '-dn',
    '--dataset_name',
    default='sports')#eurosat,satelimgslocs,sports

parser.add_argument(
    '-res','--resolution',
    type=int,
    default=64)



parser.add_argument(
  '-hst',
  '--host',
  default='')
    
parser.add_argument(
  '-prt',
  '--port',
  default='')


#parser.add_argument("--mode", default='client') when you run directly python console,uncomment this


inputs=parser.parse_args()

dataset_name=inputs.dataset_name
res=inputs.resolution


sql_d_o=c_sql_d.c_sql_d_c(inputs.host,inputs.port,'learning','postgres','12345')

sdaq_com=f'''
    
SELECT source,COUNT(*) FROM {dataset_name}
    
GROUP BY source
    
'''


sdaq_df=sql_d_o.query(sdaq_com)

    
all_meta_d_com=f'''
SELECT id,source,loc,label FROM {dataset_name}
'''
    
all_metad_df=sql_d_o.query(all_meta_d_com)



sql_d_o.close_connection()



from output_classification_preparation import output_classification_preparation



zipped=output_classification_preparation.preparation(all_metad_df,dataset_name)


gaiso=get_all_image_samples.get_all_img_samples_c()

resh=res
resw=res


# for i in zipped.take(1):
#     print(i[1]['loc'].numpy().decode())


    
locs_ds= tf.data.Dataset.from_tensor_slices(all_metad_df['loc'])

all_imgs=gaiso.read_and_get_all(resh,resw,sdaq_df,locs_ds)#all_metad_df['loc'])



print(all_imgs.shape)



num_bins=256
channelq=all_imgs.shape[-1]
ipo=images_properties.images_properties_c(channelq,num_bins)


all_imgs=tf.data.Dataset.from_tensor_slices(all_imgs)



zipped=tf.data.Dataset.zip((all_imgs, zipped)).map(lambda a,b:(a,
                                                               b[0],
                                                               b[1]))


zipped=zipped.map(lambda a,b,c:(a,
                             b,
                             c,
                             tf.numpy_function(ipo.compute_histogram,[a],Tout=tf.float32)),
                  num_parallel_calls=tf.data.AUTOTUNE)

q=zipped.cardinality()

zipped=zipped.map(lambda a,b,c,d:(a,
                                  b,
                                  c,
                                  ipo.compute_histograms_moments(d)),#tf.numpy_function(ipo.compute_histograms_moments,[d],Tout=tf.float32)),
                  num_parallel_calls=tf.data.AUTOTUNE).batch(q)#all_metad_df.shape[0]


def z_score_norm(x):
    
    u=tf.reduce_mean(x,axis=0)
    
    s=tf.math.reduce_std(x,axis=0)

    imgs_histograms_moments=(x-u)/s
    
    return imgs_histograms_moments,u,s




zipped=zipped.map(lambda a,b,c,d:(a,
                                    b,
                                    c,
                                    z_score_norm(d)
                            )
                  )


stat2_m=zipped.map(lambda a,b,c,d:(d[1],d[2]))


zipped=zipped.map(lambda a,b,c,d:(a,
                                      b,
                                      c,
                                      d[0]
                                      )
                  ).unbatch()



zipped=zipped.apply(tf.data.experimental.assert_cardinality(q))


fio=filtering_images.filtering_images_c()



zipped=zipped.map(lambda a,b,c,d:(a,
                                    b,
                                    c,
                                    d,
                                    tf.numpy_function(fio._augment_image,[a,d,zipped.cardinality()],Tout=tf.uint8)#,[tf.uint8,tf.uint8]
                                    )
                  )#,num_parallel_calls=tf.data.AUTOTUNE)



import matplotlib.pyplot as plt

for org,_,_,_,image in zipped.skip(25).take(5):
    plt.subplot(2,1,1)
    plt.imshow(image)
    plt.subplot(2,1,2)
    plt.imshow(org)
    plt.show()



zipped=zipped.map(lambda a,b,c,d,e:(e,
                                   b,
                                   c,
                                   d
                                   )
               )


zipped=zipped.map(lambda a,b,c,d:(a,
                                    b,
                                    c,
                                    tf.numpy_function(ipo.compute_histogram,[a],Tout=tf.float32)
                                    )
                  ,num_parallel_calls=tf.data.AUTOTUNE)



zipped=zipped.map(lambda a,b,c,d:(a,
                                    b,
                                    c,
                                    d,
                                    ipo.compute_histograms_moments(d)
                                    )
                  ,num_parallel_calls=tf.data.AUTOTUNE).batch(all_metad_df.shape[0])


def z_score_norm_trans(x,y):
    
    u,s=y
    
    return (x-u)/s


zipped=tf.data.Dataset.zip((zipped,stat2_m)).map(lambda a,b:(a[0],
                                                             a[1],
                                                             a[2],
                                                             a[3],
                                                             z_score_norm_trans(a[4],b)
                                                             )
                                                 )#.unbatch()



iro=images_relationships.images_relationships_c()


zipped=zipped.map(lambda a,b,c,d,e :(a,
                                     b,
                                     c,
                                     tf.reshape(d,[tf.shape(d)[0],-1]),
                                     )
                  )



zipped=zipped.map(lambda a,b,c,d :(a,
                                       b,
                                       c,
                                       tf.cast(
                                           tf.numpy_function(iro.compute_histogram_correlation,[d],Tout=tf.float64),
                                           tf.float32)
                                       )
                  )


zipped=zipped.map(lambda a,b,c,d:(a,
                                         b,
                                         c,
                                         d,
                                         tf.numpy_function(iro.anomaly_detection,[d],Tout=tf.bool)
                                         )
                  )



zipped=zipped.map(lambda a,b,c,d,e:(a,
                                          b,
                                          c,
                                          tf.boolean_mask(d,mask=tf.reshape(e, [-1]),axis=1),
                                          e
                                          )
                  )

# for i in zipped:
#     sumy=tf.math.reduce_sum(tf.cast(i[7],tf.int32)) 

# sumy=zipped.map(lambda a,b,c,d,e,f,g,h:tf.math.reduce_sum(tf.cast(h,tf.int32)))


# for i in sumy:
#     print(i)
    
    
zipped=zipped.unbatch()

# zipped=zipped.apply(tf.data.experimental.assert_cardinality(q))


zipped=zipped.filter(lambda a,b,c,d,e:e)


print('counting..')

q_rs=zipped.reduce(np.int32(0), lambda x, _: x + 1).numpy()

zipped=zipped.apply(tf.data.experimental.assert_cardinality(q_rs))



zipped=zipped.map(lambda a,b,c,d,e:(tf.cast(a,tf.float32),
                                    b,
                                    c,
                                    tf.cast(d,tf.float32)
                                    )
                  )



zipped=zipped.batch(zipped.cardinality()).map(lambda a,b,c,d:(a,
                                                              b,
                                                              c,
                                                              tf.cast(z_score_norm(d)[0],tf.float32)
                                                      )
                              )


# iro.cluster_quantity_test(correlation_matrix)

num_clusters=tf.constant(1,dtype=tf.int32)



zipped_order=zipped.map(lambda a,b,c,d:(d,
                                          tuple(
                                              tf.numpy_function(iro.cluster_images_kmeans,
                                                                [d,num_clusters],
                                                                Tout=[tf.int32,tf.float32,tf.float64])
                                              )
                                          )
                  )


zipped_pca=zipped_order.map(lambda d,i:
                                      tf.numpy_function(iro.pca,[d,i[1],i[2],num_clusters],Tout=tf.float32)
                          )


print('pca')
for _ in zipped_pca:#.map(lambda a,b,c,d,e,f,g,h,i,j:a) :
    print('2D pca has created')#samq=i[0].shape[0]
    
samq=q_rs


perc=2

cv_test_size=perc*2*samq//100

    
order=zipped_order.map(lambda d,i:tf.cast(tf.argsort(tf.norm(d - i[1], axis=1),axis=0),tf.float32))

print('order')
    
other_indices_train_traincv=order.unbatch().skip(cv_test_size).batch(samq).map(lambda x:tf.sort(x))

other_indices_train_traincv=tf.convert_to_tensor(list(other_indices_train_traincv),dtype=tf.int64)


closest_indices_cvtest=order.unbatch().take(cv_test_size).batch(samq).map(lambda x:tf.sort(x))

closest_indices_cvtest=tf.convert_to_tensor(list(closest_indices_cvtest),dtype=tf.int64)


print('zipped')

for i in zipped :
    print(i)


zipped=zipped.unbatch().map(lambda a,b,c,d:(a,b,c))


zipped_train_traincv=(zipped.enumerate()
        .filter(lambda idx, row: tf.reduce_any(tf.equal(idx,other_indices_train_traincv)))
        .map(lambda idx,row : row)
        )


zipped_cv_test=(zipped.enumerate()
        .filter(lambda idx, row: tf.reduce_any(tf.equal(idx,closest_indices_cvtest)))
        .map(lambda idx,row : row)
        )


q_tr_trcv=other_indices_train_traincv.shape[1]#zipped_train_traincv.reduce(np.int32(0), lambda x, _: x + 1).numpy()

train_traincv_d=zipped_train_traincv.shuffle(q_tr_trcv,seed=42)#train_traincv_d.cardinality()


perc=2

perc=perc/100

# Shuffle and split the dataset
traincv_perc = 1*perc/(1-(perc*2)) # 2 is for cv and test
train_size = int((1 - traincv_perc) * q_tr_trcv)


# Split the dataset
train_dataset_d = train_traincv_d.take(train_size)

    
traincv_dataset_d = train_traincv_d.skip(train_size)


q_cv_test=closest_indices_cvtest.shape[1]#zipped_cv_test.reduce(np.int32(0), lambda x, _: x + 1).numpy()

cv_test_d=zipped_cv_test.shuffle(q_cv_test,seed=42)


# Shuffle and split the dataset
test_perc = 0.5
cv_size = int((1 - test_perc) *q_cv_test)


# Split the dataset
cv_dataset_d = cv_test_d.take(cv_size)

test_dataset_d =cv_test_d.skip(cv_size)



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

def serialize_example(x,y,md,dataset_name):
    
    md_serializable = convert_md_to_json_serializable(md)
    
    feature = {
        'x': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(x).numpy()])),
        'y': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(y).numpy()])),
        'md': tf.train.Feature(bytes_list=tf.train.BytesList(value=[json.dumps(md_serializable).encode()])),#md is dictionary
        'dataset': tf.train.Feature(bytes_list=tf.train.BytesList(value=[dataset_name.encode()]))  # Store dataset name
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


options1 = tf.io.TFRecordOptions(compression_type="GZIP")

# Write TFRecord
with tf.io.TFRecordWriter(f"processed_datasets/{dataset_name}{res}_images_and_labels.tfrecord",options=options1) as writer:
    # Process train dataset
    for x,y,md in train_dataset_d:
        serialized = serialize_example(x, y,md, "train")
        writer.write(serialized)
    
    # Process traincv dataset
    for x,y,md in traincv_dataset_d:
        serialized = serialize_example(x, y,md, "traincv")
        writer.write(serialized)
    
    # Process cv dataset
    for x,y,md in cv_dataset_d:
        serialized = serialize_example(x, y,md, "cv")
        writer.write(serialized)

    # Process test dataset
    for x,y,md in test_dataset_d:
        serialized = serialize_example(x, y,md, "test")
        writer.write(serialized)
