import tensorflow as tf

import h5py

import pdb
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import argparse


parser = argparse.ArgumentParser(
    description='training')

parser.add_argument(
    '-dn',
    '--dataset_name',
    default='sports')#eurosat,satelimgslocs,sports

parser.add_argument(
    '-res','--resolution',
    type=int,
    default=64)



inputs=parser.parse_args()

dataset_name=inputs.dataset_name
res=inputs.resolution

# st='sports'
# Load TFRecord file
dataset = tf.data.TFRecordDataset(f"processed_datasets/{dataset_name}{res}_images_and_labels.tfrecord",compression_type="GZIP")



def parse_example(serialized_example):
    feature_description = {
        'x': tf.io.FixedLenFeature([], tf.string),
        'y': tf.io.FixedLenFeature([], tf.string),
        'md': tf.io.FixedLenFeature([], tf.string),
        'dataset': tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)  # Change dtype if needed
    y = tf.io.parse_tensor(example['y'], out_type=tf.float32)  # Change dtype if needed
    md=example['md']#dictionary .numpy().decode('utf-8')
    #md=json.loads(md_json)
    dataset_name = example['dataset']

    #print()
    # x_shape=[64,64,3]
    # y_shape=[10]

    # x=tf.reshape(x,x_shape)
    # y=tf.reshape(y,y_shape)


    return x, y,md, dataset_name


dataset_d = dataset.map(parse_example,num_parallel_calls=tf.data.AUTOTUNE)




import json

def f1(md):
  #print(md)
  md_json=md.numpy().decode('utf-8')
  md=json.loads(md_json)
  return md#['id']

# x=dataset_d.map(lambda x,y,z,t:
#                tf.numpy_function(f1,[z],Tout=?))#it is dictionary xd


# for i in x:
#   print(i)


for x,y,md,name in dataset_d.skip(100).take(1):
  md_lib=f1(md)
  name=name.numpy().decode()

  xs=x.shape
  ys=y.shape


  print(xs,y)
  print(md_lib,name)


d_input_shape=xs
class_q=ys[0]

dataset_d=dataset_d.map(lambda x,y,md,name:(tf.reshape(x,xs),
                                            tf.reshape(y,ys),
                                            md,name))


dic=dataset_d.map(lambda x,y,md,name:md)


data_list = []
for batch in dic:
  batch=f1(batch)
  # Convert tensors to numpy and flatten if needed
  batch_dict = {key: value for key, value in batch.items()}
  data_list.append(batch_dict)

df = pd.DataFrame(data_list)

print(df)

lioc=sorted(list(set(df['label'])))
print(lioc)
#x=tf.data.experimental.to_pandas_dataframe(your_dataset)



# Function to filter datasets
def filter_by_type(dataset_type):
    return dataset_d.filter(lambda x, y,md,dtype: tf.equal(dtype, dataset_type)).map(lambda x, y,md,_: (x,y,md))

# Get separate datasets
train_dataset = filter_by_type("train")
traincv_dataset = filter_by_type("traincv")
cv_dataset = filter_by_type("cv")
test_dataset = filter_by_type("test")


import matplotlib.pyplot as plt

for i,j,k in train_dataset.take(5):
  i=tf.cast(i,tf.uint8)
  plt.imshow(i)
  plt.show()
  print(j)
  print(f1(k)['label'])
  #print(f1(md))



import tensorflow.keras.layers as layers
from tensorflow.keras.applications import MobileNetV2,InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2


batch_size=64

train_dataset=train_dataset.map(lambda x,y,k:(preprocess_input(x),y,k),num_parallel_calls=tf.data.AUTOTUNE)

train_dataset=train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

traincv_dataset=traincv_dataset.map(lambda x,y,k:(preprocess_input(x),y,k),num_parallel_calls=tf.data.AUTOTUNE)

traincv_dataset=traincv_dataset.batch(1).prefetch(tf.data.experimental.AUTOTUNE)

cv_dataset=cv_dataset.map(lambda x,y,k:(preprocess_input(x),y,k),num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset=test_dataset.map(lambda x,y,k:(preprocess_input(x),y,k),num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.experimental.AUTOTUNE)




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


predictions = Dense(class_q, activation='linear',kernel_initializer=tf.keras.initializers.GlorotNormal(),name='last_unit')(x)  # Output layer for class quantity




# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)


# Print the model summary
model.summary()


import os

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


optimizer1=tf.keras.optimizers.Adam(learning_rate=5e-5)#1e-5
loss1=tf.keras.losses.CategoricalCrossentropy(from_logits=True)

# Compile the model
model.compile(optimizer=optimizer1, loss=loss1, metrics=['accuracy'])


# sports64 5e-5 epoch 20,satelimgslocs64 1e-6 epoch 25 maybe less ,eurosat64 5e-6 epoch 25,
train_dataset1=train_dataset.map(lambda x,y,k:(x,y))
traincv_dataset1=traincv_dataset.map(lambda x,y,k:(x,y))
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


h1=model.evaluate(cv_dataset.map(lambda x,y,k:(x,y)))

h2=model.evaluate(test_dataset.map(lambda x,y,k:(x,y)))


from sklearn.metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay
import seaborn as sns


datasets=[traincv_dataset,cv_dataset,test_dataset]


for dataset in datasets:

  pred=tf.argmax(tf.nn.softmax(model.predict(dataset.map(lambda x,y,z:x))),axis=-1)

  pred=np.squeeze(pred)

  #print(pred)

  # k=list(set(pred))

  # f=[lioc[i] for i in k]


  des=list(dataset.unbatch().map(lambda a,b,c:tf.argmax(b)).as_numpy_iterator())

  ok=sorted(list(set(pred) | set(des)))

  oky= [lioc[i][:2] for i in ok]

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



  pred_ds=tf.data.Dataset.from_tensor_slices(pred)

  mists=tf.data.Dataset.zip((dataset.unbatch(),pred_ds)).filter(lambda x,y:~tf.equal(tf.argmax(x[1]),y))#.map(lambda x,y:x[0],x[1],x[2])

  mists=mists.map(lambda x,y:(x[0],x[1],x[2],y))

  # for i in ex:
  #   print(i)
  #v='label'
  for mist in mists.take(3):

    image=tf.cast((mist[0] + 1.0) * 127.5,tf.uint8)
    plt.imshow(image)
    plt.show()

    print(f'prediction:{lioc[mist[3]]}')
    print(f"md:{f1(mist[2])}")



# Save the entire model as a `.keras` zip archive.
model.save(f'models/my_model_{dataset_name}{res}.keras')


