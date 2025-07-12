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
    default='eurosat')#eurosat,satelimgslocs,sports

parser.add_argument(
    '-res','--resolution',
    type=int,
    default=64)



inputs=parser.parse_args()

dataset_name=inputs.dataset_name
res=inputs.resolution

# st='sports'
# Load TFRecord file
import os
os.makedirs('processed_datasets', exist_ok=True)
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

from input_preparation import shp

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

dataset_d=dataset_d.map(lambda x,y,md,name:(shp(x,xs),
                                            shp(y,ys),
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

unique_labels=df.sort_values(by='label_id')['label'].unique()

print(unique_labels)
#x=tf.data.experimental.to_pandas_dataframe(your_dataset)



# Function to filter datasets
def filter_by_type(dataset_type):
    return dataset_d.filter(lambda x, y,md,dtype: tf.equal(dtype, dataset_type)).map(lambda x, y,md,_: (x,y,md)).cache()

# Get separate datasets
train_dataset_d = filter_by_type("train")
traincv_dataset_d = filter_by_type("traincv")
cv_dataset_d = filter_by_type("cv")
test_dataset_d = filter_by_type("test")

# zipped=zipped.cache(f"cached/{dataset_name}{res}")

batch_size=64

train_dataset_d=train_dataset_d.repeat().batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

traincv_dataset_d=traincv_dataset_d.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)#q_tr_trcv-train_size

cv_dataset_d=cv_dataset_d.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

test_dataset_d=test_dataset_d.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


import tensorflow.keras.layers as layers
from tensorflow.keras.applications import MobileNetV2,InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# Load MobileNet with pre-trained weights (ImageNet)
# Set `include_top=False` to exclude the final classification layer
base_model = MobileNetV2(alpha=0.5,weights='imagenet', include_top=False, input_shape=d_input_shape)


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

train_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "train"))
traincv_summary_writer = tf.summary.create_file_writer(os.path.join(log_dir, "traincv"))

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3e-5,#eurosat
    decay_steps=1,
    decay_rate=1,
    staircase=True  # Set True to decay in discrete steps
)


import matplotlib.pyplot as plt

steps = list(range(20))
lrs = [lr_schedule(step) for step in steps]

plt.scatter(steps, lrs)
plt.xlabel("Training step")
plt.ylabel("Learning rate")
plt.title("Exponential Decay")
plt.grid()
plt.show()

optimizer1=tf.keras.optimizers.Adam(learning_rate=lr_schedule)#1e-5
loss1=tf.keras.losses.CategoricalCrossentropy(from_logits=True)


# Define our metrics
train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
train_accuracy = tf.keras.metrics.CategoricalAccuracy('train_accuracy')

traincv_loss = tf.keras.metrics.Mean('traincv_loss', dtype=tf.float32)
traincv_accuracy = tf.keras.metrics.CategoricalAccuracy('traincv_accuracy')

cv_loss = tf.keras.metrics.Mean('cv_loss', dtype=tf.float32)
cv_accuracy = tf.keras.metrics.CategoricalAccuracy('cv_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
test_accuracy = tf.keras.metrics.CategoricalAccuracy('test_accuracy')

# # Compile the model
# model.compile(optimizer=optimizer1, loss=loss1, metrics=['accuracy'])


# sports224 4e-5 epoch 20,satelimgslocs64 1e-6 epoch 25 maybe less ,eurosat64 2e-5 epoch 20,
train_dataset1=train_dataset_d.map(lambda x,y,k:(x,y))
traincv_dataset1=traincv_dataset_d.map(lambda x,y,k:(x,y))
#test_dataset1=test_dataset.map(lambda x,y,k:(x,y))
#train_dataset_d= train_dataset_d.repeat()


# h=model.fit(train_dataset1,validation_data=traincv_dataset1, epochs=20,callbacks=[tensorboard_callback])#25

epochs=20
steps_for_epochs=df.shape[0]//batch_size

@tf.function
def train_step(x,y):
  with tf.GradientTape() as tape:
    # Make a prediction on the batch of images.
    pred = model(x, training=True)
    # Pass the predictions to the loss function.
    loss = loss1(y, pred)

  gradient=tape.gradient(loss,model.trainable_variables)
  optimizer1.apply_gradients(zip(gradient,model.trainable_variables))
  train_loss(loss)
  train_accuracy(y,pred)

@tf.function
def eval(x,y,dsloss,dsacc):
  pred=model(x,training=False)
  loss=loss1(y,pred)
  dsloss(loss)
  dsacc(y,pred)

for epoch in range(epochs):

  for i,(x,y) in enumerate(train_dataset1):
    train_step(x,y)

    if steps_for_epochs == i:
      break



  # Log training loss to TensorBoard
  with train_summary_writer.as_default():
    tf.summary.scalar('loss', train_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

  for x,y in traincv_dataset1:
    eval(x,y,traincv_loss,traincv_accuracy)

  # Log training loss to TensorBoard
  with traincv_summary_writer.as_default():
    tf.summary.scalar('loss', traincv_loss.result(), step=epoch)
    tf.summary.scalar('accuracy', traincv_accuracy.result(), step=epoch)


  template = 'Epoch {}, Loss: {}, Accuracy: {}, traincv Loss: {}, traincv Accuracy: {}'
  print(template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         traincv_loss.result(),
                         traincv_accuracy.result()*100))

  # Reset metrics every epoch
  train_loss.reset_state()
  traincv_loss.reset_state()
  train_accuracy.reset_state()
  traincv_accuracy.reset_state()

# x=np.arange(len(h.history['loss']))
# y_train=h.history['loss']
# y_val=h.history['val_loss']

import matplotlib.pyplot as plt
# # to see tensorboard try to enter tensorboard --logdir logs/fit inside of command prompt.


# plt.plot(x,y_train,color='red')
# plt.plot(x,y_val,color='blue')
# plt.show()

for x,y in cv_dataset_d.map(lambda x,y,k:(x,y)):
  eval(x,y,cv_loss,cv_accuracy)

print(f"cv_loss:{cv_loss.result()} cv_acc:{cv_accuracy.result()}")

for x,y in test_dataset_d.map(lambda x,y,k:(x,y)):
  eval(x,y,test_loss,test_accuracy)

print(f"test_loss:{test_loss.result()} test_acc:{test_accuracy.result()}")


# h1=model.evaluate(cv_dataset_d.map(lambda x,y,k:(x,y)))

# h2=model.evaluate(test_dataset_d.map(lambda x,y,k:(x,y)))


from sklearn.metrics import confusion_matrix , classification_report, ConfusionMatrixDisplay
import seaborn as sns

import pandas as pd

datasets=[traincv_dataset_d,cv_dataset_d,test_dataset_d]

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



