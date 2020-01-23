# Install-Sequence-for-Anaconda-TensorFlow-and-Keras

1) Install Anaconda
2) GoTo "Anaconda Prompt (Anaconda3)" with Admin rights
3) Update Conda Environment by below command
     conda update -n base -c defaults conda
4) Install keras ( Install keras first before TensorFlow because if you reverse the order of installation of these two softwares then
   sometimes keras downgrade the installed files ( downgrades versions of installed files )
     conda install -c conda-forge keras
5) Install TensorFlow 
     conda install -c anaconda tensorflow-gpu
     

Check if tensorflow-gpu shows GPU list on Laptop

Open Jupyter NoteBook -> New Python3 NoteBook file
Enter below code in cell to check GPU is detected.

Code 1 :
#==================================================================================================
# Ensure you have the latest TensorFlow gpu release installed.
from __future__ import absolute_import, division, print_function, unicode_literals
# from keras import backend as K

import tensorflow as tf
print("GPU Name : " + str(tf.test.gpu_device_name()))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Below line is important.
tf.compat.v1.disable_eager_execution()

# Graph creation.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# Running the operation.
print(sess.run(c))
print("---------------------------------------------")
# Using `Tensor.eval()`.
with tf.compat.v1.Session():
  print (c.eval())

# K.tensorflow_backend._get_available_gpus()

Output :

GPU Name : /device:GPU:0
Num GPUs Available:  1
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: GeForce GTX 1080, pci bus id: 0000:01:00.0, compute capability: 6.1

[[22. 28.]
 [49. 64.]]
---------------------------------------------
[[22. 28.]
 [49. 64.]]
#==================================================================================================

Code 2 :
#==================================================================================================

from tensorflow.python.client import device_lib
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]
print(get_available_devices()) 


OutPut :

['/device:CPU:0', '/device:GPU:0']
#==================================================================================================



