# Tensorflow的class和method
# 鼠标放在函数名上不动，可以查看函数的输入参数和例子

import tensorflow as tf 

tf.argmax()         # 找到最大值并返回索引
tf.reduce_max()     # 求最大值
tf.reduce_mean()    # 求平均值
tf.square()
tf.reduce_sum()
tf.one_hot()
tf.cast()
tf.keras.utils.to_categorical()

tf.keras.Sequential()
tf.keras.layers.Dense()
tf.keras.layers.Dropout()
tf.keras.layers.Flatten()
tf.keras.layers.Concatenate()
tf.keras.layers.Conv2D()
tf.keras.layers.MaxPool2D()
tf.keras.layers.MaxPooling2D()
tf.keras.layers.AveragePooling2D()

tf.keras.losses.mean_squared_error()    # 均方损失函数

tf.data.Dataset.from_tensor_slices()
tf.data.Dataset.repeat()
tf.data.Dataset.shuffle()
tf.data.Dataset.batch()
tf.data.Dataset.take()
tf.data.Dataset.map()
tf.data.Dataset.zip()

tf.keras.callbacks.EarlyStopping()
tf.keras.callbacks.TensorBoard()

x = 0
with tf.GradientTape as tape:
    tape.watch(x)
    y = x * x 
grad = tape.gradient(y, x)      # 求梯度，注意此处的y 必须显含第二个参数

