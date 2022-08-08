# Tensorflow的class和method
# 鼠标放在函数名上不动，可以查看函数的输入参数和例子

import tensorboard
import tensorflow as tf 

tf.enable_eager_execution()       

tf.argmax()         # 找到最大值并返回索引
tf.reduce_max()     # 求最大值
tf.reduce_mean()    # 求平均值
tf.square()
tf.reduce_sum()
tf.one_hot()
tf.cast()
tf.keras.utils.to_categorical()

tf.data.Dataset.from_tensor_slices()
tf.data.Dataset.repeat()
tf.data.Dataset.shuffle()
tf.data.Dataset.batch()
tf.data.Dataset.take()
tf.data.Dataset.map()
tf.data.Dataset.zip()

tf.keras.Sequential()
tf.keras.Sequential().trainable_weights()
tf.keras.Sequential().trainable_variables()

tf.keras.layers.Dense()
tf.keras.layers.Dropout()
tf.keras.layers.Flatten()
tf.keras.layers.Concatenate()
tf.keras.layers.Conv2D()
tf.keras.layers.MaxPool2D()
tf.keras.layers.MaxPooling2D()
tf.keras.layers.AveragePooling2D()

tf.keras.losses.mean_squared_error()    # 均方损失函数
tf.keras.losses.BinaryCrossentropy()
tf.keras.losses.CategoricalCrossentropy()
tf.keras.losses.SparseCategoricalCrossentropy()

tf.keras.optimizers.Adam()

tf.keras.metrics.Mean()
tf.keras.metrics.Accuracy()
tf.keras.metrics.CategoricalAccuracy()
tf.keras.metrics.result()

tf.keras.callbacks.EarlyStopping()
tf.keras.callbacks.TensorBoard()
tf.keras.callbacks.LearningRateScheduler()

tf.summary.scalar()
tf.summary.create_file_writer()
tf.summary.create_file_writer().set_as_default()

# tensorboard --logdir logs

x = 0
with tf.GradientTape as tape:
    tape.watch(x)
    y = x * x 
grad = tape.gradient(y, x)      # 求梯度，注意此处的y 必须显含第二个参数

