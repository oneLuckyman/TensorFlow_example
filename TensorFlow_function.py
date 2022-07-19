# Tensorflow的class和method
# 鼠标放在函数名上不动，可以查看函数的输入参数和例子

import tensorflow as tf 

tf.argmax()         # 找到最大值并返回索引
tf.reduce_max()     # 求最大值
tf.reduce_mean()    # 求平均值
tf.reduce_sum()
tf.one_hot()

tf.keras.Sequential()

x = 0
with tf.GradientTape as tape:
    tape.watch(x)
    y = x * x 
grad = tape.gradient(y, x)      # 求梯度，注意此处的y 必须显含第二个参数

tf.keras.losses.mean_squared_error()    # 均方损失函数
