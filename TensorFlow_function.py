# Tensorflow的class和method
# 鼠标放在函数名上不动，可以查看函数的输入参数和例子

# import tensorboard
import tensorflow as tf 

tf.keras.Model()
tf.keras.Input()

tf.enable_eager_execution()
tf.Tensor()
tf.Tensor.numpy()
tf.Variable()
tf.Variable.assign()
tf.Variable.assign_add()
tf.Variable.assign_sub()
tf.Variable.read_value()
tf.convert_to_tensor()

tf.matmul()
tf.constant()
tf.add()
tf.multiply()
tf.argmax()         # 找到最大值并返回索引
tf.reduce_max()     # 求最大值
tf.reduce_mean()    # 求平均值
tf.square()
tf.reduce_sum()
tf.one_hot()
tf.cast()
tf.expand_dims()
tf.newaxis()
tf.keras.utils.to_categorical()
tf.ones()
tf.zeros()

tf.io.read_file()
tf.image.decode_image()
tf.image.decode_jpeg()
tf.image.resize()

tf.data.Dataset.from_tensor_slices()
tf.data.experimental.AUTOTUNE()
tf.data.Dataset.repeat()
tf.data.Dataset.shuffle()
tf.data.Dataset.batch()
tf.data.Dataset.take()
tf.data.Dataset.skip()
tf.data.Dataset.map()
tf.data.Dataset.zip()
tf.data.Dataset.prefetch()

tf.keras.Sequential()
tf.keras.Sequential().trainable_weights()
tf.keras.Sequential().trainable_variables()

tf.keras.layers.get_shape()
tf.keras.layers.Dense()
tf.keras.layers.Dropout()
tf.keras.layers.Flatten()
tf.keras.layers.Concatenate()
tf.keras.layers.Conv2D()
tf.keras.layers.MaxPool2D()
tf.keras.layers.MaxPooling2D()
tf.keras.layers.AveragePooling2D()
tf.keras.layers.GlobalAveragePooling2D()
tf.keras.layers.GlobalMaxPooling2D()
tf.keras.layers.Layer
tf.keras.layers.Layer.add_weight()

tf.random_normal_initializer()
tf.zeros_initializer()

tf.keras.losses.mean_squared_error()    # 均方损失函数
tf.keras.losses.binary_crossentropy()
tf.keras.losses.BinaryCrossentropy()
tf.keras.losses.categorical_crossentropy()
tf.keras.losses.CategoricalCrossentropy()
tf.keras.losses.sparse_categorical_crossentropy()
tf.keras.losses.SparseCategoricalCrossentropy()

tf.keras.optimizers.Adam()

tf.keras.metrics.Mean()
tf.keras.metrics.Mean().result()
tf.keras.metrics.Mean().result().numpy()
tf.keras.metrics.Mean().reset_states()
tf.keras.metrics.Accuracy()
tf.keras.metrics.categorical_accuracy()
tf.keras.metrics.CategoricalAccuracy()
tf.keras.metrics.sparse_categorical_accuracy()
tf.keras.metrics.SparseCategoricalAccuracy()
tf.keras.metrics.result()

tf.keras.callbacks.EarlyStopping()
tf.keras.callbacks.TensorBoard()
tf.keras.callbacks.LearningRateScheduler()
tf.keras.callbacks.ModelCheckpoint()

tf.summary.scalar()
tf.summary.create_file_writer()
tf.summary.create_file_writer().set_as_default()

tf.keras.applications.MobileNetV2()

tf.train.Checkpoint()
tf.train.latest_checkpoint()
tf.train.Checkpoint().restore()

# tensorboard --logdir logs

x = 0
with tf.GradientTape as tape:
    tape.watch(x)
    y = x * x 
grad = tape.gradient(y, x)      # 求梯度，注意此处的y 必须显含第二个参数

