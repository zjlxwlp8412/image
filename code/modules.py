# encoding: utf-8

import tensorflow as tf

from tensorflow.python.training import moving_averages


def batch_normalization(
        inputs,
        is_training=None,
        reuse=None,
        scope="batch_normalization"
):
    with tf.variable_scope(scope, reuse=reuse):
        params_shape = [inputs.get_shape()[-1]]
        beta = tf.get_variable(
            name='beta',
            shape=params_shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32)
        )
        gamma = tf.get_variable(
            name='gamma',
            shape=params_shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32)
        )
        if is_training:
            mean, variance = tf.nn.moments(inputs, axes=[0, 1, 2], name='moments')
            moving_mean = tf.get_variable(
                name='moving_mean',
                shape=params_shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False
            )
            moving_variance = tf.get_variable(
                name='moving_variance',
                shape=params_shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False
            )
            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, 0.9)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, 0.9)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)
        else:
            mean = tf.get_variable(
                name='moving_mean',
                shape=params_shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0, tf.float32),
                trainable=False
            )
            variance = tf.get_variable(
                name='moving_variance',
                shape=params_shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(1.0, tf.float32),
                trainable=False
            )
        output = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, 1e-8)
        output.set_shape(output.get_shape())
        return output

def conv_layer(
        inputs,
        kernel_size,
        out_channels,
        strides,
        is_training=None,
        activation="relu",
        reuse=None,
        scope="conv_layer"
):
    with tf.variable_scope(scope, reuse=reuse):
        kernel = tf.get_variable(
            name="kernel",
            shape=kernel_size + [inputs.get_shape().as_list()[-1], out_channels],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.001, mean=0.001)
        )
        output = tf.nn.conv2d(
            inputs,
            kernel,
            strides=strides,
            padding="SAME",
            name="conv2d"
        )
        output = batch_normalization(output, is_training=is_training)
        if activation == "relu":
            output = tf.nn.relu(output, name="relu")
        return output


def max_pooling(
        inputs,
        ksize,
        strides,
        reuse=None,
        scope="max_pooling"
):
    with tf.variable_scope(scope, reuse=reuse):
        output = tf.nn.max_pool(
            value=inputs,
            ksize=ksize,
            strides=strides,
            padding="SAME",
            name="max_pool"
        )
        return output


def cnn(
        images,
        num_class,
        is_training=True,
        reuse=None,
        scope="cnn"
):
    """
    :param images: [batch, height, width, channels]
    :param num_class:
    :param reuse:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        output = conv_layer(
            images,
            kernel_size=[3, 3],
            out_channels=32,
            strides=[1, 1, 1, 1],
            is_training=is_training,
            activation="relu",
            scope="conv1"
        ) # [None, 32, 32, 32]
        output = conv_layer(
            output,
            kernel_size=[3, 3],
            out_channels=32,
            strides=[1, 1, 1, 1],
            is_training=is_training,
            activation="relu",
            scope="conv2"
        ) # [None, 32, 32, 32]
        output = max_pooling(
            output,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            scope="max_pool1"
        ) # [None, 16, 16, 32]
        output = conv_layer(
            output,
            kernel_size=[3, 3],
            out_channels=32,
            strides=[1, 1, 1, 1],
            is_training=is_training,
            activation="relu",
            scope="conv3"
        ) # [None, 16, 16, 32]
        output = conv_layer(
            output,
            kernel_size=[3, 3],
            out_channels=32,
            strides=[1, 1, 1, 1],
            is_training=is_training,
            activation="relu",
            scope="conv4"
        ) # [None, 16, 16, 32]
        output = max_pooling(
            output,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            scope="max_pool2"
        ) # [None, 8, 8, 32]
        output = tf.reshape(output, shape=[-1, 8 * 8 * 32], name="flatten")
        output = tf.layers.dense(
            output,
            units=512,
            activation=tf.nn.relu,
            name="fc1"
        )
        output = tf.layers.dense(
            output,
            units=256,
            activation=tf.nn.relu,
            name="fc2"
        )
        output = tf.layers.dense(
            output,
            units=num_class,
            activation=tf.nn.softmax,
            name="output"
        )
        return output


def get_cross_entropy(
        logits,
        labels,
        reuse=None,
        scope="cross_entropy"
):
    with tf.variable_scope(scope, reuse=reuse):
        labels = tf.one_hot(labels, depth=logits.get_shape().as_list()[-1])
        output = tf.reduce_mean(tf.reduce_sum(-labels * tf.log(logits), axis=-1))
        return output


def get_accuracy(
        logits,
        labels,
        reuse=None,
        scope="accuracy"
):
    with tf.variable_scope(scope, reuse=reuse):
        output = tf.argmax(logits, dimension=-1)
        output = tf.reduce_mean(tf.to_float(tf.equal(output, labels)))
        return output

def get_train_op(
        loss,
        global_step,
        lr,
        reuse=None,
        scope="train_op"
):
    with tf.variable_scope(scope, reuse=reuse):
        train_op = tf.train.AdamOptimizer(learning_rate=lr, name="Adam").minimize(
            loss=loss,
            global_step=global_step,
            name="train_op"
        )
        return train_op


def save_model(sess, name):
    saver = tf.train.Saver()
    saver.save(sess, name)


def load_model(sess, name):
    saver = tf.train.Saver()
    saver.restore(sess, name)
