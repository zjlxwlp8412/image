# encoding: utf-8

import tensorflow as tf
from utils import cifar10
from code import parameters as params
from code.modules import cnn, get_cross_entropy, get_accuracy, get_train_op, save_model, load_model
import os
import numpy as np


def train():
    train_images, train_labels = cifar10.load_data("train")
    train_images = train_images / 255.0

    test_images, test_labels = cifar10.load_data("test", 2000)
    test_images = test_images / 255.0

    image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="image")
    label = tf.placeholder(dtype=tf.int64, shape=[None], name="label")

    output = cnn(image, num_class=10, is_training=True)

    cross_entropy = get_cross_entropy(output, label)
    accuracy = get_accuracy(output, label)
    global_step = tf.train.get_or_create_global_step()

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = get_train_op(loss=cross_entropy, global_step=global_step, lr=0.0001)

    with tf.Session() as sess:
        if os.path.exists(params.model_dir + "/" + "checkpoint"):
            load_model(sess, params.model_dir + "/" + params.model_name)
        else:
            sess.run(tf.global_variables_initializer())
        bst_acc = accuracy.eval(feed_dict={image: test_images, label: test_labels})
        for epoch in range(params.num_epochs):
            batches = cifar10.get_batches(train_images, train_labels, params.batch_size)
            for batch in batches:
                _, loss_val, acc_val, global_step_val = sess.run(
                    [
                        train_op,
                        cross_entropy,
                        accuracy,
                        global_step
                    ],
                    feed_dict={
                        image: batch[0],
                        label: batch[1]
                    }
                )
                if global_step_val % 50 == 0:
                    print(
                        "epoch: %d step: %d train_loss: %.4f train_acc: %.4f" % (
                            epoch + 1,
                            global_step_val,
                            loss_val,
                            acc_val
                        )
                    )
            cur_acc = accuracy.eval(feed_dict={image: test_images, label: test_labels})
            if cur_acc > bst_acc:
                save_model(sess, params.model_dir + "/" + params.model_name)
                print(
                    "Congratulations!!! The accuracy is improved from %.4f to %.4f" %
                    (
                        bst_acc,
                        cur_acc
                    )
                )
                bst_acc = cur_acc
            else:
                print(
                    "Sorry!!! The accuracy is NOT improved by %.4f." % cur_acc
                )


def inference():
    test_images, test_labels = cifar10.load_data("test", 100)
    test_images = test_images / 255.0

    image = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 3], name="image")
    output = cnn(image, num_class=10, is_training=False)
    pred = tf.argmax(output, axis=-1)

    with tf.Session() as sess:
        load_model(sess, params.model_dir + "/" + params.model_name)
        pred_result = []
        true_reuslt = []
        for i in range(20):
            test_image = test_images[i, :, :, :]
            pred_result.append(pred.eval(feed_dict={image: np.expand_dims(test_image, axis=0)})[0])
            true_reuslt.append(test_labels[i])
        print(pred_result)
        print(true_reuslt)


if __name__ == "__main__":
    inference()

