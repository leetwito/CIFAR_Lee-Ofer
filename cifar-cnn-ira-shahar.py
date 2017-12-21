import tensorflow as tf
import numpy as np
import cifar
import time

def prepare_data():
    train_images, train_cls_res, train_cls_vec = cifar.load_training_data()
    test_images, test_cls_res, test_cls_vec = cifar.load_test_data()
    return train_images, train_cls_vec, test_images, test_cls_vec


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def tensor_to_image(T,channel_num):
    T_single_input = T[0:1,:,:,channel_num:channel_num+1]
    x_min = tf.reduce_min(T_single_input)
    x_max = tf.reduce_max(T_single_input)
    W_0_to_1 = (T_single_input - x_min) / (x_max - x_min)
    W_0_to_255_uint8 = tf.image.convert_image_dtype(W_0_to_1, dtype=tf.uint8)
    return W_0_to_255_uint8
    # W_transposed = tf.transpose(W_0_to_255_uint8, [3, 0, 1, 2])
    # return W_transposed


def deepnn(x):

    # image_size = 32
    conv1_shape = [5,5,3,32]
    conv2_shape = [5,5,32,20]
    conv3_shape = [5,5,20,20]
    fc1_shape = 4 * 4 * 20


    with tf.name_scope('conv1'):
        W_conv1 = weight_variable(conv1_shape)
        b_conv1 = bias_variable([conv1_shape[-1]])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

        for i in range(6):
            h_conv1_images = tensor_to_image(h_conv1,i)
            tf.summary.image('conv1 - output '+str(i), h_conv1_images, max_outputs=16)

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2'):
        W_conv2 = weight_variable(conv2_shape)
        b_conv2 = bias_variable([conv2_shape[-1]])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('conv3'):
        W_conv3 = weight_variable(conv3_shape)
        b_conv3 = bias_variable([conv3_shape[-1]])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    with tf.name_scope('pool3'):
        h_pool3 = max_pool_2x2(h_conv3)



    # with tf.name_scope('dropout'):
    #     keep_prob = tf.placeholder(tf.float32)
    #     h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc1'):
        h_pool3_flat = tf.reshape(h_pool3, [-1, fc1_shape])
        W_fc1 = weight_variable([fc1_shape, 10])
        b_fc1 = bias_variable([10])
        y_conv = tf.matmul(h_pool3_flat, W_fc1) + b_fc1

    return y_conv

def get_batch_data(x,y,batch_size):

    # perm = np.random.permutation(range(x.shape[0]), batch_size)
    batch_idxs = np.random.choice(range(x.shape[0]), batch_size, replace=False)
    batch_x = x[batch_idxs,:,:,:]
    batch_y = y[batch_idxs,:]
    return batch_x, batch_y





def main(_):
    train_images, train_cls_vec, test_images, test_cls_vec = prepare_data()
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    print(train_images.shape, train_cls_vec.shape, test_images.shape, test_cls_vec.shape)

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])

    y_conv = deepnn(x)

    with tf.name_scope('cross_entropy_loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        cross_entropy = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('cross_entropy_loss', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)

        accuracy = tf.reduce_mean(correct_prediction)


    tf.summary.scalar('accuracy', accuracy)


    merged = tf.summary.merge_all()



    Iterations = 500*500
    batch_size = 100

    tolerance = 1e-7
    t0 = time.time()


    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter("logs/train", sess.graph)
        test_writer = tf.summary.FileWriter("logs/test")
        sess.run(tf.global_variables_initializer())
        old_cost = 0.0
        for i in range(Iterations):
            batch_x, batch_y = get_batch_data(train_images, train_cls_vec, batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
            new_cost = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y})

            if(np.square(new_cost-old_cost) < tolerance):
                batch_x_test, batch_y_test = test_images, test_cls_vec
                cost_empirical, accuracy_empirical = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x_test, y_: batch_y_test})
                summary = sess.run(merged, feed_dict={x: batch_x_test, y_: batch_y_test})
                test_writer.add_summary(summary, i)
                print("Iteration {}, Test cost: {}, Test accuracy: {}".format(i,cost_empirical, accuracy_empirical))
                print("tolerance {} reached, stopping".format(tolerance))
                break

            if i % 100 == 0:
                cost_empirical, accuracy_empirical = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x, y_: batch_y})
                summary = sess.run(merged, feed_dict={x: batch_x, y_: batch_y})
                train_writer.add_summary(summary, i)
                print("Iteration {}, time passed: {}, train cost: {}, train accuracy: {}".format(i, time.time() - t0 , cost_empirical, accuracy_empirical))

            if i % 500 == 0:
                batch_x_test, batch_y_test = test_images, test_cls_vec
                cost_empirical, accuracy_empirical = sess.run([cross_entropy, accuracy], feed_dict={x: batch_x_test, y_: batch_y_test})
                summary = sess.run(merged, feed_dict={x: batch_x_test, y_: batch_y_test})
                test_writer.add_summary(summary, i)
                print("Test cost: {}, Test accuracy: {}".format(cost_empirical, accuracy_empirical))



if __name__ == '__main__':
    tf.app.run(main=main)