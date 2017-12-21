import tensorflow as tf
import cifar
import time
import random
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0.0)


BATCH_SIZE = 1000 #(there are 50,000 images in train)


images, cls_res, cls_vec = cifar.load_images()

with tf.name_scope('adam_optimizer'):
    im = images[1]
tf.summary.image('image 2', im)
merged = tf.summary.merge_all()



# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
# lr = tf.placeholder(tf.float32)
lr = 0.001
# test flag for batch norm
# tst = tf.placeholder(tf.bool)
# iter = tf.placeholder(tf.int32)
# dropout probability
# pkeep = tf.placeholder(tf.float32)
# pkeep_conv = tf.placeholder(tf.float32)


# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 16  # first convolutional layer output depth
L = 20  # second convolutional layer output depth
M = 20  # third convolutional layer
N = 10  # fully connected layer

# conv1: 32x32x3 -> 32x32x16
with tf.name_scope('conv1'):
    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
# relu
# pool: 32x32x16 -> 16x16x16
# conv2: 16x16x16 -> 16x16x20
with tf.name_scope('conv2'):
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
# relu
# pool: 16x16x20 -> 8x8x20
# conv3: 8x8x20 -> 8x8x20
with tf.name_scope('conv3'):
    W3 = tf.Variable(tf.truncated_normal([5, 5, L, M], stddev=0.1))
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
# relu
# pool: 8X8x20 -> 4X4x20.

# fc: 8x8x20 -> 1x1x10
with tf.name_scope('fc'):
    W4 = tf.Variable(tf.truncated_normal([M * 4 * 4, N], stddev=0.1))
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
# softmax


# The model
# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases
stride = 1  # output is 28x28
with tf.name_scope('conv1'):
    Y1c = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME', name="conv1") + B1

    Y1r = tf.nn.relu(Y1c, name='relu1')
    Y1p = tf.layers.max_pooling2d(Y1r, 2, 2, name='pool1')

with tf.name_scope('conv2'):
    Y2c = tf.nn.conv2d(Y1p, W2, strides=[1, stride, stride, 1], padding='SAME', name="conv2") + B2
    Y2r = tf.nn.relu(Y2c, name="relu2")
    Y2p = tf.layers.max_pooling2d(Y2r, 2, 2, name="pool2")

with tf.name_scope('conv3'):
    Y3c = tf.nn.conv2d(Y2p, W3, strides=[1, stride, stride, 1], padding='SAME', name="conv3") + B3
    Y3r = tf.nn.relu(Y3c, name="relu3")
    Y3p = tf.layers.max_pooling2d(Y3r, 2, 2, name="pool3")

# reshape the output from the third convolution for the fully connected layer
with tf.name_scope('fc'):
    YY = tf.reshape(Y3p, shape=[-1, 4 * 4 * M], name="reshape")
    Y4fc = tf.matmul(YY, W4) + B4 # Y4fc is Ylogits
    Y = tf.nn.softmax(Y4fc, name='softmax')


# cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Y4fc, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
writer = tf.summary.FileWriter("./logs/cifar10", sess.graph_def)
sess.run(init)


t_start = time.clock()
for epoch in range(2):
    for batch in range(300):
        indices = random.sample(xrange(len(images)), BATCH_SIZE)
        images_batch = images[indices, :, :, :]
        labels_batch = cls_vec[:, indices]
    sess.run(train_step, feed_dict={X: images_batch, Y_: labels_batch})
    summary = sess.run(merged, feed_dict={X: images_batch, Y_: labels_batch})
    test_writer.add_summary(summary, epoch)
    # if batch % 100 == 0:
        # print stuff
t_end = time.clock()
print('Elapsed time ', t_end - t_start)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))
