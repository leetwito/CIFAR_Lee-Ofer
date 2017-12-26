import tensorflow as tf
import numpy as np
# import load_cifar_DS
import numpy as np
import cifar

# def prepare_data():
#     train_images, train_cls_res, train_cls_vec = cifar.load_training_data()
#     test_images, test_cls_res, test_cls_vec = cifar.load_test_data()
#     return train_images, train_cls_vec, test_images, test_cls_vec





def requ(x, derive=False):
    if derive:
        return x
    return np.sign(x) * x ** 2


class Cifar10(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.Y_ = tf.placeholder(tf.float32, [None, 10])
        self.lr = tf.placeholder(tf.float32)
        self.pkeep=tf.placeholder(tf.float32)
        self.max_step = 200000
        self.sess = tf.Session()
        self.batch_size = 100

        # x_image = x.reshape([-1, 28, 28, 1])
        conv1 = self.conv_layer(self.x, 3, 16, tf.nn.relu, filter_size=[5, 5], stride=1, name="conv1")
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
        # layer 2
        conv2 = self.conv_layer(pool1, 16, 20, tf.nn.relu, filter_size=[5, 5], stride=1, name="conv2")
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

        # layer 3
        conv3 = self.conv_layer(pool2, 20, 20, tf.nn.relu, filter_size=[5, 5], stride=1, name="conv3")
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")

        # layer 4
        fc4_input = tf.reshape(pool3, [-1, pool3.shape[1] * pool3.shape[2] * pool3.shape[3]])
        print fc4_input.shape[1]
        self.logits = self.fully_connected(fc4_input, int(fc4_input.shape[1]), 10, tf.identity, name="fc4")
        self.y = tf.nn.softmax(self.logits)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y_))

        is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
        tf.summary.scalar("accuracy",self.accuracy)
        tf.summary.scalar("cross_entropy", self.cross_entropy)
        self. train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        self.merged=tf.summary.merge_all()
        self.batch_size=100
        self.train_images,self.train_cls_vec,self.test_images,self.test_cls_vec=self.prepare_data()
        self.sess = tf.Session()
        self.sess.run(init)
        self.writer_train = tf.summary.FileWriter('./tensorboard/second/train',self.sess.graph)
        self.writer_test= tf.summary.FileWriter('./tensorboard/second/test')
        # self.writer.add_graph(self.sess.graph)

    def prepare_data(self):
        train_images, train_cls_res, train_cls_vec = cifar.load_training_data()
        test_images, test_cls_res, test_cls_vec = cifar.load_test_data()
        return train_images, train_cls_vec, test_images, test_cls_vec

    def get_batch_data(self,x, y, batch_size):
        # perm = np.random.permutation(range(x.shape[0]), batch_size)
        batch_idxs = np.random.choice(range(x.shape[0]), batch_size, replace=False)
        batch_x = x[batch_idxs, :, :, :]
        batch_y = y[batch_idxs, :]
        return batch_x, batch_y


    def train_net(self):
        # images, cls_num, labels = load_cifar_DS.load_training_data()
        for i in range(self.max_step):
            with tf.name_scope("train"):
                X_data, Y_data = self.get_batch_data(self.train_images,self.train_cls_vec,self.batch_size)
                train_dict = {self.x: X_data, self.Y_: Y_data, self.lr: 0.0001, self.pkeep: 0.75}
                self.sess.run(self.train_step, feed_dict=train_dict)
            if i % 10==0:
                train_summ=self.test(X_data,Y_data)
                self.writer_train.add_summary(train_summ,i)
            if i%100==0:
                test_summ=self.test(self.test_images,self.test_cls_vec)
                self.writer_test.add_summary(test_summ,i)

    def test(self,X_data,Y_data):
        with tf.name_scope("test"):
            # X_data, Y_data = self.test_images, self.test_cls_vec
            test_dict = {self.x: X_data, self.Y_: Y_data,self.pkeep:1}
            tf.summary.image("test_images",X_data)
            return self.sess.run(self.merged, feed_dict=test_dict)
            # self.writer.add_summary(summ)

    def conv_layer(self,inputs, channels_num, filter_amount, activation_function, filter_size=[5, 5], stride=1,
                   name="conv"):
        with tf.name_scope(name):
            w = tf.Variable(
                tf.truncated_normal([filter_size[0], filter_size[1], channels_num, filter_amount], stddev=0.1),
                name="W")
            b = tf.Variable(tf.ones([filter_amount]) / 10, name="B")
            tf.summary.histogram("W", w)
            tf.summary.histogram("B", b)
            return activation_function(
                tf.nn.conv2d(inputs, w, strides=[1, stride, stride, 1], padding="SAME") + b)

    def fully_connected(self,inputs, input_degree, output_degree, activation_function, name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([input_degree, output_degree], stddev=0.1), name="W")
            b = tf.Variable(tf.ones([output_degree]) / 10, name="B")
            tf.summary.histogram("W", w)
            tf.summary.histogram("B", b)
            act = tf.nn.dropout(activation_function(tf.matmul(inputs, w) + b), self.pkeep, name=name)
            tf.summary.histogram("activation function", act)
            return act
    # print "entropy {}, acuracy {}".format(c, a)

#
# class AlexNet(object):
#     def __init__(self):
#         # layer 1
#         x = tf.placeholder(tf.float32, [None, 784])
#         self.Y_ = tf.placeholder(tf.float32, [None, 10])
#         self.lr = 0.0001
#         self.max_step = 200000
#
#         x_image = x.reshape([-1, 28, 28, 1])
#         conv1 = conv_layer(x, 1, 48, requ, filter_size=[11, 11], stride=4, name="conv1")
#
#         # layer 2
#         conv2 = conv_layer(conv1, 48, 128, requ, filter_size=[5, 5])
#         pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="pool2")
#
#         # layer 3
#         conv3 = conv_layer(pool2, 128, 192, requ, filter_size=[3, 3])
#         pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="pool3")
#
#         # layer 4
#         conv4 = conv_layer(pool3, 192, 192, requ, filter_size=[3, 3], name="conv4")
#
#         # layer 5
#         conv5 = conv_layer(conv4, 192, 128, requ, filter_size=[3, 3], name="conv5")
#         pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME", name="pool1")
#
#         # layer 6
#         fc6_input = pool5.reshape([-1, pool5.shape[1] * pool5.shape[2] * pool5.shape[3]])
#         fc6 = fully_connected(fc6_input, fc6_input.shape[1], 4096, requ,
#                               pkeep=1, name="fc6")
#
#         # layer 7
#         self.fc7 = fully_connected(fc6, 10, tf.nn.softmax, pkeep=1, name="fc7")
#
#     def train_step(self):
#         return tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy())


def main():
    cifarObj = Cifar10()
    cifarObj.train_net()


if __name__ == "__main__":
    main()
