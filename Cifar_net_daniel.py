import tensorflow as tf
import numpy as np
# import load_cifar_DS
import numpy as np
import cifar
import os


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
        self.pkeep = tf.placeholder(tf.float32)
        self.max_step = 200000
        self.sess = tf.Session()
        self.batch_size = 100
        self.files_and_labels = []
        self.num_to_label={}

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
        tf.summary.scalar("accuracy", self.accuracy)
        tf.summary.scalar("cross_entropy", self.cross_entropy)
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.cross_entropy)

        # init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()
        self.merged = tf.summary.merge_all()
        self.batch_size = 100
        # self.train_images, self.train_cls_vec, self.test_images, self.test_cls_vec = self.prepare_data()
        self.sess = tf.Session()
        self.sess.run(init)
        self.writer_train = tf.summary.FileWriter('./tensorboard/second/train', self.sess.graph)
        self.writer_test = tf.summary.FileWriter('./tensorboard/second/test')
        # self.writer.add_graph(self.sess.graph)

    def prepare_data(self):
        train_images, train_cls_res, train_cls_vec = cifar.load_training_data()
        test_images, test_cls_res, test_cls_vec = cifar.load_test_data()
        return train_images, train_cls_vec, test_images, test_cls_vec

    def get_batch_data(self, x, y, batch_size):
        # perm = np.random.permutation(range(x.shape[0]), batch_size)
        batch_idxs = np.random.choice(range(x.shape[0]), batch_size, replace=False)
        batch_x = x[batch_idxs, :, :, :]
        batch_y = y[batch_idxs, :]
        return batch_x, batch_y

    def get_translation_dict(self, translation_file, wnids_file):
        with open(translation_file,'r') as f:
            txt=f.read()
            lines = txt.split('\n')
            with open(wnids_file, 'r') as f2:
                wnids_txt=f2.read()
                for line in lines:
                    num,label=line.split('\t')
                    if num in wnids_txt:
                        self.num_to_label[num] = label


    def get_images_and_labels(self, directory):
        directories = os.listdir(directory)
        self.files_and_labels = []
        for dir in directories:
            for img in os.listdir(os.path.join(directory, dir, "images")):
                if img.lower().endswith('.jpeg'):
                    self.files_and_labels.append((os.path.join(directory, dir,"images", img), self.num_to_label[dir]))

        filenames, labels = zip(*self.files_and_labels)
        filenames = list(filenames)
        labels = list(labels)
        unique_labels = list(set(labels))

        # label_to_int = {}
        # for i, label in enumerate(unique_labels):
        #     label_to_int[label] = i
        #
        # labels = [label_to_int[l] for l in labels]

        return filenames, labels

    def read_images_from_disk(self,input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """
        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        example = tf.image.decode_jpeg(file_contents, channels=3)
        return example, label

    def train_net(self):
        # images, cls_num, labels = load_cifar_DS.load_training_data()
        for i in range(self.max_step):
            with tf.name_scope("train"):
                X_data, Y_data = self.get_batch_data(self.train_images, self.train_cls_vec, self.batch_size)
                train_dict = {self.x: X_data, self.Y_: Y_data, self.lr: 0.0001, self.pkeep: 0.75}
                self.sess.run(self.train_step, feed_dict=train_dict)
            if i % 10 == 0:
                train_summ = self.test(X_data, Y_data)
                self.writer_train.add_summary(train_summ, i)
            if i % 100 == 0:
                test_summ = self.test(self.test_images, self.test_cls_vec)
                self.writer_test.add_summary(test_summ, i)

    def test(self, X_data, Y_data):
        with tf.name_scope("test"):
            # X_data, Y_data = self.test_images, self.test_cls_vec
            test_dict = {self.x: X_data, self.Y_: Y_data, self.pkeep: 1}
            tf.summary.image("test_images", X_data)
            return self.sess.run(self.merged, feed_dict=test_dict)
            # self.writer.add_summary(summ)

    def conv_layer(self, inputs, channels_num, filter_amount, activation_function, filter_size=[5, 5], stride=1,
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

    def fully_connected(self, inputs, input_degree, output_degree, activation_function, name="fc"):
        with tf.name_scope(name):
            w = tf.Variable(tf.truncated_normal([input_degree, output_degree], stddev=0.1), name="W")
            b = tf.Variable(tf.ones([output_degree]) / 10, name="B")
            tf.summary.histogram("W", w)
            tf.summary.histogram("B", b)
            act = tf.nn.dropout(activation_function(tf.matmul(inputs, w) + b), self.pkeep, name=name)
            tf.summary.histogram("activation function", act)
            return act
    # print "entropy {}, acuracy {}".format(c, a)


def main():
    cifarObj = Cifar10()
    # cifarObj.read_all_images('/home/razor30/PycharmProjects/VGG/imagenet/tiny-imagenet-200/train')
    # cifarObj.train_net()
    cifarObj.get_translation_dict('/home/razor30/PycharmProjects/VGG/imagenet/tiny-imagenet-200/words.txt','/home/razor30/PycharmProjects/VGG/imagenet/tiny-imagenet-200/wnids.txt')
    images,labels= cifarObj.get_images_and_labels('/home/razor30/PycharmProjects/VGG/imagenet/tiny-imagenet-200/train')
    images = tf.convert_to_tensor(images, dtype=tf.string)
    # print images[0]
    labels = tf.convert_to_tensor(labels, dtype=tf.string)
    input_queue = tf.train.slice_input_producer([images, labels],
                                               num_epochs=3,
                                               shuffle=True)
    image, label = cifarObj.read_images_from_disk(input_queue)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=10)

    print image_batch,label_batch

    print "hi"
if __name__ == "__main__":
    main()
