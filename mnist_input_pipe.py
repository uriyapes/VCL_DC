import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


class MnistDataset(object):
    def __init__(self):
        data_dir = './datasets/MNIST_data'
        mnist = input_data.read_data_sets(data_dir, one_hot=False)
        self.train_set, self.train_labels = mnist.train.images, mnist.train.labels
        self.validation_set, self.validation_labels = mnist.validation.images, mnist.validation.labels
        self.test_set, self.test_labels = mnist.test.images, mnist.test.labels
        self.train_labels = np.int32(self.train_labels)
        self.validation_labels = np.int32(self.validation_labels)
        self.test_labels = np.int32(self.test_labels)

        min_label = np.min(self.train_labels)
        max_label = np.max(self.train_labels)
        self.num_labels = max_label - min_label + 1
        # print self.train_labels.shape

    def prepare_datasets(self):

        # self.ds_train = tf.data.Dataset.from_tensor_slices((self.train_set, self.train_labels)) # BAD since it consumes too much memory (see offical website)
        self.batch_size_ph = tf.placeholder(tf.int64)
        self.seed_ph = tf.placeholder(tf.int64, shape=())

        self._input_data, self.input_labels = tf.placeholder(tf.float32, shape=[None, 784]), tf.placeholder(tf.int32, shape=[None])

        self.ds_train = tf.data.Dataset.from_tensor_slices((self._input_data, self.input_labels))
        self.ds_train = self.ds_train.shuffle(buffer_size=50000, seed=self.seed_ph) # Shuffle before repeat to guarantee data ordering (see https://www.tensorflow.org/performance/datasets_performance)
        self.ds_train = self.ds_train.batch(self.batch_size_ph)
        # self.ds_train = self.ds_train.prefetch(batch_size)
        self.ds_train = self.ds_train.repeat(1)
        # iter = self.ds_train.make_one_shot_iterator()
        # iter = self.ds_train.make_initializable_iterator()
        # features, labels = iter.get_next()

        ds_valid = tf.data.Dataset.from_tensor_slices((self._input_data, self.input_labels))
        ds_valid = ds_valid.batch(self.batch_size_ph)
        ds_valid = ds_valid.repeat(1) # no repeat since we want to control what happens after every epoch

        # ds_test = mnist_dataset.test(data_dir)
        # ds_test = tf.data.Dataset.from_tensor_slices((self.test_set, self.test_labels))
        ds_test = tf.data.Dataset.from_tensor_slices((self._input_data, self.input_labels))
        ds_test = ds_test.batch(self.batch_size_ph)
        ds_test = ds_test.repeat(1) # no repeat since we want to control what happens after every epoch

        # Create reinitializable which can be initialized from multiple different Dataset objects.
        # A reinitializable iterator is defined by its structure. We could use the `output_types` and `output_shapes` properties
        #  of either `ds_train` or `ds_test` here, because they are compatible.
        iter = tf.data.Iterator.from_structure(self.ds_train.output_types, self.ds_train.output_shapes)
        print self.ds_train.output_shapes
        self.features, self.labels = iter.get_next()
        self.training_init_op = iter.make_initializer(self.ds_train)
        self.valid_init_op = iter.make_initializer(ds_valid)
        self.test_init_op = iter.make_initializer(ds_test)


    def get_data(self):
        return self.features

    def get_labels(self):
        return self.labels

    def prepare_train_ds(self, sess, batch_size, ds_seed):
        # Use a predetermined seeds so that data is shuffled randomly each epoch but also it is reproducible
        sess.run(self.training_init_op, feed_dict={self.seed_ph: np.int64(ds_seed), self.batch_size_ph: batch_size, 
                                                   self._input_data: self.train_set, self.input_labels: self.train_labels})

    def prepare_validation_ds(self, sess, batch_size):
        sess.run(self.valid_init_op, feed_dict={self.batch_size_ph: batch_size, self._input_data: self.validation_set, 
                                                self.input_labels: self.validation_labels})

    def prepare_test_ds(self, sess, batch_size):
        sess.run(self.test_init_op, feed_dict={self.batch_size_ph: batch_size, self._input_data: self.test_set,
                                               self.input_labels: self.test_labels})

    def get_dimensions(self):
        instance_attributes = self.ds_train.output_shapes[0][1].value # In Mnist case this equals 784
        if type(instance_attributes) == int:
            num_of_channels = 1
        else:
            assert 0
        return instance_attributes, num_of_channels

    def get_num_of_labels(self):
        return self.num_labels
