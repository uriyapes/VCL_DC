import math
import tensorflow as tf
import parse_image_seg
import nearest_neighbor
import numpy as np

class Model(object):
    def __init__(self, embed_vec_size=256, num_of_layers=4):
        self.embed_vec_size = embed_vec_size
        self.reshuffle_flag = True

    def get_dataset(self, dataset_dict):
        if dataset_dict['name'] == "image_segmentation":
            self.dataset = parse_image_seg.Dataset(dataset_dict)
        else:
            assert(0)

    def build_model(self):
        T, D = self.dataset.get_dimensions()
        num_labels = self.dataset.get_num_of_labels()
        self.batch_size = 20
        # if minibatch size is bigger than train dataset size then make minibatch the same size as dataset size and
        # disable reshuffling every epoch.
        if self.batch_size > self.dataset.train_labels.shape[0]:
            self.batch_size = self.dataset.train_labels.shape[0]
            self.reshuffle_flag = False

        num_hidden = self.embed_vec_size
        self.learning_rate = 0.01
        self.learning_rate_update_at_epoch = 200
        self.learning_rate_updated = 1e-3

        # Input data.
        self.tf_train_minibatch = tf.placeholder(
            tf.float32, shape=(self.batch_size, T), name="train_minibatch_placeholder")
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, num_labels),
                                              name="train_labels_placeholder")
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate_ph = tf.placeholder(tf.float32)

        self.tf_train_dataset = tf.constant(self.dataset.get_train_set())
        if self.dataset.validation_set_exist:
            self.tf_valid_dataset = tf.constant(self.dataset.get_validation_set())
        tf_test_dataset = tf.constant(self.dataset.get_test_set())



        # He. initalizer AKA "MSRA initialization"
        he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)

        layer1_weights = tf.get_variable('layer1_weights', shape=[T, num_hidden], dtype=tf.float32, initializer=he_init)
        layer1_biases = tf.get_variable('layer1_bias', shape=[num_hidden], dtype=tf.float32, initializer=tf.zeros_initializer())

        layer2_weights = tf.get_variable('layer2_weights', shape=[num_hidden, num_hidden], dtype=tf.float32, initializer=he_init)
        layer2_biases = tf.get_variable('layer2_bias', shape=[num_hidden], dtype=tf.float32, initializer=tf.zeros_initializer())

        layer3_weights = tf.get_variable('layer3_weights', shape=[num_hidden, num_hidden], dtype=tf.float32, initializer=he_init)
        layer3_biases = tf.get_variable('layer3_bias', shape=[num_hidden], dtype=tf.float32, initializer=tf.zeros_initializer())

        layer4_weights = tf.get_variable('layer4_weights', shape=[num_hidden, num_labels], dtype=tf.float32, initializer=he_init)
        layer4_biases = tf.get_variable('layer4_bias', shape=[num_labels], dtype=tf.float32, initializer=tf.zeros_initializer())

        # I want to have the weights explictly so don't use dense
        # input = tf.layers.dense(self.tf_train_minibatch, num_hidden, kernel_initializer=he_init,
        #                         activation=self._activation, name='layer{}'.format(i + 1))

        # Model.
        def model(data):
            hidden_no_relu = tf.matmul(data, layer1_weights) + layer1_biases
            hidden = tf.nn.relu(hidden_no_relu)

            hidden_no_relu = tf.matmul(hidden, layer2_weights) + layer2_biases
            hidden = tf.nn.relu(hidden_no_relu)

            hidden_no_relu = tf.matmul(hidden, layer3_weights) + layer3_biases
            hidden = tf.nn.relu(hidden_no_relu)
            drop_out = tf.nn.dropout(hidden, self.keep_prob)

            output = tf.matmul(drop_out, layer4_weights) + layer4_biases
            return output

        # Training computation.
        # Predictions for the training, validation, and test data.
        logits = model(self.tf_train_minibatch)
        self.train_prediction = tf.nn.softmax(logits)

        train_logits_full_set = model(self.tf_train_dataset)
        self.train_prediction_full_set = tf.nn.softmax(train_logits_full_set)

        if self.dataset.validation_set_exist:
            valid_logits = model(self.tf_valid_dataset)
            self.valid_prediction = tf.nn.softmax(valid_logits)

        self.test_logits = model(tf_test_dataset)
        self.test_prediction = tf.nn.softmax(self.test_logits)

        # TODO: change to softmax_cross_entropy_with_logits_v2 when tf is updated
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=logits))
        # Optimizer.
        self.optimizer = self.build_optimizer(self.learning_rate_ph)


    def train_model(self):
        self.epoch = 0
        step = 0
        self.initial_train_labels = np.copy(self.dataset.get_train_labels())

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        with self.sess.as_default():
            print('Initialized')
            self.mini_batch_step = 0
            while self.epoch < 500:
                step += 1
                batch_data, batch_labels = self.get_mini_batch()
                feed_dict = {self.tf_train_minibatch : batch_data, self.tf_train_labels : batch_labels,
                             self.keep_prob : 0.5, self.learning_rate_ph: self.learning_rate}
                _, l, predictions = self.sess.run(
                    [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (step % 100 == 0):
                    print('batch_labels: {}'.format(np.argmax(batch_labels, 1)))
                    print('predictions: {}'.format(np.argmax(predictions, 1)))
                    print('Minibatch loss at step %d: %f' % (step, l))
                    print('Minibatch accuracy: %.3f' % self.accuracy(predictions, batch_labels))
                    self.eval_validation_accuracy()

        self.dataset.count_classes_for_all_datasets()
        #self.dataset.count_classes(batch_labels)
        print "Training stopped at epoch: %i" % self.epoch


    def eval_model(self):
        assert ~np.array_equal(self.initial_train_labels, self.dataset.get_train_labels())
        test_labels = self.dataset.get_test_labels()
        with self.sess.as_default():
            print('Train accuracy: %.3f' % self.accuracy(
                self.train_prediction_full_set.eval(feed_dict={self.keep_prob : 1}), self.initial_train_labels))

            self.eval_validation_accuracy()

            test_pred_eval = self.test_prediction.eval(feed_dict={self.keep_prob : 1})
            network_acc = self.accuracy(test_pred_eval, test_labels)
            print('Test accuracy of the network classifier: %.3f' % network_acc)
        return self.dataset.get_test_set(), (np.argmax(self.dataset.get_test_labels(), 1) + 1), network_acc, test_pred_eval

    def get_mini_batch(self):
        offset = (self.mini_batch_step * self.batch_size)
        if (offset + self.batch_size) > self.dataset.train_labels.shape[0]:
            offset = 0
            self.mini_batch_step = 0
            self.new_epoch_update()

        batch_data = self.dataset.train_set[offset:(offset + self.batch_size), :]
        batch_labels = self.dataset.train_labels[offset:(offset + self.batch_size), :]
        self.mini_batch_step += 1
        return batch_data, batch_labels

    def new_epoch_update(self):
        self.epoch += 1
        if self.learning_rate_update_at_epoch == self.epoch:
            self.learning_rate = self.learning_rate_updated
        if self.reshuffle_flag:
            self.dataset.re_shuffle()

    @staticmethod
    def accuracy(predictions, labels):
        assert not np.array_equal(labels, None)
        return (1.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    def eval_validation_accuracy(self):
        if self.dataset.validation_set_exist:
            print('Validation accuracy: %.3f' % self.accuracy(
                self.valid_prediction.eval(feed_dict={self.keep_prob : 1}), self.dataset.get_validation_labels()))
        else:
            print 'Validation accuracy: Nan - no validation set, IGNORE'

    def run_baseline(self, train_set, train_labels, test_set, test_labels):
        nn = nearest_neighbor.NearestNeighbor()
        return nn.compute_one_nearest_neighbor_accuracy(train_set, train_labels, test_set, test_labels)

    def build_optimizer(self, init_learning_rate):
        # return tf.train.AdamOptimizer(init_learning_rate).minimize(self.loss)
        return tf.train.GradientDescentOptimizer(init_learning_rate).minimize(self.loss)


if __name__ == '__main__':
    FILENAME_TRAIN = r'datasets/image-segmentation/segmentation.data'
    FILENAME_TEST = r'datasets/image-segmentation/segmentation.test'
    assert_values_flag = True
    dataset_dict = {'name': 'image_segmentation', 'file_names': (FILENAME_TRAIN, FILENAME_TEST),
                    'assert_values_flag': assert_values_flag, 'validation_percentage': 0.0}

    model = Model(embed_vec_size=256, num_of_layers=4)
    model.get_dataset(dataset_dict)
    # arabic_model.dataset.pca_scatter_plot(arabic_model.dataset.test_set)
    # print('1NN Baseline accuarcy: %.3f' % arabic_model.run_baseline(arabic_model.dataset.train_set,
    #                                                                 arabic_model.dataset.train_labels,
    #                                                                 arabic_model.dataset.test_set,
    #                                                                 arabic_model.dataset.test_labels))
    model.build_model()
    model.train_model()
    test_set, test_labels, network_acc, test_pred_eval = model.eval_model()
    # arabic_model.dataset.pca_scatter_plot(arabic_model.test_embed_vec_result)