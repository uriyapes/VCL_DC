import math
import tensorflow as tf
import parse_image_seg
import nearest_neighbor
import numpy as np

class NeuralNet(object):
    def __init__(self, hidden_size_list, dropout_hidden_list):
        # Make sure the dropout list size is equal to the hidden list size, both of them ignore the last layer
        assert(len(dropout_hidden_list) == len(hidden_size_list))
        # No dropout allowed in the last layer
        self.dropout_l = dropout_hidden_list
        self.hidden_size_list = hidden_size_list
        self.reshuffle_flag = True
        self.learning_rate = 0.01
        self.learning_rate_update_at_epoch = 200
        self.learning_rate_updated = 1e-3

    def get_dataset(self, dataset_dict):
        if dataset_dict['name'] == "image_segmentation":
            self.dataset = parse_image_seg.Dataset(dataset_dict)
        else:
            assert(0)

    def build_model(self):
        T, D = self.dataset.get_dimensions()
        num_labels = self.dataset.get_num_of_labels()

        self.layer_size_l = self.hidden_size_list
        # Add another layer for classification
        self.layer_size_l.append(num_labels)

        batch_size = 20
        self.set_batch_size(batch_size)
        self.set_placeholders()

        self.tf_train_dataset = tf.constant(self.dataset.get_train_set())
        if self.dataset.validation_set_exist:
            self.tf_valid_dataset = tf.constant(self.dataset.get_validation_set())
        tf_test_dataset = tf.constant(self.dataset.get_test_set())

        self.set_architecture_variables(weights_init_type="Xavier")

        # Training computation.
        # Predictions for the training, validation, and test data.
        logits = self.model(self.tf_train_minibatch)
        self.train_prediction = tf.nn.softmax(logits)

        train_logits_full_set = self.model(self.tf_train_dataset)
        self.train_prediction_full_set = tf.nn.softmax(train_logits_full_set)

        if self.dataset.validation_set_exist:
            valid_logits = self.model(self.tf_valid_dataset)
            self.valid_prediction = tf.nn.softmax(valid_logits)

        self.test_logits = self.model(tf_test_dataset)
        self.test_prediction = tf.nn.softmax(self.test_logits)

        # TODO: change to softmax_cross_entropy_with_logits_v2 when tf is updated
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=logits))
        # Optimizer.
        self.optimizer = self.build_optimizer(self.learning_rate_ph, self.loss)

    def set_architecture_variables(self, weights_init_type="He"):
        if weights_init_type == "He":
            # He. initializer AKA "MSRA initialization"
            weights_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
        elif weights_init_type == "Xavier":
            # Xavier initializer - not exactly because we only look at the input size(FAN_IN) and not the output size and
            # it is random distrbution and not uniform.
            # weights_init = tf.random_normal_initializer(stddev=1/np.sqrt(shape[1]))
            # weights_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
            pass
        else:
            assert 0
        self.weights_l = []
        self.biases_l = []

        # first layer parameters
        # TODO: move the initializer back to the if statment in the start of the method.
        weights_init = tf.random_normal_initializer(stddev=1 / np.sqrt(self.dataset.get_dimensions()[0]))
        self.weights_l.append(tf.get_variable('layer1_weights', shape=[self.dataset.get_dimensions()[0], self.layer_size_l[0]],
                                              dtype=tf.float32, initializer=weights_init))
        self.biases_l.append(tf.get_variable('layer1_bias', shape=[self.layer_size_l[0]], dtype=tf.float32,
                                         initializer=tf.zeros_initializer()))

        weights_init = tf.random_normal_initializer(stddev=1 / np.sqrt(self.layer_size_l[0]))
        for i in range(1, len(self.layer_size_l)):
            self.weights_l.append(tf.get_variable('layer{}_weights'.format(i+1), shape=[self.layer_size_l[i-1],self.layer_size_l[i]],
                                             dtype=tf.float32, initializer=weights_init))
            self.biases_l.append(tf.get_variable('layer{}_bias'.format(i+1), shape=[self.layer_size_l[i]], dtype=tf.float32,
                                            initializer=tf.zeros_initializer()))
        # I want to have the weights explictly so don't use dense
        # input = tf.layers.dense(self.tf_train_minibatch, self.layer_size, kernel_initializer=he_init,
        #                         activation=self._activation, name='layer{}'.format(i + 1))


    def model(self, data):
        def _add_dropout_layer(op_layer_index, layer_index):
            if self.dropout_l[layer_index] != 0:
                print "Added dropout layer after layer {} with dropout of {}".format(layer_index+1, self.dropout_l[layer_index])
                layer_l.append(tf.nn.dropout(layer_l[op_layer_index - 1], self.keep_prob_ph))
                return 1
            return 0

        assert(self.layer_size_l[-1] == self.dataset.get_num_of_labels())
        assert(len(self.dropout_l) == len(self.layer_size_l)-1)
        layer_l = []
        layer_index = 0
        op_layer_index = 0
        layer_l.append(tf.nn.relu(tf.matmul(data, self.weights_l[layer_index]) + self.biases_l[layer_index]))
        op_layer_index += 1
        op_layer_index += _add_dropout_layer(op_layer_index, layer_index)
        layer_index += 1

        for layer_index in range(1, len(self.layer_size_l) - 1):
            layer_l.append(tf.nn.relu(tf.matmul(layer_l[op_layer_index - 1], self.weights_l[layer_index]) + self.biases_l[layer_index]))
            op_layer_index += 1
            op_layer_index += _add_dropout_layer(op_layer_index, layer_index)


        output = tf.matmul(layer_l[-1], self.weights_l[-1]) + self.biases_l[-1]
        return output

    def train_model(self):
        self.epoch = 0
        prev_epoch = -1
        step = 0
        self.initial_train_labels = np.copy(self.dataset.get_train_labels())
        dropout = self.dropout_l[-1]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        with self.sess.as_default():
            print('Initialized')
            self.mini_batch_step = 0
            while self.epoch < 500:
                step += 1
                batch_data, batch_labels = self.get_mini_batch()
                feed_dict = {self.tf_train_minibatch : batch_data, self.tf_train_labels : batch_labels,
                             self.keep_prob_ph : dropout, self.learning_rate_ph: self.learning_rate}
                _, l, predictions = self.sess.run(
                    [self.optimizer, self.loss, self.train_prediction], feed_dict=feed_dict)
                if (prev_epoch != self.epoch):
                    prev_epoch = self.epoch
                    print('batch_labels: {}'.format(np.argmax(batch_labels, 1)))
                    print('predictions: {}'.format(np.argmax(predictions, 1)))
                    print('Minibatch loss at epoch %d: %f' % (self.epoch, l))
                    print('Minibatch accuracy: %.3f' % self.accuracy(predictions, batch_labels))
                    # self.eval_validation_accuracy()
                    self.eval_model()

        self.dataset.count_classes_for_all_datasets()
        #self.dataset.count_classes(batch_labels)
        print "Training stopped at epoch: %i" % self.epoch


    def eval_model(self):
        assert ~np.array_equal(self.initial_train_labels, self.dataset.get_train_labels())
        test_labels = self.dataset.get_test_labels()
        with self.sess.as_default():
            print('Train accuracy: %.3f' % self.accuracy(
                self.train_prediction_full_set.eval(feed_dict={self.keep_prob_ph : 1}), self.initial_train_labels))

            self.eval_validation_accuracy()

            test_pred_eval = self.test_prediction.eval(feed_dict={self.keep_prob_ph : 1})
            network_acc = self.accuracy(test_pred_eval, test_labels)
            print('Test accuracy of the network classifier: %.3f' % network_acc)
        return self.dataset.get_test_set(), (np.argmax(self.dataset.get_test_labels(), 1) + 1), network_acc, test_pred_eval

    def get_mini_batch(self):
        offset = (self.mini_batch_step * self.batch_size)
        # TODO: when reshuffle flag is False we will constantly ignore the end of the dataset. solution: take the end of the dataset, do reshuffle and take the begining of the suffled dataset
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
            print "Update learning rate to {} in epoch {}".format(self.learning_rate_updated, self.epoch)
            self.learning_rate = self.learning_rate_updated
        if self.reshuffle_flag:
            self.dataset.re_shuffle()
        else:
            print "reshuffle is OFF"

    @staticmethod
    def accuracy(predictions, labels):
        assert not np.array_equal(labels, None)
        return (1.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

    def eval_validation_accuracy(self):
        if self.dataset.validation_set_exist:
            print('Validation accuracy: %.3f' % self.accuracy(
                self.valid_prediction.eval(feed_dict={self.keep_prob_ph : 1}), self.dataset.get_validation_labels()))
        else:
            print 'Validation accuracy: Nan - no validation set, IGNORE'

    def run_baseline(self, train_set, train_labels, test_set, test_labels):
        nn = nearest_neighbor.NearestNeighbor()
        return nn.compute_one_nearest_neighbor_accuracy(train_set, train_labels, test_set, test_labels)

    def build_optimizer(self, lr_node, loss):
        # return tf.train.AdamOptimizer(lr_node).minimize(self.loss)
        # return tf.train.GradientDescentOptimizer(lr_node).minimize(self.loss)
        t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = tf.train.MomentumOptimizer(lr_node, 0.9)
        grads_and_vars = optimizer.compute_gradients(loss, var_list=t_vars)
        clip_constant = 1
        grads_and_vars_rescaled = [(tf.clip_by_norm(gv[0], clip_constant), gv[1]) for gv in grads_and_vars]
        train_op_net = optimizer.apply_gradients(grads_and_vars_rescaled)
        return train_op_net

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        # if minibatch size is bigger than train dataset size then make minibatch the same size as dataset size and
        # disable reshuffling every epoch.
        if self.batch_size > self.dataset.train_labels.shape[0]:
            self.batch_size = self.dataset.train_labels.shape[0]
            self.reshuffle_flag = False

    def set_placeholders(self):
        # Input data placeholders
        self.tf_train_minibatch = tf.placeholder(
            tf.float32, shape=(self.batch_size, self.dataset.get_dimensions()[0]), name="train_minibatch_placeholder")
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.dataset.get_num_of_labels()),
                                              name="train_labels_placeholder")
        # hyper-parameters placeholders
        self.keep_prob_ph = tf.placeholder(tf.float32)
        self.learning_rate_ph = tf.placeholder(tf.float32)

if __name__ == '__main__':
    FILENAME_TRAIN = r'datasets/image-segmentation/segmentation.data'
    FILENAME_TEST = r'datasets/image-segmentation/segmentation.test'
    assert_values_flag = True
    dataset_dict = {'name': 'image_segmentation', 'file_names': (FILENAME_TRAIN, FILENAME_TEST),
                    'assert_values_flag': assert_values_flag, 'validation_train_ratio': 0.1,
                    'test_alldata_ratio' : 0.01}
    assert(type(dataset_dict['test_alldata_ratio']) == float and type(dataset_dict['validation_train_ratio'] == float))
    # TODO: add support for different dropout rates in different layers
    keep_prob = 0.5
    #depth of 5
    hidden_size_list = [256, 256, 256, 256]
    dropout_hidden_list = [0, 0, 0, keep_prob]
    #depth of 8
    # hidden_size_list = [256,256,256,256,256,256,256]
    # dropout_hidden_list = [0, 0, 0, 0, 0, 0, keep_prob]
    #depth of 16
    # hidden_size_list = [256] * 15
    # dropout_hidden_list = [0] *14 +[keep_prob]

    model = NeuralNet(hidden_size_list, dropout_hidden_list)
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