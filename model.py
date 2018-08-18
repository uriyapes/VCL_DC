import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import parse_image_seg2 as parse_image_seg
import nearest_neighbor
import numpy as np
import argparse
import my_utilities
import param_manager
import os
from datetime import datetime
import random

parser = argparse.ArgumentParser()
parser.add_argument('--params_dir', default='./Params', help="directory containing .json file detailing the model params")

class NeuralNet(object):
    def __init__(self, hidden_size_list, dropout_hidden_list, dataset, logger, model_params):
        # Make sure the dropout list size is equal to the hidden list size, both of them ignore the last layer
        assert(len(dropout_hidden_list) == len(hidden_size_list))
        # No dropout allowed in the last layer
        self.dropout_l = dropout_hidden_list
        self.hidden_size_list = hidden_size_list
        self.logger = logger
        self.reshuffle_flag = True
        self.learning_rate = 0.01
        self.learning_rate_update_at_epoch = 200
        self.learning_rate_updated = 1e-3
        self.dataset = dataset
        self.params = model_params

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
        self.tf_test_dataset = tf.constant(self.dataset.get_test_set())

        self.set_architecture_variables(weights_init_type="SELU")

        # Training computation.
        # Predictions for the training, validation, and test data.
        # logits = self.model(self.tf_train_minibatch)
        # self.train_prediction = tf.nn.softmax(logits)
        #
        # train_logits_full_set = self.model(self.tf_train_dataset)
        # self.train_prediction_full_set = tf.nn.softmax(train_logits_full_set)
        #
        # if self.dataset.validation_set_exist:
        #     valid_logits = self.model(self.tf_valid_dataset)
        #     self.valid_prediction = tf.nn.softmax(valid_logits)
        #
        # self.test_logits = self.model(tf_test_dataset)
        # self.test_prediction = tf.nn.softmax(self.test_logits)


        logits = self.model()
        self.prediction = tf.nn.softmax(logits)
        # TODO: change to softmax_cross_entropy_with_logits_v2 when tf is updated
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels, logits=logits))
        if self.params['vcl']:
            l2_norm = tf.get_collection('l2_norm')
            l2_norm = tf.add_n(l2_norm)
            self.loss = self.loss + self.params['gamma'] * l2_norm
        # Optimizer.
        self.optimizer = self.build_optimizer(self.learning_rate_ph, self.loss)

    def set_architecture_variables(self, weights_init_type="He"):
        if weights_init_type == "He":
            # He. initializer AKA "MSRA initialization"
            weights_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
        elif weights_init_type == "SELU":
            # SELU initializer https://github.com/deeplearning4j/deeplearning4j/issues/3739 like MSRA but with factor=1
            # weights_init = tf.random_normal_initializer(stddev=1/np.sqrt(shape[1]))
            # weights_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN', uniform=False, seed=None, dtype=tf.float32)
            pass
        else:
            assert 0
        self.weights_l = []
        self.biases_l = []

        # first layer parameters
        with tf.variable_scope("Layer_0"):
            # TODO: move the initializer back to the if statment in the start of the method.
            weights_init = tf.random_normal_initializer(stddev=1 / np.sqrt(self.dataset.get_dimensions()[0]))
            self.weights_l.append(tf.get_variable('weights', shape=[self.dataset.get_dimensions()[0], self.layer_size_l[0]],
                                                  dtype=tf.float32, initializer=weights_init))
            self.biases_l.append(tf.get_variable('bias', shape=[self.layer_size_l[0]], dtype=tf.float32,
                                             initializer=tf.zeros_initializer()))
            tf.add_to_collection('l2_loss', (tf.nn.l2_loss(self.weights_l[0])))

        weights_init = tf.random_normal_initializer(stddev=1 / np.sqrt(self.layer_size_l[0]))
        for i in range(1, len(self.layer_size_l)):
            with tf.variable_scope("Layer_{}".format(i)):
                self.weights_l.append(tf.get_variable('weights', shape=[self.layer_size_l[i-1],self.layer_size_l[i]],
                                                 dtype=tf.float32, initializer=weights_init))
                self.biases_l.append(tf.get_variable('bias', shape=[self.layer_size_l[i]], dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))
                tf.add_to_collection('l2_loss', (tf.nn.l2_loss(self.weights_l[i])))
        # I want to have the weights explictly so don't use dense
        # input = tf.layers.dense(self.tf_train_minibatch, self.layer_size, kernel_initializer=he_init,
        #                         activation=self._activation, name='layer{}'.format(i + 1))

        self.isTrain_node = tf.Variable(False, name='istrainvar', trainable=False)
        tf.add_to_collection('istrainvar', self.isTrain_node)


    def model(self):
        assert(self.layer_size_l[-1] == self.dataset.get_num_of_labels())
        assert(len(self.dropout_l) == len(self.layer_size_l)-1)
        layer_l = []
        layer_index = 0
        dropout_flag = self.dropout_l[layer_index] != 0
        if self.params['activation'] == 'RELU':
            activation = tf.nn.relu
        else:
            assert 0

        # layer_l.append(tf.nn.relu(tf.matmul(self.data_node_ph, self.weights_l[layer_index]) + self.biases_l[layer_index]))
        layer_l.append(self.linear(self.data_node_ph, self.weights_l[layer_index], self.biases_l[layer_index], 'Layer_0',
                                   dropout_flag, self.params['batch norm'], activation, self.params['vcl']))
        layer_index += 1

        for layer_index in range(1, len(self.layer_size_l) - 1):
            # layer_l.append(tf.nn.relu(tf.matmul(layer_l[op_layer_index - 1], self.weights_l[layer_index]) + self.biases_l[layer_index]))
            dropout_flag = self.dropout_l[layer_index] != 0
            layer_l.append(self.linear(layer_l[layer_index - 1], self.weights_l[layer_index], self.biases_l[layer_index]
                                       , 'Layer_{}'.format(layer_index), dropout_flag, self.params['batch norm'],
                                          activation, self.params['vcl']))

        output = tf.matmul(layer_l[-1], self.weights_l[-1]) + self.biases_l[-1]
        return output

    # def linear(input_, output_size, sample_size, eps, scope=None, bn=False, activation=None, hidden=True):
    def linear(self, input_to_layer, weights, bias, scope, dropout=None, bn=False, activation=tf.nn.relu, vcl=0, sample_size=10):

        with tf.variable_scope(scope):
            output = tf.matmul(input_to_layer, weights) + bias
            if bn:
                output = self.batchnorm(output, scope=scope)
            if vcl:
                self.add_vcl_loss(output, sample_size)
            if activation:
                output = activation(output)
            if dropout:
                output = tf.nn.dropout(output, self.keep_prob_ph)

            return output

    @staticmethod
    def batchnorm(inputT, is_training=False, scope=None):
        # Note: is_training is tf.placeholder(tf.bool) type
        is_training = tf.get_collection('istrainvar')[0]
        return tf.cond(is_training,
                       lambda: batch_norm(inputT, is_training=True,
                                          center=True, scale=True, decay=0.9, updates_collections=None, scope=scope),
                       lambda: batch_norm(inputT, is_training=False,
                                          center=True, scale=True, decay=0.9, updates_collections=None, scope=scope,
                                          reuse=True))

    @staticmethod
    def add_vcl_loss(inp, sample_size):
        me1, var1 = tf.nn.moments(inp[0:sample_size, :], 0)
        me2, var2 = tf.nn.moments(inp[sample_size:2 * sample_size, :], 0)
        shape = var1.get_shape()
        eps1 = tf.get_variable("epsilon", (shape[0]), tf.float32, tf.constant_initializer(0.1))
        var1 = tf.abs(var1)
        var2 = tf.abs(var2)
        tf.add_to_collection('l2_norm', (tf.reduce_mean(tf.square(1 - (var1) / (var2 + eps1)))))


    def train_model(self):
        self.tf_saver = tf.train.Saver()
        self.epoch = 0
        prev_epoch = -1
        step = 0
        self.initial_train_labels = np.copy(self.dataset.get_train_labels())
        dropout = self.dropout_l[-1]
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        with self.sess.as_default():
            self.logger.info('Initialized')
            self.mini_batch_step = 0
            while self.epoch < self.params['number of epochs']:

                step += 1
                batch_data, batch_labels = self.get_mini_batch()
                feed_dict = {self.data_node_ph: batch_data, self.tf_train_labels : batch_labels,
                             self.keep_prob_ph : dropout, self.learning_rate_ph: self.learning_rate}
                _, l, predictions = self.sess.run(
                    [self.optimizer, self.loss, self.prediction], feed_dict=feed_dict)
                if (prev_epoch != self.epoch):
                    prev_epoch = self.epoch
                    self.logger.info('batch_labels: {}'.format(np.argmax(batch_labels, 1)))
                    self.logger.info('predictions: {}'.format(np.argmax(predictions, 1)))
                    self.logger.info('Minibatch loss at epoch %d: %f' % (self.epoch, l))
                    self.logger.info('Minibatch accuracy: %.3f' % self.accuracy(predictions, batch_labels))
                    self.eval_model()
                    self.sess.run(self.isTrain_node.assign(True))
        self.dataset.count_classes_for_all_datasets()
        #self.dataset.count_classes(batch_labels)
        self.logger.info("Training stopped at epoch: %i" % self.epoch)


    def eval_model(self):
        assert ~np.array_equal(self.initial_train_labels, self.dataset.get_train_labels())
        test_labels = self.dataset.get_test_labels()
        with self.sess.as_default():
            self.sess.run(self.isTrain_node.assign(False))

            self.logger.info('Train accuracy: %.3f' % self.accuracy(self.prediction.eval(
                                 feed_dict={self.data_node_ph: self.dataset.get_train_set(), self.keep_prob_ph : 1}),
                                                                    self.dataset.get_train_labels()))

            self.eval_validation_accuracy()

            test_pred_eval = self.prediction.eval(feed_dict={self.data_node_ph: self.dataset.get_test_set(), self.keep_prob_ph : 1})
            network_acc = self.accuracy(test_pred_eval, test_labels)
            self.logger.info('Test accuracy: %.3f' % network_acc)
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
            self.logger.info("Update learning rate to {} in epoch {}".format(self.learning_rate_updated, self.epoch))
            self.learning_rate = self.learning_rate_updated
        if self.reshuffle_flag:
            self.dataset.re_shuffle()
        else:
            self.logger.info("reshuffle is OFF")

    @staticmethod
    def accuracy(predictions, labels):
        assert not np.array_equal(labels, None)
        return 1.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

    def eval_validation_accuracy(self):
        if self.dataset.validation_set_exist:
            valid_acc = self.prediction.eval(feed_dict={self.data_node_ph: self.dataset.get_validation_set(), self.keep_prob_ph: 1})
            self.logger.info('Validation accuracy: %.3f' % self.accuracy(valid_acc, self.dataset.get_validation_labels()))
        else:
            self.logger.info('Validation accuracy: Nan - no validation set, IGNORE')

    @staticmethod
    def run_baseline(train_set, train_labels, test_set, test_labels):
        nn = nearest_neighbor.NearestNeighbor()
        return nn.compute_one_nearest_neighbor_accuracy(train_set, train_labels, test_set, test_labels)

    @staticmethod
    def build_optimizer(lr_node, loss):
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
        self.data_node_ph = tf.placeholder(
            tf.float32, shape=(None, self.dataset.get_dimensions()[0]), name="data_node_placeholder")
        self.tf_train_labels = tf.placeholder(tf.float32, shape=(self.batch_size, self.dataset.get_num_of_labels()),
                                              name="train_labels_placeholder")
        # hyper-parameters placeholders
        self.keep_prob_ph = tf.placeholder(tf.float32)
        self.learning_rate_ph = tf.placeholder(tf.float32)


    def save_variables(self, filename=None):
        assert hasattr(self, 'sess')
        if filename:
            assert type(filename) == str
            filename = filename + ".ckpt"
        else:
            timestamp = "_" + str(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            filename = "./results/tf_variables_" + timestamp + ".ckpt"
        self.tf_saver.save(self.sess, filename)

    def load_variables(self, filename):
        assert hasattr(self, 'sess')
        self.tf_saver.restore(self.sess, filename)

    @staticmethod
    def set_seeds(tf_seed, np_seed):
        assert(type(tf_seed) == int)
        assert (type(np_seed) == int)
        tf.set_random_seed(tf_seed)
        np.random.seed(np_seed)




if __name__ == '__main__':

    logger = my_utilities.set_a_logger('log', dirpath="./Logs")
    logger.info('Start logging')
    # Load the parameters from json file
    args = parser.parse_args()
    # json_filename = 'model_params_template.json'
    # json_filename = 'unitest_params1.json'
    json_filename = 'vcl.json'
    json_path = os.path.join(args.params_dir, json_filename)
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = param_manager.ModelParams(json_path).dict
    # params = param_manager.ModelParams.create_model_params(batch_norm=1)
    # params['number of epochs'] = 0
    # params['check point flag'] = 1
    # params['check point name'] = './results/unitest2'
    params['batch norm'] = 1
    if params['random seeds'] == 1:
        params['tf seed'] = random.randint(1, 2**31)
        params['np seed'] = random.randint(1, 2**31)

    NeuralNet.set_seeds(params['tf seed'], params['np seed'])


    json_path = os.path.join(args.params_dir, 'image_segmentation_params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    dataset_dict = param_manager.DatasetParams(json_path).dict
    # dataset_dict['fold'] = 2

    # TODO: add support for different dropout rates in different layers
    keep_prob = 0.5
    #depth of 5
    hidden_size_list = [256, 256, 256, 256]
    dropout_hidden_list = [0, 0, 0, keep_prob]
    #depth of 8
    # hidden_size_list = [256]*8
    # dropout_hidden_list = [0, 0, 0, 0, 0, 0, 0, keep_prob]
    #depth of 16
    # hidden_size_list = [256] * 15
    # dropout_hidden_list = [0] *14 +[keep_prob]
    dataset = parse_image_seg.Dataset(dataset_dict)
    model = NeuralNet(hidden_size_list, dropout_hidden_list, dataset, logger, params)

    # arabic_model.dataset.pca_scatter_plot(arabic_model.dataset.test_set)
    # logger.info('1NN Baseline accuarcy: %.3f' % arabic_model.run_baseline(arabic_model.dataset.train_set,
    #                                                                 arabic_model.dataset.train_labels,
    #                                                                 arabic_model.dataset.test_set,
    #                                                                 arabic_model.dataset.test_labels))
    model.build_model()
    model.train_model()
    test_set, test_labels, network_acc, test_pred_eval = model.eval_model()
    # arabic_model.dataset.pca_scatter_plot(arabic_model.test_embed_vec_result)

    if model.params['check point flag']:
        model.save_variables(model.params['check point name'])