import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
import parse_image_seg2 as parse_image_seg
import numpy as np
import argparse
import my_utilities
import param_manager
import os
from datetime import datetime
import time
import mnist_input_pipe

parser = argparse.ArgumentParser()
parser.add_argument('--params_dir', default='./Params', help="directory containing .json file detailing the model params")

class NeuralNet(object):
    def __init__(self, dataset, logger, model_params):
        self.params = model_params
        logger.info('Create model with the following parameters:\n{}'.format(str(self.params)))
        # TODO: add support for different dropout rates in different layers
        # The depth include only the hidden layer, the total number of layers include another classification layer
        self.hidden_size_list = self.params['hidden size list']
        self.dropout_l = self.params['dropout keep prob list']
        # Make sure the dropout list size is equal to the hidden list size, both of them ignore the last layer
        assert(len(self.dropout_l) == len(self.hidden_size_list))
        self.logger = logger
        self.reshuffle_flag = True
        self.learning_rate = 0.005
        self.learning_rate_update_at_epoch = 10
        self.learning_rate_updated = 1e-3
        # TODO: fix
        # reload(mnist_input_pipe)
        self.dataset = dataset
        self.set_seeds(self.params['tf seed'], self.params['np seed'])


    def __enter__(self):
        self.sess = None
        pass

    def __exit__(self, type, value, traceback):
        if self.sess is not None:
            self.sess.close()
            tf.reset_default_graph()



    def build_model(self):
        T, D = self.dataset.get_dimensions()
        num_labels = self.dataset.get_num_of_labels()

        self.layer_size_l = list(self.hidden_size_list)
        # Add another layer for classification
        self.layer_size_l.append(num_labels)

        batch_size = 50
        self.set_batch_size(batch_size)
        self.set_placeholders()

        # The following constants effects the initialization of other variables so removing them FAILS the unitests.
        # TODO: remove the following constants and update ckpt file
        # self.tf_train_dataset = tf.constant(self.dataset.get_train_set())
        # if self.dataset.validation_set_exist:
        #     self.tf_valid_dataset = tf.constant(self.dataset.get_validation_set())
        # self.tf_test_dataset = tf.constant(self.dataset.get_test_set())

        self.set_architecture_variables(weights_init_type="SELU")

        logits = self.model()
        self.prediction = tf.nn.softmax(logits)
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.prediction, 1), tf.int32),
                                           self.dataset.get_labels())
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, "float"))
        # TODO: change to softmax_cross_entropy_with_logits_v2 when tf is updated
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.dataset.get_labels(), logits=logits)
        l2_loss = tf.get_collection('l2_loss')
        l2_loss = tf.add_n(l2_loss)
        self.loss = self.loss + 0.0001 * l2_loss
        if self.params['vcl']:
            l2_norm = tf.get_collection('l2_norm')
            l2_norm = tf.add_n(l2_norm)
            self.loss = self.loss + self.params['gamma'] * l2_norm
        # Optimizer.
        self.optimizer = self.build_optimizer(self.learning_rate_ph, self.loss)
        self.tf_saver = tf.train.Saver()
        self.set_prune_op()

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
        self.prune_mask_l = []
        tf.GraphKeys.PRUNING_MASKS = "pruning_masks"  # Add this to prevent pruning variables from being stored with the model
        # first layer parameters
        with tf.variable_scope("Layer_0"):
            # TODO: move the initializer back to the if statment in the start of the method.
            weights_init = tf.random_normal_initializer(stddev=1 / np.sqrt(self.dataset.get_dimensions()[0]))
            self.weights_l.append(tf.get_variable('weights', shape=[self.dataset.get_dimensions()[0], self.layer_size_l[0]],
                                                  dtype=tf.float32, initializer=weights_init))
            self.biases_l.append(tf.get_variable('bias', shape=[self.layer_size_l[0]], dtype=tf.float32,
                                             initializer=tf.zeros_initializer()))
            tf.add_to_collection('l2_loss', (tf.nn.l2_loss(self.weights_l[0])))
            self.prune_mask_l.append(tf.Variable(tf.ones_like(self.weights_l[0]), trainable=False,
                                                 collections=[tf.GraphKeys.PRUNING_MASKS]))

        weights_init = tf.random_normal_initializer(stddev=1 / np.sqrt(self.layer_size_l[0]))
        for i in range(1, len(self.layer_size_l)):
            with tf.variable_scope("Layer_{}".format(i)):
                self.weights_l.append(tf.get_variable('weights', shape=[self.layer_size_l[i-1],self.layer_size_l[i]],
                                                 dtype=tf.float32, initializer=weights_init))
                self.biases_l.append(tf.get_variable('bias', shape=[self.layer_size_l[i]], dtype=tf.float32,
                                                initializer=tf.zeros_initializer()))
                tf.add_to_collection('l2_loss', (tf.nn.l2_loss(self.weights_l[i])))
                self.prune_mask_l.append(tf.Variable(tf.ones_like(self.weights_l[i]), trainable=False,
                                                     collections=[tf.GraphKeys.PRUNING_MASKS]))
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
        elif self.params['activation'] == 'ELU':
            activation = tf.nn.elu
        elif self.params['activation'] == 'SELU':
            activation = tf.nn.selu
        else:
            assert 0

        # layer_l.append(tf.nn.relu(tf.matmul(self.data_node_ph, self.weights_l[layer_index]) + self.biases_l[layer_index]))
        layer_l.append(self.linear(self.dataset.get_data(), self.weights_l[layer_index], self.biases_l[layer_index],
                                   self.prune_mask_l[layer_index], 'Layer_0', dropout_flag, self.params['batch norm'],
                                   activation, self.params['vcl']))
        layer_index += 1

        for layer_index in range(1, len(self.layer_size_l) - 1):
            # layer_l.append(tf.nn.relu(tf.matmul(layer_l[op_layer_index - 1], self.weights_l[layer_index]) + self.biases_l[layer_index]))
            dropout_flag = self.dropout_l[layer_index] != 0
            layer_l.append(self.linear(layer_l[layer_index - 1], self.weights_l[layer_index], self.biases_l[layer_index],
                                       self.prune_mask_l[layer_index],  'Layer_{}'.format(layer_index), dropout_flag,
                                       self.params['batch norm'], activation, self.params['vcl']))

        output = tf.matmul(layer_l[-1], self.weights_l[-1]) + self.biases_l[-1]
        return output

    def linear(self, input_to_layer, weights, bias, prune_mask, scope, dropout=None, bn=False, activation=tf.nn.relu, vcl=False, sample_size=10):

        if vcl != 0 and bn != False:
            self.logger.warning('BOTH VCL AND BN ARE ACTIVE')
        with tf.variable_scope(scope):
            pruned_weights = tf.multiply(weights, prune_mask)
            output = tf.matmul(input_to_layer, pruned_weights) + bias
            if bn:
                output = self.batchnorm(output, scope=scope)
            if vcl:
                self.add_vcl_loss(output, sample_size)
            if activation:
                output = activation(output)
            if dropout:
                if self.params['activation'] != 'SELU':
                    output = tf.nn.dropout(output, self.keep_prob_ph)
                elif self.params['activation'] == 'SELU':
                    output = tf.contrib.nn.alpha_dropout(output, self.keep_prob_ph)

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


    def train_model(self, ckpt_path = None):
        self.sess = tf.Session()
        self._init_sess(ckpt_path)
        return self._train()


    def _init_sess(self, ckpt_path=None):
        if ckpt_path is None:
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.initialize_variables(tf.get_collection(tf.GraphKeys.PRUNING_MASKS)))
        else:
            self.load_variables(ckpt_path)

    def _train(self):
        train_acc_l = []
        valid_acc_l = []
        test_acc_l = []

        self.epoch = 0
        prev_epoch = 0
        # TODO: check shuffling is working
        # self.initial_train_labels = np.copy(self.dataset.get_train_labels())
        feed_dict = {self.keep_prob_ph: self.dropout_l[-1]}

        with self.sess.as_default():  # Note: This statement open a context manager of sess and not sess itself so after exiting the with need to manually close self.sess
            self.logger.info('Initialized')
            self.mini_batch_step = 0
            start_time = time.time()
            while self.epoch < self.params['number of epochs']:
                if self.learning_rate_update_at_epoch == self.epoch:
                    self.logger.info(
                        "Update learning rate to {} in epoch {}".format(self.learning_rate_updated, self.epoch))
                    self.learning_rate = self.learning_rate_updated

                self.dataset.prepare_train_ds(self.sess, self.batch_size,
                                              np.int64(self.epoch * self.params['tf seed']))
                while True:
                    try:
                        feed_dict[self.learning_rate_ph] = self.learning_rate
                        # Notice: calculating accuracy here is incorrect because the model is training, meaning that each iteration works on different model. Evaluating model and dataset must be done when nothing changes
                        _, l, predictions = self.sess.run([self.optimizer, self.loss, self.prediction],
                                                          feed_dict=feed_dict)
                    except tf.errors.OutOfRangeError:
                        self.epoch += 1
                        break

                if (prev_epoch != self.epoch):
                    # self.logger.info('batch_labels: {}'.format(np.argmax(batch_labels, 1)))
                    # self.logger.info('predictions: {}'.format(np.argmax(predictions, 1)))
                    # self.logger.info('Minibatch loss at epoch %d: %f' % (self.epoch, l))
                    # self.logger.info('Minibatch accuracy: %.3f' % self.accuracy(predictions, batch_labels))
                    self.logger.info('epoch %d' % prev_epoch)
                    train_acc, valid_acc, test_acc = self.eval_model()
                    train_acc_l.append(train_acc)
                    valid_acc_l.append(valid_acc)
                    test_acc_l.append(test_acc)
                    prev_epoch = self.epoch
                    self.sess.run(self.isTrain_node.assign(True))

        # self.dataset.count_classes_for_all_datasets()
        # self.dataset.count_classes(batch_labels)
        self.logger.info(
            "Training stopped at epoch: {} after {:.0f} seconds".format(self.epoch, (time.time() - start_time)))
        return train_acc_l, valid_acc_l, test_acc_l

    def set_prune_op(self):
        # "The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer's weights."
        threshold = 0.1
        self.update_prune_mask_op_l = []
        self.apply_prune_weights_l = []
        self.count_nnz_weights_l = []
        self.count_nnz_mask_l = []
        for i in xrange(len(self.weights_l)):
            weights = self.weights_l[i]
            threshold_per_layer = tf.sqrt(tf.nn.moments(tf.reshape(weights, [-1]), 0)[1]) * threshold # [-1] reshape tensor as vector. [1] means to take the variance
            # threshold_per_layer = tf.sqrt(2*tf.nn.l2_loss(weights)- tf.square(tf.reduce_mean(weights))) * threshold
            index_to_keep = tf.multiply(tf.to_float(tf.greater_equal(tf.abs(weights), tf.ones_like(weights) * threshold_per_layer)),
                self.prune_mask_l[i])  # Multiply by prune_mask_l[i] so pruning indexes will by apply on previous mask

            self.update_prune_mask_op_l.append(tf.assign(self.prune_mask_l[i], index_to_keep))
            self.apply_prune_weights_l.append(weights.assign(tf.multiply(weights, self.prune_mask_l[i]))) #Set weights to be the prune weights
            self.count_nnz_weights_l.append(tf.count_nonzero(weights))
            self.count_nnz_mask_l.append(tf.count_nonzero(self.prune_mask_l[i]))

        self.update_prune_masks = tf.group(self.update_prune_mask_op_l)
        self.apply_prune_weights = tf.group(self.apply_prune_weights_l)
        # self.count_nnz_weights = tf.group(self.count_nnz_weights_l)


    def eval_model(self):
        # assert not np.array_equal(self.initial_train_labels, self.dataset.get_train_labels())
        self.sess.run(self.isTrain_node.assign(False))
        with self.sess.as_default() as sess:
            # the seed here isn't like the seed is like the seed in the training part which comes right after eval_model
            self.dataset.prepare_train_ds(self.sess, self.batch_size, np.int64(self.epoch * self.params['tf seed']))
            train_pred, train_acc = self.eval_dataset(sess)
            self.logger.info('Train accuracy: %.3f' % train_acc)
            # train_pred, train_acc = self.eval_set(sess, self.dataset.get_train_set(), self.dataset.get_train_labels())
            # self.logger.info('Train accuracy: %.3f' % train_acc)

            # if self.dataset.validation_set_exist:
            #     valid_pred, valid_acc = self.eval_set(sess, self.dataset.get_validation_set(), self.dataset.get_validation_labels())
            #     self.logger.info('Validation accuracy: %.3f' % valid_acc)
            # else:
            #     valid_pred, valid_acc = None, None
            #     self.logger.info('Validation accuracy: Nan - no validation set, IGNORE')
            self.dataset.prepare_validation_ds(sess, self.batch_size)
            valid_pred, valid_acc = self.eval_dataset(sess)
            self.logger.info('Validation accuracy: %.3f' % valid_acc)

            self.dataset.prepare_test_ds(sess, self.batch_size)
            # test_pred, test_acc = self.eval_set(sess, self.dataset.get_test_set(), self.dataset.get_test_labels())
            test_pred, test_acc = self.eval_dataset(sess)
            self.logger.info('Test accuracy: %.3f' % test_acc)
        return train_acc, valid_acc, test_acc


    def eval_dataset(self, sess):
        iter_num = 0
        avg_acc = 0.0
        total_pred = np.zeros([self.batch_size, self.dataset.get_num_of_labels()])
        while True:         #While epoch isn't over
            iter_num += 1
            try:
                acc_val, predictions = sess.run([self.accuracy, self.prediction], feed_dict={self.keep_prob_ph: 1})
                avg_acc += acc_val
                total_pred += predictions
            except tf.errors.OutOfRangeError:
                avg_acc = avg_acc / (iter_num - 1)
                # self.logger.info("Finish validating the full dataset at iteration {}".format(iter_num - 1))
                # self.logger.info("Validation accuracy at epoch {} is: {}".format(self.epoch, avg_acc/(iter_num - 1))) # minus 1 since last iteration didn't happen
                break
        return total_pred, avg_acc

    def build_optimizer(self, lr_node, loss):
        # return tf.train.AdamOptimizer(lr_node).minimize(self.loss)
        # return tf.train.GradientDescentOptimizer(lr_node).minimize(self.loss)
        t_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.optimizer_class = tf.train.MomentumOptimizer(lr_node, 0.9)
        # optimizer = tf.train.GradientDescentOptimizer(lr_node)
        grads_and_vars = self.optimizer_class.compute_gradients(loss, var_list=t_vars)
        clip_constant = 1
        grads_and_vars_rescaled = [(tf.clip_by_norm(gv[0], clip_constant), gv[1]) for gv in grads_and_vars]
        train_op_net = self.optimizer_class.apply_gradients(grads_and_vars_rescaled)
        return train_op_net

    def find_best_accuracy(self, train_acc_l, valid_acc_l, test_acc_l):
        val_acc_ma_l = []
        moving_average_win_size = 10
        best_val_acc_value = 0.0
        best_val_acc_ind = 0
        for i in xrange(10, self.params['number of epochs']):
            val_acc_ma = float(np.mean(np.asarray(valid_acc_l[(i-moving_average_win_size):i])))
            assert(val_acc_ma <= 1.0)
            val_acc_ma_l.append(val_acc_ma)
            if val_acc_ma >= best_val_acc_value:
                best_val_acc_value = val_acc_ma
                best_val_acc_ind = i-1 # minus one since valid_acc_l[0:10] don't take index 10 into account and therefore the last index is 10-1

        self.logger.info('Best moving average validation accuracy appeared in epoch {} and his value is: {}'.format(best_val_acc_ind, best_val_acc_value))
        self.logger.info('Test accuracy in epoch {} is: {}'.format(best_val_acc_ind, test_acc_l[best_val_acc_ind]))

        return best_val_acc_ind, train_acc_l[best_val_acc_ind], best_val_acc_value, test_acc_l[best_val_acc_ind]


    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        # if minibatch size is bigger than train dataset size then make minibatch the same size as dataset size and
        # disable reshuffling every epoch.
        if self.batch_size > self.dataset.train_labels.shape[0]:
            self.batch_size = self.dataset.train_labels.shape[0]
            self.reshuffle_flag = False

    def set_placeholders(self):
        # Input data placeholders
        # self.data_node_ph = tf.placeholder(
        #     tf.float32, shape=(None, self.dataset.get_dimensions()[0]), name="data_node_placeholder")
        # self.label_node_ph = tf.placeholder(tf.int32, shape=(None, self.dataset.get_num_of_labels()),
        #                                       name="train_labels_placeholder")
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

    def prune(self):
        with self.sess.as_default() as sess:

            self.logger.info("NNZ weights before pruning {}".format(sess.run((self.count_nnz_weights_l))))
            self.logger.info("NNZ mask values before pruning {}".format(sess.run((self.count_nnz_mask_l))))
            logger.debug("memory usage 1: {}".format(my_utilities.memory()))
            sess.run(self.update_prune_masks)
            logger.debug("memory usage 2: {}".format(my_utilities.memory()))
            sess.run(self.apply_prune_weights)
            logger.debug("memory usage 3: {}".format(my_utilities.memory()))
            self.logger.info("NNZ weights after pruning {}".format(sess.run((self.count_nnz_weights_l))))
            sess.run(tf.variables_initializer(self.optimizer_class.variables()))
            logger.debug("memory usage 4: {}".format(my_utilities.memory()))
            train_acc_l, valid_acc_l, test_acc_l = self._train()
            logger.debug("memory usage 5: {}".format(my_utilities.memory()))
            self.logger.info("NNZ weights after retraining {}".format(sess.run((self.count_nnz_weights_l))))
            self.logger.info("NNZ mask values after retraining  {}".format(sess.run((self.count_nnz_mask_l))))
            logger.debug("memory usage 6: {}".format(my_utilities.memory()))
            return train_acc_l, valid_acc_l, test_acc_l




if __name__ == '__main__':

    logger = my_utilities.set_a_logger('log', dirpath="./Logs")
    logger.info('Start logging')
    # Load the parameters from json file
    args = parser.parse_args()
    # json_filename = 'model_params_template.json'
    # json_filename = 'unitest_params1.json'
    # json_filename = 'vcl.json'
    json_filename = 'lenet_300_100.json'
    json_path = os.path.join(args.params_dir, json_filename)
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    model_params = param_manager.ModelParams()
    model_params.update(json_path)
    params = model_params.dict
    # params = param_manager.ModelParams.create_model_params(batch_norm=1)
    # params['number of epochs'] = 1
    # params['check point flag'] = 1
    # params['check point name'] = './results/unitest2'
    # params['batch norm'] = 0
    # params['activation'] = 'ELU'


    json_path = os.path.join(args.params_dir, 'image_segmentation_params.json')
    # json_path = os.path.join(args.params_dir, 'abalone.json')
    # json_path = os.path.join(args.params_dir, 'car.json')

    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    dataset_params = param_manager.DatasetParams()
    dataset_params.update(json_path)
    dataset_dict = dataset_params.dict
    # dataset_dict['fold'] = 2
    # dataset = parse_image_seg.Dataset(dataset_dict)
    dataset = mnist_input_pipe.MnistDataset()
    dataset.prepare_datasets()

    model = NeuralNet(dataset, logger, params)
    with model:
        model.build_model()
        train_acc_l, valid_acc_l, test_acc_l = model.train_model()
        # train_acc, valid_acc, test_acc = model.eval_model()
        for i in xrange(20):
            logger.debug("memory usage before {} prune iter: {}".format(i, my_utilities.memory()))
            train_acc_l, valid_acc_l, test_acc_l = model.prune()
            logger.debug("memory usage after {} prune iter: {}".format(i, my_utilities.memory()))
            index, train_acc_at_ind, valid_acc_ma_at_ind, test_acc_at_ind = model.find_best_accuracy(train_acc_l, valid_acc_l, test_acc_l)

        if model.params['check point flag']:
            model.save_variables(model.params['check point name'])