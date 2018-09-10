import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data_dir = './datasets/MNIST_data'
mnist = input_data.read_data_sets(data_dir, one_hot=False)
train_set, train_labels = mnist.train.images, mnist.train.labels
validation_set, validation_labels = mnist.validation.images, mnist.validation.labels
test_set, test_labels = mnist.test.images, mnist.test.labels
train_labels = np.int32(train_labels)
validation_labels = np.int32(validation_labels)
test_labels = np.int32(test_labels)

min_label = np.min(train_labels)
max_label = np.max(train_labels)
num_labels = max_label - min_label + 1
# print train_labels.shape

# ds_train = tf.data.Dataset.from_tensor_slices((train_set, train_labels)) # BAD since it consumes too much memory (see offical website)
batch_size_ph = tf.placeholder(tf.int64)
seed_ph = tf.placeholder(tf.int64, shape=())

_input_data, input_labels = tf.placeholder(tf.float32, shape=[None, 784]), tf.placeholder(tf.int32, shape=[None])

ds_train = tf.data.Dataset.from_tensor_slices((_input_data, input_labels))
ds_train = ds_train.shuffle(buffer_size=50000, seed=seed_ph) # Shuffle before repeat to guarantee data ordering (see https://www.tensorflow.org/performance/datasets_performance)
ds_train = ds_train.batch(batch_size_ph)
# ds_train = ds_train.prefetch(batch_size)
ds_train = ds_train.repeat(1)
# iter = ds_train.make_one_shot_iterator()
# iter = ds_train.make_initializable_iterator()
# features, labels = iter.get_next()

ds_valid = tf.data.Dataset.from_tensor_slices((_input_data, input_labels))
ds_valid = ds_valid.batch(batch_size_ph)
ds_valid = ds_valid.repeat(1) # no repeat since we want to control what happens after every epoch

# ds_test = mnist_dataset.test(data_dir)
# ds_test = tf.data.Dataset.from_tensor_slices((test_set, test_labels))
ds_test = tf.data.Dataset.from_tensor_slices((_input_data, input_labels))
ds_test = ds_test.batch(batch_size_ph)
ds_test = ds_test.repeat(1) # no repeat since we want to control what happens after every epoch

# Create reinitializable which can be initialized from multiple different Dataset objects.
# A reinitializable iterator is defined by its structure. We could use the `output_types` and `output_shapes` properties
#  of either `ds_train` or `ds_test` here, because they are compatible.
iter = tf.data.Iterator.from_structure(ds_train.output_types, ds_train.output_shapes)
print ds_train.output_shapes
features, labels = iter.get_next()
training_init_op = iter.make_initializer(ds_train)
valid_init_op = iter.make_initializer(ds_valid)
test_init_op = iter.make_initializer(ds_test)


def get_data():
    return features


def get_labels():
    return labels


def prepare_train_ds(sess, batch_size, ds_seed):
    # Use a predetermined seeds so that data is shuffled randomly each epoch but also it is reproducible
    sess.run(training_init_op, feed_dict={seed_ph: np.int64(ds_seed), batch_size_ph: batch_size, _input_data: train_set,
                                          input_labels: train_labels})


def prepare_validation_ds(sess, batch_size):
    sess.run(valid_init_op, feed_dict={batch_size_ph: batch_size, _input_data: validation_set, input_labels: validation_labels})


def prepare_test_ds(sess, batch_size):
    sess.run(test_init_op, feed_dict={batch_size_ph: batch_size, _input_data: test_set, input_labels: test_labels})


def get_dimensions():
    instance_attributes = ds_train.output_shapes[0][1].value # In Mnist case this equals 784
    if type(instance_attributes) == int:
        num_of_channels = 1
    else:
        assert 0
    return instance_attributes, num_of_channels


def get_num_of_labels():
    return num_labels
