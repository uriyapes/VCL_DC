import tensorflow as tf
import numpy as np
import mnist_dataset # This doesn't contains validation dataset so use tutorials.mnist instead
from tensorflow.examples.tutorials.mnist import input_data
import time

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)




data_dir = './datasets/MNIST_data'
mnist = input_data.read_data_sets(data_dir, one_hot=False)
train_set, train_labels = mnist.train.images, mnist.train.labels
validation_set, validation_labels = mnist.validation.images, mnist.validation.labels
test_set, test_labels = mnist.test.images, mnist.test.labels
train_labels = np.int32(train_labels)
validation_labels = np.int32(validation_labels)
test_labels = np.int32(test_labels)
print train_labels.shape

EPOCHS = 91
batch_size = 50
ds_seed_init = 1000
tf.set_random_seed(10)
# ds_train = mnist_dataset.train(data_dir)
# ds_train = tf.data.Dataset.from_tensor_slices((train_set, train_labels)) # BAD since it consumes too much memory (see offical website)
batch_size_ph = tf.placeholder(tf.int64)
seed_ph = tf.placeholder(tf.int64, shape=())

x, y = tf.placeholder(tf.float32, shape=[None, 784]), tf.placeholder(tf.int32, shape=[None])

ds_train = tf.data.Dataset.from_tensor_slices((x, y))
ds_train = ds_train.shuffle(buffer_size=50000, seed=seed_ph) # Shuffle before repeat to guarantee data ordering (see https://www.tensorflow.org/performance/datasets_performance)
ds_train = ds_train.batch(batch_size_ph)
# ds_train = ds_train.prefetch(batch_size)
ds_train = ds_train.repeat(1)
# iter = ds_train.make_one_shot_iterator()
# iter = ds_train.make_initializable_iterator()
# features, labels = iter.get_next()

ds_valid = tf.data.Dataset.from_tensor_slices((x, y))
ds_valid = ds_valid.batch(batch_size_ph)
ds_valid = ds_valid.repeat(1) # no repeat since we want to control what happens after every epoch

# ds_test = mnist_dataset.test(data_dir)
# ds_test = tf.data.Dataset.from_tensor_slices((test_set, test_labels))
ds_test = tf.data.Dataset.from_tensor_slices((x, y))
ds_test = ds_test.batch(batch_size_ph)
ds_test = ds_test.repeat(1) # no repeat since we want to control what happens after every epoch

# Create reinitializable which can be initialized from multiple different Dataset objects.
# A reinitializable iterator is defined by its structure. We could use the `output_types` and `output_shapes` properties
#  of either `ds_train` or `ds_test` here, because they are compatible.
iter = tf.data.Iterator.from_structure(ds_train.output_types, ds_train.output_shapes)
features, labels = iter.get_next()
training_init_op = iter.make_initializer(ds_train)
valid_init_op = iter.make_initializer(ds_valid)
test_init_op = iter.make_initializer(ds_test)


# make a simple model
# x = tf.placeholder("float", shape=[None, 28 * 28])
# y_ = tf.placeholder("float", shape=[None, 10])

# fc1
W_fc1 = weight_variable([28 * 28, 300])
b_fc1 = bias_variable([300])
h_fc1 = tf.nn.relu(tf.matmul(features, W_fc1) + b_fc1)

# fc2
W_fc2 = weight_variable([300, 100])
b_fc2 = bias_variable([100])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

# fc3
W_fc3 = weight_variable([100, 10])
b_fc3 = bias_variable([10])
logits = tf.matmul(h_fc2, W_fc3) + b_fc3

prediction = tf.nn.softmax(logits)

# Define loss function, optimization technique, and accuracy metric
# Add epsilon to prevent 0 log 0; See http://quabr.com/33712178/tensorflow-nan-bug
# http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="cross_entropy_loss")
# "Overall, L2 regularization gives the best pruning results."
# l2_loss = tf.nn.l2_loss(tf.concat(0, [tf.reshape(W_fc1, [-1]), tf.reshape(W_fc2, [-1]), tf.reshape(W_fc3, [-1])]))
tf.add_to_collection('l2_loss', (tf.nn.l2_loss(W_fc1)))
tf.add_to_collection('l2_loss', (tf.nn.l2_loss(W_fc2)))
tf.add_to_collection('l2_loss', (tf.nn.l2_loss(W_fc3)))
l2_loss = tf.get_collection('l2_loss')
l2_loss = tf.add_n(l2_loss)

l2_weight_decay = 0.0001  # 0.001 Suggested by Hinton et al. in 2012 ImageNet paper, but smaller works here
loss = cross_entropy + l2_loss * l2_weight_decay

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(y, 1), labels)
correct_prediction = tf.equal(tf.cast(tf.argmax(prediction, 1), tf.int32), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    all_l1_lists = []
    for epoch in range(EPOCHS):
        # sess.run(iter.initializer, feed_dict={seed_ph: tf.cast(i*ds_seed_init, tf.int64)})
        sess.run(training_init_op, feed_dict={seed_ph: np.int64(epoch * ds_seed_init), batch_size_ph: batch_size, # Use a predetermined seeds so that data is shuffled randomly each epoch but also it is reproducible
                                              x: train_set, y: train_labels})

        iter_num = 0
        which_labels = np.array([])
        avg_acc = 0.0
        while True: # while epoch isn't over
            iter_num += 1
            try:
                _, acc_val, l1 = sess.run([train_step, accuracy, labels])
                avg_acc += acc_val
                if iter_num % 200 == 0:
                    print "Train batch accuracy in epoch {} in iteration {} is {}".format(epoch, iter_num, acc_val)
                which_labels = np.concatenate((which_labels, l1))
            except tf.errors.OutOfRangeError:
                print "dataset completed full epoch at iteration: {}".format(iter_num - 1) # minus 1 since last iteration didn't happen
                print "Train accuracy in the end of epoch {} is {}".format(epoch, avg_acc / (iter_num - 1))
                break
        all_l1_lists.append(which_labels)


        # Validation
        sess.run(valid_init_op, feed_dict={batch_size_ph: batch_size, x: validation_set, y: validation_labels})
        iter_num = 0
        avg_acc = 0.0
        while True:         #While epoch isn't over
            iter_num += 1
            try:
                acc_val, loss_val = sess.run([accuracy, loss])
                avg_acc += acc_val
            except tf.errors.OutOfRangeError:
                print "Finish validating the full validation dataset at iteration {}".format(iter_num - 1)
                print "Validation accuracy at epoch {} is: {}".format(epoch, avg_acc/(iter_num - 1)) # minus 1 since last iteration didn't happen
                break


        # Test
        sess.run(test_init_op,feed_dict={batch_size_ph: batch_size, x: test_set, y: test_labels})
        iter_num = 0
        avg_acc = 0.0
        while True:         #While epoch isn't over
            iter_num += 1
            try:
                acc_val, loss_val = sess.run([accuracy, loss])
                avg_acc += acc_val
            except tf.errors.OutOfRangeError:
                print "Finish testing the full test dataset at iteration {}".format(iter_num - 1)
                print "Test accuracy at epoch {} is: {}".format(epoch, avg_acc/(iter_num - 1)) # minus 1 since last iteration didn't happen
                break

assert not np.array_equal(all_l1_lists[0], all_l1_lists[1])
for i in xrange(0, len(all_l1_lists)):
    print all_l1_lists[i][0:30]
print "Finished after {} seconds".format(time.time() - start_time)




