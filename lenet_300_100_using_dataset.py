import tensorflow as tf
import numpy as np
import mnist_dataset
import time

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

EPOCHS = 2
data_dir = './datasets/MNIST_data'
batch_size = 50

ds_train = mnist_dataset.train(data_dir)
ds_seed_init = 1000
# tf.set_random_seed(tf_seed)
seed_ph = tf.placeholder(tf.int64, shape=())
ds_train = ds_train.shuffle(buffer_size=50000, seed=seed_ph) # Shuffle before repeat to guarantee data ordering (see https://www.tensorflow.org/performance/datasets_performance)
ds_train = ds_train.batch(batch_size)
# ds_train = ds_train.prefetch(batch_size)

ds_train = ds_train.repeat(1)
# iter = ds_train.make_one_shot_iterator()
# iter = ds_train.make_initializable_iterator()
# features, labels = iter.get_next()

ds_test = mnist_dataset.test(data_dir)
ds_test = ds_test.batch(batch_size) # TODO: change to be a placeholder
ds_test = ds_test.repeat(1) # no repeat since we want to control what happens after every epoch

# Create reinitializable which can be initialized from multiple different Dataset objects.
# A reinitializable iterator is defined by its structure. We could use the `output_types` and `output_shapes` properties
#  of either `ds_train` or `ds_test` here, because they are compatible.
iter = tf.data.Iterator.from_structure(ds_train.output_types, ds_train.output_shapes)
features, labels = iter.get_next()
training_init_op = iter.make_initializer(ds_train)
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

y = tf.nn.softmax(logits)

# Define loss function, optimization technique, and accuracy metric
# Add epsilon to prevent 0 log 0; See http://quabr.com/33712178/tensorflow-nan-bug
# http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
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
correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

start_time = time.time()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # all_l1_lists = []
    for epoch in range(EPOCHS):
        # sess.run(iter.initializer, feed_dict={seed_ph: tf.cast(i*ds_seed_init, tf.int64)})
        sess.run(training_init_op, feed_dict={seed_ph: np.int64(epoch * ds_seed_init)}) # This guarantee that data is shuffled randomly each epoch but also it is reproducible

        # l1, l2 = sess.run([labels, labels])
        # # l1 = sess.run(labels)
        #
        # print l1, l2
        # l1_list = []
        iter_num = 0
        while True: # while epoch isn't over
            iter_num += 1
            try:
                _, acc_val = sess.run([train_step, accuracy])
                if iter_num % 200 == 0:
                    print "accuracy in epoch {} in iteration {} is {}".format(epoch, iter_num, acc_val)
                # l1 = sess.run([labels])
                # l1_list.append(l1)
            except tf.errors.OutOfRangeError:
                print "dataset completed full epoch at iteration: {}".format(iter_num - 1) # minus 1 since last iteration didn't happen
                break
        # all_l1_lists.append(l1_list)
        print "accuracy in the end of epoch {} is {}".format(epoch, acc_val)

        # Test
        sess.run(test_init_op)
        iter_num = 0
        avg_acc = 0.0
        while True: # while epoch isn't over
            iter_num += 1
            try:
                acc_val, loss_val = sess.run([accuracy, loss])
                avg_acc += acc_val
            except tf.errors.OutOfRangeError:
                print "Finish testing the full test dataset at iteration {}".format(iter_num - 1)
                print "Test accuracy at epoch {} is: {}".format(epoch, avg_acc/(iter_num - 1)) # minus 1 since last iteration didn't happen
                break

print "Finished after {} seconds".format(time.time() - start_time)




