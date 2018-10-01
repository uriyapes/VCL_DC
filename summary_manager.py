import tensorflow as tf
import my_utilities

def variable_summaries(var, name='summaries'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



class TensorboardLogger:
    def __init__(self, summary_dir, create_dir_flag=False, scalar_tags=None, images_tags=None):
        """
        removed :param sess: The Graph tensorflow session used in your graph.
        :param summary_dir: the directory which will save the summaries of the graph
        :param scalar_tags: The tags of summaries you will use in your training loop
        :param images_tags: The tags of image summaries you will use in your training loop
        """

        self.summary_tags = []
        self.summary_placeholders = {}
        self.summary_ops = {}

        self.set_summaries(scalar_tags, images_tags)

        self.summary_writer = None
        self.set_file_writer(summary_dir, create_dir_flag)
        self._write_enable = True
        # if "comet_api_key" in config:
        #     from comet_ml import Experiment
        #     self.experiment = Experiment(api_key=config['comet_api_key'], project_name=config['exp_name'])
        #     self.experiment.log_multiple_params(config)

    def set_file_writer(self, summary_dir, create_dir_flag=False):
        if create_dir_flag:
            my_utilities.mkdir_safe(summary_dir)
        self.summary_writer = tf.summary.FileWriter(summary_dir)

    def set_summaries(self, scalar_tags=None, images_tags=None):
        self.scalar_tags = scalar_tags
        self.images_tags = images_tags
        self.init_summary_ops()

    def init_summary_ops(self):
        with tf.variable_scope('summary_ops'):
            if self.scalar_tags is not None:
                for tag in self.scalar_tags:
                    self.summary_tags += [tag]
                    self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                    self.summary_ops[tag] = tf.summary.scalar(tag, self.summary_placeholders[tag])
            if self.images_tags is not None:
                for tag, shape in self.images_tags:
                    self.summary_tags += [tag]
                    self.summary_placeholders[tag] = tf.placeholder('float32', shape, name=tag)
                    self.summary_ops[tag] = tf.summary.image(tag, self.summary_placeholders[tag], max_outputs=10)

    def summarize(self, sess, step, summaries_dict=None, summaries_merged=None):
        """
        Add the summaries to tensorboard
        :param sess: The Graph tensorflow session used in your graph.
        :param step: the number of iteration in your training
        :param summaries_dict: the dictionary which contains your summaries . Those summaries should be scalar values or images
        :param summaries_merged: Merged summaries which they come from your graph, in other words send the output of sess.run(merge_op)
        :return:
        """
        if summaries_dict is not None:
            # Note: the calculation here don't effect your graph/dataset because all we do here is to feed the pre-defined
            # scalar placeholders with the data.
            summary_list = sess.run([self.summary_ops[tag] for tag in summaries_dict.keys()],
                                         {self.summary_placeholders[tag]: value for tag, value in
                                          summaries_dict.items()})
            for summary in summary_list:
                self.summary_write(summary, step)
        if summaries_merged is not None:
            self.summary_write(summaries_merged, step)

            if hasattr(self, 'experiment') and self.experiment is not None:
                self.experiment.log_multiple_metrics(summaries_dict, step=step)

    def summary_write(self, summary, step):
        if self._write_enable:
            self.summary_writer.add_summary(summary, step)

    def set_write_enable(self, true_or_false):
        self._write_enable = true_or_false

    # TODO: was the idea to flush (write all pending events before closing) when deleting the object? and if so does finalize actually does it? shouldn't I use __del__?
    def finalize(self):
        self.summary_writer.flush()



# if __name__ == '__main__':
#     # create tensorboard logger
#     tb_logger = TensorboardLogger(sess, summary_dir=config.summary_dir,
#                                scalar_tags=['train/loss_per_epoch', 'train/acc_per_epoch',
#                                             'test/loss_per_epoch', 'test/acc_per_epoch'])