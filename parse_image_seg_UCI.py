from scipy.io import arff
import collections
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit

# train_set_percentage = 0.8
# validation_set_percentage = 0.0
# test_set_percentage = 0.2
# percentages = (train_set_percentage,
#         validation_set_percentage,
#         test_set_percentage)
num_labels = 7
min_label = 0
max_label = min_label + num_labels - 1


#
# assert train_set_percentage + \
#         validation_set_percentage + \
#         test_set_percentage == 1.0


class Dataset(object):
    """
        This class create a dataset from datasets files
        To create the class pass it the dataset_dict.
        dataset_dict = {name: 'image_segmentation', file_names: (<file name train>, <file name test>,
                        <file name validation>), assert_values_flag: <Boolean>, 'validation_train_ratio': <valid_size/train_size>
                         'test_alldata_ratio' : <test_size / (all_dataset_size}
        file_names can contain 1,2 or 3 file names, accordingly those files names determine the train/validation/test
        split. If only 1 file is present the split must be determined by the user.

    """

    def __init__(self, dataset_dict):
        self.name = dataset_dict['name']
        self.dict = dataset_dict
        # data, labels = self.file_to_dataset(dataset_dict['file_names'][0])
        data = np.genfromtxt(dataset_dict['file_names'][0], delimiter=',', dtype='float32')
        labels = np.genfromtxt(dataset_dict['file_names'][1], delimiter=',', dtype='float32')
        # labels = self._map_class_str_to_num(labels)

        if data.ndim == 2:
            N, T = data.shape
            self.D = 1
        else:
            assert(0, "shape isn't 2, add extra dimensions")

        if dataset_dict['assert_values_flag']:
            self._assert_dataset_val(data, labels, N, T)

        index_matrix_test = self.read_folds_indexes(dataset_dict['file_names'][2])
        fold_0_test_indexes = index_matrix_test[:, 0]
        if self.dict['validation_train_ratio'] != 0:
            index_matrix_validation = self.read_folds_indexes(dataset_dict['file_names'][3])
            fold_0_validation_indexes = index_matrix_validation[:, 0]
        else:
            fold_0_validation_indexes = np.zeros(data.shape[0])

        train_set, train_labels, validation_set, validation_labels, test_set, test_labels = self.gen_splits_from_index_vec\
            (fold_0_test_indexes, fold_0_validation_indexes, data, labels)


        # test_set, test_labels = self.file_to_dataset(dataset_dict['file_names'][1])
        # test_labels = self._map_class_str_to_num(test_labels)
        #
        # test_alldata_ratio = 1.0 * test_labels.shape[0] / (test_labels.shape[0] + train_labels.shape[0])
        # if (self.dict['test_alldata_ratio'] is None) or (np.isclose(self.dict['test_alldata_ratio'], test_alldata_ratio)):
        #     print "Don't change train/test ratio, keep it: {}".format(test_alldata_ratio)
        # elif self.dict['test_alldata_ratio'] != test_alldata_ratio:
        #     print "Change ratio between train/test to: {}".format(self.dict['test_alldata_ratio'])
        #     full_dataset = np.concatenate((train_set, test_set), axis=0)
        #     full_labels = np.concatenate((train_labels, test_labels), axis=0)
        #     train_set, train_labels, test_set, test_labels = \
        #         self.generate_balanced_splits(full_dataset, full_labels, self.dict['test_alldata_ratio'])
        # else:
        #     assert(0)
        #
        #
        # train_set, test_set = self.norm_input(train_set, test_set)
        # #update T in case we removed some features
        self.T = train_set.shape[1]

        self.validation_labels = validation_labels
        self.validation_set = validation_set

        # self.train_labels = self.encode_one_hot(train_labels)
        # self.test_labels = self.encode_one_hot(test_labels)
        self.train_labels = train_labels
        self.test_labels = test_labels


        self.train_set = train_set
        self.test_set = test_set


    def norm_input(self, train_set, test_set):
        # full_dataset = np.concatenate((train_set, valid_set, test_set), axis=0)
        mean = np.mean(train_set, axis=0)
        std = np.std(train_set, axis=0)

        train_set = self._norm_and_remove_invalid_val(train_set, mean, std)
        test_set = self._norm_and_remove_invalid_val(test_set , mean, std)

        assert (np.sum(np.isnan(train_set)) == 0)
        assert (np.sum(np.isnan(test_set)) == 0)

        return train_set, test_set

    @staticmethod
    def _norm_and_remove_invalid_val(input, mean, std):
        input = (input - mean) / std
        # check for Nan values in case some columns have constant values -> if this is the case we will get Nan values
        # for the entire column so only need to check for Nan values in the first row and delete the columns with Nans
        return np.delete(input, np.where(np.isnan(input)[0])[0], 1)

    def _map_class_str_to_num(self, labels_str):
        if not self.dict.has_key('class_dict'):
            class_dict = {}
            # return the unique indexes in the sorted array, meaning where new items appeared first (in the sorted array)
            _, indexes = np.unique(labels_str, return_index=True)
            # return the unique labels in unsorted array
            classes_str = [labels_str[index] for index in sorted(indexes)]
            # reminder: enumerate works like that - for counter, value in enumerate(some_list):
            for class_val, class_name in enumerate(classes_str):
                # create the dict
                class_dict.update({class_name: class_val})
            self.dict.update({'class_dict': class_dict})

        # For dict object .items() does the same as enumerate -  for key, val in enumm.items():
        for class_name, class_val in self.dict['class_dict'].items():
            labels_str[labels_str == class_name] = class_val
        labels = np.asarray(labels_str, dtype='float32')
        return labels

    @classmethod
    def file_to_dataset(cls, filename, label_ind=0):
        if filename.endswith(".arff"):
            data_from_file, _ = cls._read_from_arff_file(filename)
            samples, labels = cls._split_samples_from_labels(data_from_file, label_ind)
        elif filename.endswith(".data") or filename.endswith(".test") or filename.endswith(".dat"):
            samples, labels = cls._read_from_csv(filename, label_column_ind=label_ind)
        else:
            assert (0)
        return samples, labels

    @staticmethod
    def _read_from_arff_file(filename):
        raw_data, meta = arff.loadarff(filename)
        # for some reason when loading arff file one get a numpy array, each element made from a numpy void which contains
        # array. In order to transform this array to matrix we perform the following line:
        data_from_file = np.asarray(raw_data.tolist(), dtype=np.float32)

        # data_from_file = np.genfromtxt(filename, delimiter=',')
        return data_from_file, meta

    @staticmethod
    def _read_from_csv(filename, skip_header_size=5, num_of_column=20, label_column_ind=0):
        """
        read csv file and return numpy
        :param filename: the file from which the method extract the data
        :param skip_header_size: how many header lines does the file has
        :param num_of_column: number of columns the method needs to read
        :param label_column_ind: in which column the label is found
        :return:
            data_from_csv: numpy matrix containing data from all samples in file
            labels_from_csv: numpy vector containing labels for each sample
        """
        labels_from_csv = np.genfromtxt(filename, delimiter=',', skip_header=skip_header_size, dtype='str',
                                        usecols=label_column_ind)
        data_column_ind = range(num_of_column)
        data_column_ind.remove(label_column_ind)
        data_from_csv = np.genfromtxt(filename, delimiter=',', skip_header=skip_header_size, dtype='float32',
                                      usecols=data_column_ind)
        return data_from_csv, labels_from_csv

    @staticmethod
    def _split_samples_from_labels(data_from_file, label_ind):
        """Input:
            data_from_file - numpy matrix, each row is a sample with label and data
            label_ind - the colum index of the labels in data_from_file
         """
        # Assume each row represent a sample and a label
        bool_ind = [True] * data_from_file.shape[1]
        bool_ind[label_ind] = False
        labels = data_from_file[:, label_ind]
        samples = data_from_file[:, bool_ind]
        # Assumes D==1
        assert samples.ndim == 2
        return samples, labels

    @staticmethod
    def _assert_dataset_val(samples, labels, N, T):
        assert samples[0, 0] == np.float32(0.207173)
        assert samples[1, 0] == np.float32(0.854911)
        assert samples[N - 1, T - 1] == np.float32(-0.427312868586892)
        assert np.max(labels) == max_label

    @classmethod
    def generate_balanced_splits(cls, samples, labels, ratio_test_samples):
        """
        Split the datasets into folds, each fold contain train and set data with a constant <ratio> between them.
        :param samples: The dataset samples we want to split
        :param labels: The dataset labels we want to split in a balanced way
        :param ratio_test_samples: The ratio between the size of the generated dataset and the old dataset
        :return: 1 fold containing the train set (data and labels) and the test set (data and labels)
        """
        N = samples.shape[0]
        # ratio_test_samples = cls.floor_ratio(ratio_test_samples, N)
        test_size = int(np.floor(ratio_test_samples * N))
        n_splits = 1
        # random_state=None for real random or random_state={seed number}
        # test_size - If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in
        #             the test split. If int, represents the absolute number of test samples
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=None)
        sss.get_n_splits(samples, labels)

        # if n_splits isn't 1 we should expand the function to return more than 1 fold
        assert(n_splits == 1)
        for train_index, test_index in sss.split(samples, labels):
            train_set = samples[train_index]
            train_labels = labels[train_index]
            test_set = samples[test_index]
            test_labels = labels[test_index]

        assert test_size == test_set.shape[0]
        assert train_set.shape[0] + test_set.shape[0] == N

        return train_set, train_labels, test_set, test_labels


    @staticmethod
    def read_folds_indexes(filename):
        """
        the file should contain a number of columns equals to the number of folds. Each column contains 0's and 1's,
        representing if the according index belongs to the train or test set.
        :param filename: the name of the file containing the indexes of each fold
        :return:
        """
        # dtype='bool' don't work for 0's and 1's so read as uint8 and use astype
        index_matrix = np.genfromtxt(filename, delimiter=',', dtype='uint8').astype(bool)
        return index_matrix

    @staticmethod
    def gen_splits_from_index_vec(not_test_indexes, validation_indexes, samples, labels):
        """
        :param not_test_indexes: a vector which contains 0's and 1's, 0 means the according instance is used for testing and 1
        means the according instance is used for training/validation
        :param validation_indexes: a vector which contains 0's and 1's, 0 means the according instance is used for
        training/testing and 1 means the according instance is used for validation
        :param samples: the dataset samples we want to split according to the index_vec
        :param labels: the dataset labels we want to split according to the index_vec
        :return:
        """

        test_indexes_mask = np.logical_and(not_test_indexes == 0, validation_indexes == 0)
        test_set = samples[test_indexes_mask]
        test_labels = labels[test_indexes_mask]

        validation_set = samples[validation_indexes == 1]
        validation_labels = labels[validation_indexes == 1]

        train_indexes_mask = np.logical_and(not_test_indexes == 1, validation_indexes == 0)
        train_set = samples[train_indexes_mask]
        train_labels = labels[train_indexes_mask]
        return train_set, train_labels, validation_set, validation_labels, test_set, test_labels

    @staticmethod
    def floor_ratio(ratio, N):
        assert(ratio > 0)
        req_length = np.floor(N * ratio)
        round_ratio = 1.0 * req_length / N
        return round_ratio

    @staticmethod
    def encode_one_hot(class_labels):
        labels_one_hot = \
            (np.arange(min_label, max_label + 1) ==
             class_labels[:, None]).astype(np.float32)
        return labels_one_hot

    @classmethod
    def balanced_splits(cls, labels, min_label, max_label, percentages):
        balanced_splits = []
        for ii in range(len(percentages)):
            balanced_splits.append(np.array([], dtype=int))

        for label in range(min_label, max_label + 1):
            label_indices = np.nonzero(labels == label)[0]
            label_indices = cls.shuffle(label_indices)
            label_len = len(label_indices)

            cur_place = 0
            for ii in range(len(percentages)):
                split_len = int(round(percentages[ii] * label_len))
                split_indices = label_indices[cur_place:(cur_place + split_len)]
                balanced_splits[ii] = np.append(balanced_splits[ii], split_indices)
                cur_place += split_len

        assert np.sum([len(bs) for bs in balanced_splits]) == len(labels)

        for ii in range(len(balanced_splits)):
            balanced_splits[ii] = cls.shuffle(balanced_splits[ii])

        return balanced_splits

    @classmethod
    def shuffle(cls, vector):
        permutations = np.random.permutation(len(vector))
        return vector[permutations]

    def re_shuffle(self):
        self.train_set, self.train_labels = self.shuffle2(self.train_set, self.train_labels)

    @classmethod
    def shuffle2(cls, dataset, labels_one_hot):
        permutations = np.random.permutation(dataset.shape[0])
        labels_one_hot = labels_one_hot[permutations, :]

        dataset = dataset[permutations, :]
        return dataset, labels_one_hot

    def get_train_set(self):
        return self.train_set.astype(np.float32)

    def get_validation_set(self):
        return self.validation_set.astype(np.float32)

    def get_test_set(self):
        return self.test_set.astype(np.float32)

    def get_dimensions(self):
        return (self.T, self.D)

    def get_num_of_labels(self):
        return num_labels

    def get_train_labels(self):
        return self.train_labels

    def get_validation_labels(self):
        return self.validation_labels

    def get_test_labels(self):
        return self.test_labels

    def count_classes_for_all_datasets(self):
        print "train labels " + str(self.count_classes(self.get_train_labels()))
        if self.dict['validation_train_ratio'] != 0:
            print "validation labels " + str(self.count_classes(self.get_validation_labels()))
        else:
            print "validation labels - SKIP (no validation set)"
        print "test labels " + str(self.count_classes(self.get_test_labels()))

    @staticmethod
    def count_classes(dataset_labels):
        return collections.Counter(tuple(np.argmax(dataset_labels, 1) + 1))

if __name__ == '__main__':
    # FILENAME_TRAIN = r'datasets/image-segmentation-badArff/image-segmentation_train.arff'
    # FILENAME_TEST = r'datasets/image-segmentation-badArff/image-segmentation_test.arff'

    FILENAME_DATA = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/image-segmentation_py.dat'
    FILENAME_LABELS = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/labels_py.dat'
    # FILENAME_TEST = r'datasets/image-segmentation/segmentation.test'
    FILENAME_INDEXES_TEST = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/folds_py.dat'
    FILENAME_VALIDATION_INDEXES = r'/home/a/Downloads/UCI_from_Michael/data/image-segmentation/validation_folds_py.dat'
    assert_values_flag = True
    dataset_dict = {'name': 'image_segmentation', 'file_names': (FILENAME_DATA, FILENAME_LABELS, FILENAME_INDEXES_TEST, FILENAME_VALIDATION_INDEXES),
                    'assert_values_flag': assert_values_flag,
                    'validation_train_ratio': 0.0,
                    'test_alldata_ratio' : 300.0/330}
    ds = Dataset(dataset_dict)
    #  #  labels = np.concatenate([1 * np.ones(11), 2 * np.ones(11), 3 * np.ones(11)]).astype(int)
    #  labels = np.concatenate([ii*np.ones(24) for ii in range(1,16)]).astype(int)
    #  percentages = [0.7, 0.1, 0.2]
    #  indices = ds.balanced_splits(labels, 1, 15, percentages)
    #  print indices
    #  xx = [labels[indices[ii]] for ii in range(len(percentages))]
    #  for ii in xx:
    #  print ii
    #  import collections
    #  for ii in xx:
    #  print collections.Counter(ii)
