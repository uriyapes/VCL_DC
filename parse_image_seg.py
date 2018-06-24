from scipy.io import arff

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import ShuffleSplit



#data, meta = arff.loadarff(FILENAME_TRAIN)


D = 1
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
                        <file name validation>), assert_values_flag: <Boolean>, 'validation_percentage': <percent from train>}
        file_names can contain 1,2 or 3 file names, accordingly those files names determine the train/validation/test
        split. If only 1 file is present the split must be determined by the user.

    """
    def __init__(self, dataset_dict):
        self.name = dataset_dict['name']
        self.dict = dataset_dict
        samples, labels = self.file_to_dataset(dataset_dict['file_names'][0])

        if samples.ndim == 2:
            N, T = samples.shape
        else:
            assert(0, "shape isn't 2, add extra dimensions")

        assert (N, T, D, 1) == samples[:,:,None,None].shape
        if dataset_dict['assert_values_flag']:
            self._assert_dataset_val(samples, labels, N, T)

        if dataset_dict['validation_percentage'] == 0:
            validation_set_len = 0
        else:
            validation_set_len = np.floor(N * dataset_dict['validation_percentage']/100.0)
            validation_ratio_floor = 1.0 * validation_set_len/N

        train_set, train_labels, validation_set, validation_labels = \
                self.generate_balanced_splits(samples, labels, validation_ratio_floor)

        assert validation_set_len == validation_set.shape[0]
        assert train_set.shape[0] + validation_set.shape[0] == N

        self.train_set = train_set
        self.validation_set = validation_set
        self.train_labels = self.encode_one_hot(train_labels)
        self.validation_labels = self.encode_one_hot(validation_labels)

        test_set, test_labels = self.file_to_dataset(dataset_dict['file_names'][1])
        self.test_set = test_set
        self.test_labels = self.encode_one_hot(test_labels)
        self.T = T
        self.D = D

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

    @classmethod
    def file_to_dataset(cls, filename):
        data_from_file, _ = cls._read_from_file(filename)
        samples, labels = cls._parse_file(data_from_file)
        return samples, labels

    @staticmethod
    def _read_from_file(filename):
        raw_data, meta = arff.loadarff(filename)
        #for some reason when loading arff file one get a numpy array, each element made from a numpy void which contains
        # array. In order to transform this array to matrix we perform the following line:
        data_from_file = np.asarray(raw_data.tolist(), dtype=np.float32)

        #data_from_file = np.genfromtxt(filename, delimiter=',')
        return data_from_file, meta

    @staticmethod
    def _parse_file(data_from_file):
        labels = data_from_file[:,-1]
        samples = data_from_file[:,:-1]
        # Assumes D==2
        assert D == 1
        #samples_d1 = samples_from_file[:,0::D]
        #samples_d2 = samples_from_file[:,1::D]
        #samples = np.stack((samples_d1, samples_d2), axis=-1)
        #samples = samples[:,:,:,None]

        return samples, labels

    @staticmethod
    def _assert_dataset_val(samples, labels, N, T):
        assert samples[0,0] == np.float32(0.207173)
        assert samples[1,0] == np.float32(0.854911)
        assert samples[N-1,T-1] == np.float32(2.03876)
        assert labels[N-1] == 6

    @staticmethod
    def generate_balanced_splits(samples, labels, test_set_ratio):
        #random_state=None for real random or random_state={seed number}
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_set_ratio, random_state=None)
        sss.get_n_splits(samples, labels)
        for train_index, test_index in sss.split(samples, labels):
            train_set = samples[train_index]
            train_labels = labels[train_index]
            test_set = samples[test_index]
            test_labels = labels[test_index]

        return train_set, train_labels, test_set, test_labels

    @staticmethod
    def encode_one_hot(class_labels):
        labels_one_hot = \
                (np.arange(min_label, max_label + 1) ==
                    class_labels[:,None]).astype(np.float32)
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
                split_len = int(round(percentages[ii]*label_len))
                split_indices = label_indices[cur_place:(cur_place+split_len)]
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

if __name__=='__main__':
    FILENAME_TRAIN = r'datasets/image-segmentation/image-segmentation_train.arff'
    FILENAME_TEST = r'datasets/image-segmentation/image-segmentation_test.arff'
    assert_values_flag = True
    dataset_dict = {'name': 'image_segmentation', 'file_names': (FILENAME_TRAIN, FILENAME_TEST), 'assert_values_flag': True,
                    'validation_percentage': 15.0}
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
