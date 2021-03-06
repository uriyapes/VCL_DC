from model import NeuralNet
from model import prune_and_retrain
import param_manager
import parse_image_seg2 as parse_image_seg
import my_utilities
import csv
import numpy as np
import os
from datetime import datetime
import argparse
import logging #TODO: remove from here
from mnist_input_pipe import MnistDataset
import summary_manager

parser = argparse.ArgumentParser()
parser.add_argument('--params_dir', default='./Params', help="directory containing .json file detailing the model and datasets params")


def get_model(dataset, logger, model_params):
    model = NeuralNet(dataset, logger, model_params.dict)
    return model


def init_model(model):
    model.build_model()


def run_model(model):
    tb_logger = summary_manager.TensorboardLogger(summary_dir='./results/summaries/train', create_dir_flag=True,
                                                  scalar_tags=['train/acc_per_epoch', 'validation/acc_per_epoch',
                                                               'test/acc_per_epoch'])
    train_acc_l, valid_acc_l, test_acc_l = model.train_model(tb_logger)
    index, train_acc_at_ind, valid_acc_ma_at_ind, test_acc_at_ind = model.find_best_accuracy(train_acc_l, valid_acc_l, test_acc_l)
    return index, train_acc_at_ind, valid_acc_ma_at_ind, test_acc_at_ind


def write_results_to_csv_as_row(list, file_name = 'results.csv'):
    with open(file_name, 'ab') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=' ')  #DELETE: quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(list)


log_num = 0


def run_model_multiple_times(dataset_dict, dataset_folds_list, dataset, num_of_model_runs, model_params,
                             results_dir_path ='./results', prune_flag=False):
    global log_num

    best_index_l =[]
    final_train_acc_l = []
    final_valid_acc_l = []
    final_test_acc_l = []
    # TODO: move get model outside the loop, create a logger function that can change logging file destination which is
    # all that need to change between runs and then no need for log_num
    # https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a
    for j in xrange(len(dataset_folds_list)):
        dataset_dict['fold'] = dataset_folds_list[j]
        for i in xrange(num_of_model_runs):
            model_params.randomize_seeds()
            param_file = os.path.join(results_dir_path, "param_run_{}_fold_{}.json".format(i, j))
            model_params.save(param_file)
            logger = my_utilities.set_a_logger(str(log_num), dirpath=results_dir_path, filename='run_{}_fold_{}.log'.format(i, j),
                                               console_level=logging.DEBUG, file_level=logging.DEBUG)
            log_num += 1
            logger.info('Start logging')
            logger.info('########## Number of model run: {0} ##########'.format(i))

            # TODO: change Dataset so we could do unshuffle and then move it outside the for loop
            # dataset = parse_image_seg.Dataset(dataset_dict)
            dataset.prepare_datasets()
            model = get_model(dataset, logger, model_params)
            with model:
                init_model(model)
                index, train_acc_at_ind, valid_acc_ma_at_ind, test_acc_at_ind = run_model(model)
                best_index_l.append(index)
                final_train_acc_l.append(train_acc_at_ind)
                final_valid_acc_l.append(valid_acc_ma_at_ind)
                final_test_acc_l.append(test_acc_at_ind)
                if prune_flag:
                    prune_itr = 12
                    prune_th_l = [0.85] * (prune_itr/2) + [0.9]*(prune_itr - prune_itr/2)
                    prune_th_l = prune_th_l[0:8]
                    retrain_epoches = 25
                    pruned_test_acc_l_at_ind, nnz_weights_l = prune_and_retrain(model, prune_th_l, retrain_epoches,
                                                                    new_tb_logger=False, ckpt_pruned_models_flag=False)
                else:
                    pruned_test_acc_l_at_ind, nnz_weights_l = None, None

    return best_index_l, final_train_acc_l, final_valid_acc_l, final_test_acc_l, pruned_test_acc_l_at_ind, nnz_weights_l


def choose_activation_regularizer(activation_regularizer):
    if activation_regularizer == 'no regularizer':
        batch_norm = False
        use_vcl = False
    elif activation_regularizer == 'batch norm':
        batch_norm = True
        use_vcl = False
    elif activation_regularizer == 'vcl':
        batch_norm = False
        use_vcl = True
    else:
        assert 0
    return batch_norm, use_vcl


def run_model_with_diff_hyperparams(dataset_dict, dataset_folds_list, model_runs_per_config, depth_list, activation_list,
                                    activation_regu_list, l2_loss_coeff_list, vcl_sample_size_list, prune_flag):
    timestamp = str(datetime.now().strftime('%Y_%m_%d__%H-%M-%S'))
    path_results_per_dataset = r"./results/" + dataset_dict['dataset_name']
    if not os.path.isdir(path_results_per_dataset):
        os.mkdir(path_results_per_dataset)
    path_results_dir = os.path.join(path_results_per_dataset, timestamp)
    os.mkdir(path_results_dir)
    summary_result_filename = os.path.join(path_results_dir, "avg_results_over_{}_runs_over_{}_folds.csv".format(model_runs_per_config, len(dataset_folds_list)))
    write_results_to_csv_as_row(['activation', 'regularizer', 'depth', 'vcl_size', 'l2_coeff', 'train accuracy', 'validation accuracy',
                                 'test accuracy', 'test accuracy after pruning', 'NNZ weights'], summary_result_filename)

    dataset = MnistDataset()
    for vcl_sample_size in vcl_sample_size_list:
        for l2_coeff in l2_loss_coeff_list:
            for a in xrange(len(activation_list)):
                activation = activation_list[a]
                dropout_keep_prob = 0.5 if activation != 'SELU' else 0.95
                for r in xrange(len(activation_regu_list)):
                    batch_norm, use_vcl = choose_activation_regularizer(activation_regu_list[r])
                    for d in xrange(len(depth_list)):
                        hidden_size_list = depth_list[d] * [256]
                        config_name = "activation_{}_regularizer_{}_depth_{}_vclSize_{}_l2Coeff_{}".format(
                            activation, activation_regu_list[r], depth_list[d], vcl_sample_size, l2_coeff)
                        # This path is used to save the log information, parameter file and graph variables
                        path_run_info = os.path.join(path_results_dir, config_name)
                        os.mkdir(path_run_info)

                        # if activation != 'SELU':
                        #     dropout_hidden_list = [0] * len(hidden_size_list)
                        #     dropout_hidden_list[-1] = dropout_keep_prob
                        # else:
                        #     dropout_hidden_list = [dropout_keep_prob] * depth_list[d]

                        dropout_hidden_list = [dropout_keep_prob] * depth_list[d]

                        params = param_manager.ModelParams()
                        params_file_name = 'lenet_300_100.json'
                        params_file_path = os.path.join('./Params', params_file_name)
                        params.update(params_file_path)
                        params.dict['batch norm'] = batch_norm
                        params.dict['activation'] = activation
                        params.dict['vcl'] = use_vcl
                        params.dict['dropout keep prob list'] = dropout_hidden_list
                        params.dict['number of epochs'] = 80
                        params.dict['l2 coeff'] = l2_coeff
                        params.dict['vcl sample size'] = vcl_sample_size
                        best_index_l, final_train_acc_l, final_valid_acc_l, final_test_acc_l, pruned_test_acc_l_at_ind, nnz_weights_l\
                            = run_model_multiple_times(dataset_dict, dataset_folds_list, dataset, model_runs_per_config,
                                                       params, path_run_info, prune_flag)

                        file_name = os.path.join(path_run_info, "results_summary.csv")
                        write_results_to_csv_as_row(['train accuracy'] + final_train_acc_l, file_name)
                        write_results_to_csv_as_row(['validation accuracy'] + final_valid_acc_l, file_name)
                        write_results_to_csv_as_row(['test accuracy'] + final_test_acc_l, file_name)

                        result_summary_config = [activation, activation_regu_list[r], depth_list[d], vcl_sample_size, l2_coeff]
                        result_summary = [np.mean(final_train_acc_l), np.mean(final_valid_acc_l), np.mean(final_test_acc_l)]
                        result_summary = result_summary_config + ['{:.3f}'.format(x) for x in result_summary] + \
                                         [pruned_test_acc_l_at_ind, nnz_weights_l]

                        write_results_to_csv_as_row(result_summary, summary_result_filename)


if __name__ == '__main__':
    args = parser.parse_args()
    json_path = os.path.join(args.params_dir, 'contrac.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    dataset_params = param_manager.DatasetParams()
    dataset_params.update(json_path)
    dataset_dict = dataset_params.dict
    dataset_dict['dataset_name'] = 'MNIST'

    model_runs_per_config = 1
    dataset_folds_list = [0]
    depth_list = [2]
    # activation_list = ['RELU', 'ELU', 'SELU']
    activation_list = ['RELU', 'SELU']
    activation_regu_list = ['batch norm']
    # activation_regu_list = ['no regularizer']
    l2_loss_coeff_list = [0, 0.0001]
    vcl_sample_size_list = [10]
    prune_flag = True


    run_model_with_diff_hyperparams(dataset_dict, dataset_folds_list, model_runs_per_config, depth_list, activation_list,
                                    activation_regu_list, l2_loss_coeff_list, vcl_sample_size_list, prune_flag)
