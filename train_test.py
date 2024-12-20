# Set environment variables ========================================

# Huggingface-related environment variables
import os, sys
TESTING = os.environ.get("PAB_TESTING") == "True"
if TESTING:
    sys.path.insert(0, "/home/francois/MASTER/libs")
    sys.path.insert(0, "/home/francois/MASTER/PedestrianActionBenchmark/libs/simple_hrnet")
HUGGINGFACE_CACHE_DIRECTORY = os.environ.get("HUGGINGFACE_CACHE_DIRECTORY") or "/home/francois/huggingface"
os.environ['TRANSFORMERS_CACHE'] = HUGGINGFACE_CACHE_DIRECTORY
os.environ['HF_DATASETS_CACHE'] = HUGGINGFACE_CACHE_DIRECTORY
os.environ['TRANSFORMERS_OFFLINE'] = "0"
os.environ['HF_DATASETS_OFFLINE'] = "0"

# Imports
import copy
import getopt
import random
import sys
import yaml

import numpy as np
from transformers import set_seed as huggingface_set_seed
import torch

from action_predict import ActionPredict
from datasets_data.jaad_data import JAAD
from datasets_data.pie_data import PIE
from models.models import *
from models.multi_branch_models.combined_models import *
from utils.global_variables import get_time_writing_to_disk
from utils.hyperparameters import HyperparamsOrchestrator

SEED = 42

def write_to_yaml(yaml_path=None, data=None):
    """
    Write model to yaml results file
    
    Args:
        model_path (None, optional): Description
        data (None, optional): results from the run
    
    Deleted Parameters:
        exp_type (str, optional): experiment type
        overwrite (bool, optional): whether to overwrite the results if the model exists
    """
    with open(yaml_path, 'w') as yamlfile:
        yaml.dump(data, yamlfile)

def action_prediction(model_name):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    for cls in Static.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))

def run(config_file=None, test_only=False,
        free_memory=True, 
        compute_time_writing_to_disk=False,
        tune_hyperparameters=False):
    """
    Run train and test on the dataset with parameters specified in configuration file.
    
    Args:
        config_file: path to configuration file in yaml format
        dataset: dataset to train and test the model on (pie, jaad_beh or jaad_all)
    """
    if compute_time_writing_to_disk:
        global TIME_WRITING_TO_DISK
    print(config_file)
    # Read default Config file
    configs_default ='config_files/configs_default.yaml'
    with open(configs_default, 'r') as f:
        configs = yaml.safe_load(f)

    with open(config_file, 'r') as f:
        model_configs = yaml.safe_load(f)

    # Update configs based on the model configs
    for k in ['model_opts', 'net_opts']:
        if k in model_configs:
            configs[k].update(model_configs[k])

    # Calculate min track size
    tte = configs['model_opts']['time_to_event'] if isinstance(configs['model_opts']['time_to_event'], int) else \
        configs['model_opts']['time_to_event'][1]
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + tte

    # update model and training options from the config file
    for dataset_idx, dataset in enumerate(model_configs['exp_opts']['datasets']):
        configs['data_opts']['sample_type'] = 'beh' if 'beh' in dataset else 'all'
        configs['model_opts']['overlap'] = 0.6 if 'pie' in dataset else 0.8
        configs['model_opts']['dataset'] = dataset.split('_')[0]
        configs['model_opts']['dataset_full'] = dataset
        configs['train_opts']['batch_size'] = model_configs['exp_opts']['batch_size'][dataset_idx]
        configs['train_opts']['lr'] = model_configs['exp_opts']['lr'][dataset_idx]
        configs['train_opts']['epochs'] = model_configs['exp_opts']['epochs'][dataset_idx]

        model_name = configs['model_opts']['model']
        # Remove speed in case the dataset is jaad
        if 'RNN' in model_name and 'jaad' in dataset:
            configs['model_opts']['obs_input_type'] = configs['model_opts']['obs_input_type']

        for k, v in configs.items():
            print(k,v)

        # set batch size
        if model_name in ['ConvLSTM']:
            configs['train_opts']['batch_size'] = 2
        if model_name in ['C3D', 'I3D']:
            configs['train_opts']['batch_size'] = 16
        if model_name in ['PCPA']:
            configs['train_opts']['batch_size'] = 8
        if 'MultiRNN' in model_name:
            configs['train_opts']['batch_size'] = 8
        if model_name in ['TwoStream']:
            configs['train_opts']['batch_size'] = 16

        beh_seq_train, beh_seq_val, beh_seq_test, beh_seq_test_cross_dataset = \
            get_trajectory_sequences(configs, free_memory)
        
        model = "trajectorytransformerb"
        submodel = "VanillaTransformerForForecastClassification"
        hyperparams_orchestrator = HyperparamsOrchestrator(tune_hyperparameters, model, submodel)
        for i in range(hyperparams_orchestrator.nb_cases):
            hyperparams = hyperparams_orchestrator.get_next_case()
            if hyperparams:
                print(f"Training model with hyperparams set {i}: {str(hyperparams[model][submodel])}")
            train_test_model(configs, beh_seq_train, beh_seq_val, beh_seq_test,
                             beh_seq_test_cross_dataset, hyperparams,
                             test_only=test_only)

        
def train_test_model(configs, beh_seq_train, beh_seq_val, beh_seq_test, 
                     beh_seq_test_cross_dataset, hyperparams,
                     free_memory=True, 
                     compute_time_writing_to_disk=False,
                     enable_cross_dataset_test=False,
                     test_only=False):
    
    is_huggingface = configs['model_opts'].get("frameworks") and configs['model_opts']["frameworks"]["hugging_faces"]

    # get the model
    model_configs = copy.deepcopy(configs['net_opts'])
    configs['model_opts']['seq_type'] = configs['data_opts']['seq_type']
    model_configs["model_opts"] = configs['model_opts']
    method_class = action_prediction(configs['model_opts']['model'])(**model_configs)

    # train and save the model
    saved_files_path = method_class.train(
        beh_seq_train, beh_seq_val, 
        **configs['train_opts'], 
        model_opts=configs['model_opts'], 
        is_huggingface=is_huggingface, 
        free_memory=free_memory,
        train_opts=configs['train_opts'],
        hyperparams=hyperparams,
        test_only=test_only
    )
    
    if free_memory:
        free_train_and_val_memory(beh_seq_train, beh_seq_val)

    # get options related to the model, only needed when it is a huggingface model
    model_opts = configs['model_opts'] if is_huggingface else None

    # test and evaluate the model
    acc, auc, f1, precision, recall = method_class.test(
        beh_seq_test, saved_files_path, 
        is_huggingface=is_huggingface,
        training_result=saved_files_path,
        model_opts=model_opts,
        test_only=test_only)
    
    if enable_cross_dataset_test and beh_seq_test_cross_dataset:
        if type(beh_seq_test_cross_dataset) is list: # model was trained on combined dataset
            print("Testing on JAAD dataset...")
            method_class.test(
                beh_seq_test_cross_dataset[0], saved_files_path, 
                is_huggingface=is_huggingface,
                training_result=saved_files_path,
                model_opts=model_opts
            )
            print("Testing on PIE dataset...")
            method_class.test(
                beh_seq_test_cross_dataset[1], saved_files_path, 
                is_huggingface=is_huggingface,
                training_result=saved_files_path,
                model_opts=model_opts
            )
        else:
            if model_opts["dataset"] == "jaad":
                model_opts["dataset"] = "pie"
            elif model_opts["dataset"] == "pie":
                model_opts["dataset"] = "jaad"
            else:
                raise
            print(f"Testing on {model_opts['dataset']} dataset...")
            method_class.test(
                beh_seq_test_cross_dataset, saved_files_path, 
                is_huggingface=is_huggingface,
                training_result=saved_files_path,
                model_opts=model_opts
            )

    
    # when the model is from huggingface, saved_files_path needs to be extracted from a dictionary
    if isinstance(saved_files_path, dict) and "saved_files_path" in saved_files_path:
        saved_files_path = saved_files_path["saved_files_path"]

    # save the results
    data = {}
    data['results'] = {}
    data['results']['acc'] = float(acc)
    data['results']['auc'] = float(auc)
    data['results']['f1'] = float(f1)
    data['results']['precision'] = float(precision)
    data['results']['recall'] = float(recall)
    write_to_yaml(yaml_path=os.path.join(saved_files_path, 'results.yaml'), data=data)

    data = configs
    write_to_yaml(yaml_path=os.path.join(saved_files_path, 'configs.yaml'), data=data)

    #print('Model saved to {}'.format(saved_files_path))

    if compute_time_writing_to_disk:
        print(f"Total time writing to disk is: {get_time_writing_to_disk()}")


def get_trajectory_sequences(configs, free_memory=False,
                             compute_cross_dataset_test=True):
    imdb, beh_seq_test_cross_dataset = None, None
    if configs['model_opts']['dataset'] == 'pie':
        imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])
        if compute_cross_dataset_test:
            imdb_cross_dataset = JAAD(data_path=os.environ.copy()['JAAD_PATH'])
    elif configs['model_opts']['dataset'] == 'jaad':
        imdb = JAAD(data_path=os.environ.copy()['JAAD_PATH'])
        if compute_cross_dataset_test:
            imdb_cross_dataset = PIE(data_path=os.environ.copy()['PIE_PATH'])
    elif configs['model_opts']['dataset'] == 'combined':
        imdb_jaad = JAAD(data_path=os.environ.copy()['JAAD_PATH'])
        imdb_pie = PIE(data_path=os.environ.copy()['PIE_PATH'])

    if imdb is not None:
        # get sequences (individual datasets, i.e. jaad or pie)
        beh_seq_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
        beh_seq_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])
        beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
        if compute_cross_dataset_test:
            data_opts_copy = copy.deepcopy(configs['data_opts'])
            beh_seq_test_cross_dataset = imdb_cross_dataset.generate_data_trajectory_sequence(
                'test', **data_opts_copy)
    elif imdb_jaad is not None:
        # get sequences for combined jaad + pie dataset
        beh_seq_train_jaad = imdb_jaad.generate_data_trajectory_sequence('train', **configs['data_opts'])
        beh_seq_val_jaad = imdb_jaad.generate_data_trajectory_sequence('val', **configs['data_opts'])
        beh_seq_test_jaad = imdb_jaad.generate_data_trajectory_sequence('test', **configs['data_opts'])
        beh_seq_train_pie = imdb_pie.generate_data_trajectory_sequence('train', **configs['data_opts'])
        beh_seq_val_pie = imdb_pie.generate_data_trajectory_sequence('val', **configs['data_opts'])
        beh_seq_test_pie = imdb_pie.generate_data_trajectory_sequence('test', **configs['data_opts'])
        
        beh_seq_train = combine_beh_seq(beh_seq_train_jaad, beh_seq_train_pie)
        beh_seq_val = combine_beh_seq(beh_seq_val_jaad, beh_seq_val_pie)
        beh_seq_test = combine_beh_seq(beh_seq_test_jaad, beh_seq_test_pie)
        if compute_cross_dataset_test:
            beh_seq_test_cross_dataset = [beh_seq_test_jaad, beh_seq_test_pie]


    if free_memory:
        # imdb dataset reference is not needed anymore
        del imdb

    return beh_seq_train, beh_seq_val, beh_seq_test, beh_seq_test_cross_dataset

def combine_beh_seq(beh_seq_jaad, beh_seq_pie):
    beh_seq = {}
    jaad_keys = set(beh_seq_jaad.keys())
    pie_keys = set(beh_seq_pie.keys())
    for k in jaad_keys.intersection(pie_keys):
        if k == "image_dimension":
            beh_seq[k] = beh_seq_jaad[k]
        else:
            beh_seq[k] = beh_seq_jaad[k] + beh_seq_pie[k]

    # Add speed # todo revisit
    beh_seq['obd_speed'] = beh_seq_jaad['vehicle_act'] + beh_seq_pie['obd_speed']

    return beh_seq

def usage():
    """
    Prints help
    """
    print('Benchmark for evaluating pedestrian action prediction.')
    print('Script for training and testing models.')
    print('Usage: python train_test.py [options]')
    print('Options:')
    print('-h, --help\t\t', 'Displays this help')
    print('-c, --config_file\t', 'Path to config file')
    print()

def set_seeds(seed=SEED):
    torch.manual_seed(seed)
    # tf.random.set_seed(seed)
    huggingface_set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 
                                   'hc:', ['help', 'config_file', 'test_only'])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)    

    set_global_determinism()

    config_file = None
    model_name = None
    dataset = None
    test_only = False

    for o, a in opts:
        if o in ["-h", "--help"]:
            usage()
            sys.exit(2)
        elif o in ['-c', '--config_file']:
            config_file = a
        elif o in ["--test_only"]:
            test_only = True


    # if neither the config file or model name are provided
    if not config_file:
        print('\x1b[1;37;41m' + 'ERROR: Provide path to config file!' + '\x1b[0m')
        usage()
        sys.exit(2)

    run(
        config_file=config_file,
        test_only=test_only
    )
