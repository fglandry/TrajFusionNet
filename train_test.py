import copy
import getopt
import os
import random
import sys
import yaml

import numpy as np
from transformers import set_seed as huggingface_set_seed
import torch

from action_predict import ActionPredict
from models.models import *
from models.multi_branch_models.combined_models import *
from utils.action_predict_utils.sequences import get_trajectory_sequences 
from utils.global_variables import get_time_writing_to_disk
from utils.hyperparameters import HyperparamsOrchestrator
from utils.utils import IndentedDumper

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

def action_prediction(model_name: str):
    for cls in ActionPredict.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    for cls in Static.__subclasses__():
        if cls.__name__ == model_name:
            return cls
    raise Exception('Model {} is not valid!'.format(model_name))

def run(config_file: str = None,
        dataset_override: str = None,
        seq_type_override: str = None,
        traj_model_path_override: str = None,
        test_only: bool = False,
        train_end_to_end: bool = False,
        free_memory: bool = True, 
        compute_time_writing_to_disk: bool = False,
        tune_hyperparameters: bool = False
    ):
    """
    Run train and test on the dataset with parameters specified in configuration file.
    
    Args:
        config_file [str]: path to configuration file in yaml format
        dataset_override [str]: if specified, overrides the dataset specified in the config file
                                (pie, jaad_all, or jaad_beh)
        seq_type_override [str]: if specified, overrides the seq_type specified in the config file
                                 (trajectory or crossing)
        traj_model_path_override [str]: if specified, overrides the checkpoint hardcoded
                                        in (small)trajectorytransformer.py
        test_only [bool]: if True, only inference will be performed (no training)
        train_end_to_end [bool]: if True, all modules in the network will be trained (see modular
                                 training section in the paper)
        free_memory [bool]: free memory by deleting some variables along the way
        compute_time_writing_to_disk [bool]: compute time writing to disk
        tune_hyperparameters [bool]: fine-tune hyperparameters
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
    if dataset_override:
        model_configs["exp_opts"]["datasets"] = [dataset_override]
        with open(config_file, 'w') as f:
            yaml.dump(model_configs, f, Dumper=IndentedDumper,
                      default_flow_style=None, sort_keys=False)

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
        configs['data_opts']['seq_type'] = seq_type_override if seq_type_override else configs['data_opts']['seq_type']
        configs['model_opts']['overlap'] = 0.6 if 'pie' in dataset else 0.8
        configs['model_opts']['dataset'] = dataset.split('_')[0]
        configs['model_opts']['dataset_full'] = dataset
        configs['model_opts']['traj_model_path_override'] = traj_model_path_override if traj_model_path_override else ""
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
        
        model = ""
        submodel = ""
        hyperparams_orchestrator = HyperparamsOrchestrator(tune_hyperparameters, model, submodel)
        for i in range(hyperparams_orchestrator.nb_cases):
            hyperparams = hyperparams_orchestrator.get_next_case()
            if hyperparams:
                print(f"Training model with hyperparams set {i}: {str(hyperparams[model][submodel])}")
            saved_files_path = \
                train_test_model(configs, beh_seq_train, beh_seq_val, beh_seq_test,
                                 beh_seq_test_cross_dataset, hyperparams,
                                 test_only=test_only,
                                 train_end_to_end=train_end_to_end)
        if dataset_override:
            break
    return saved_files_path # return path for last trained model

        
def train_test_model(configs: dict, beh_seq_train: dict, 
                     beh_seq_val: dict, beh_seq_test: dict, 
                     beh_seq_test_cross_dataset: dict, hyperparams: dict,
                     free_memory: bool = True, 
                     compute_time_writing_to_disk: bool = False,
                     enable_cross_dataset_test: bool = False,
                     test_only: bool = False,
                     train_end_to_end: bool = False):
    
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
        test_only=test_only,
        train_end_to_end=train_end_to_end
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

    if compute_time_writing_to_disk:
        print(f"Total time writing to disk is: {get_time_writing_to_disk()}")

    return saved_files_path

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
                                   'hc:d:s:j:', ['help', 'config_file', 'dataset', 
                                                 'seq_type', 'traj_model_path',
                                                 'test_only', 'train_end_to_end'])
    except getopt.GetoptError as err:
        print(str(err))
        usage()
        sys.exit(2)    

    set_global_determinism()

    config_file = None
    model_name = None
    dataset, seq_type, traj_model_path = None, None, None
    test_only, train_end_to_end = False, False

    for o, a in opts:
        if o in ["-h", "--help"]:
            usage()
            sys.exit(2)
        elif o in ["-c", "--config_file"]:
            config_file = a
        elif o in ["-d", "--dataset"]:
            dataset = a
        elif o in ["-s", "--seq_type"]:
            seq_type = a
        elif o in ["-j", "--traj_model_path"]:
            traj_model_path = a
        elif o in ["--test_only"]:
            test_only = True
        elif o in ["--train_end_to_end"]:
            train_end_to_end = True

    # if neither the config file or model name are provided
    if not config_file:
        print('\x1b[1;37;41m' + 'ERROR: Provide path to config file!' + '\x1b[0m')
        usage()
        sys.exit(2)

    saved_files_path = run(
        config_file=config_file,
        test_only=test_only,
        train_end_to_end=train_end_to_end,
        dataset_override=dataset,
        seq_type_override=seq_type,
        traj_model_path_override=traj_model_path
    )
    print(f"Model saved under: {saved_files_path}")
