import copy
import os
import sys
import yaml

from datasets_data.pie_data import PIE
from datasets_data.jaad_data import JAAD
from train_test import action_prediction


def test_model(saved_files_path=None):

    model_opts, net_opts, data_opts = get_configurations(saved_files_path)

    if model_opts['dataset'] == 'pie':
        imdb = PIE(data_path=os.environ.copy()['PIE_PATH'])
        imdb.get_data_stats()
    elif model_opts['dataset'] == 'jaad':
        imdb = JAAD(data_path=os.environ.copy()['JAAD_PATH'])
    else:
        raise ValueError("{} dataset is incorrect".format(model_opts['dataset']))

    # get the model
    model_configs = copy.deepcopy(net_opts)
    model_configs["model_opts"] = model_opts
    method_class = action_prediction(model_opts['model'])(**model_configs)

    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **data_opts)
    
    acc, auc, f1, precision, recall = method_class.test(beh_seq_test, saved_files_path)

def get_configurations(saved_files_path):

    with open(os.path.join(saved_files_path, 'configs.yaml'), 'r') as yamlfile:
        opts = yaml.safe_load(yamlfile)
    print(opts)
    model_opts = opts['model_opts']
    data_opts = opts.get('data_opts')
    net_opts = opts.get('net_opts')

    if not data_opts:
        print("WARNING: data_opts cannot be found in model file")
        data_opts = {
              "fstride": 1,
              "sample_type": "beh", 
              "subset": "default",
              "data_split_type": "default",
              "seq_type": "crossing",
              "min_track_size": 16
        }
    if not net_opts:
        print("WARNING: net_opts cannot be found in model file")
        net_opts = {
              "num_hidden_units": 128,
              "global_pooling": "avg", 
              "regularizer_val": 0.0001,
              "cell_type": "gru",
              "backbone": "vgg16",
              "freeze_conv_layers": False
        }

    tte = model_opts['time_to_event'] if isinstance(model_opts['time_to_event'], int) else \
                model_opts['time_to_event'][1]
    data_opts['min_track_size'] = model_opts['obs_length'] + tte

    return model_opts, net_opts, data_opts

if __name__ == '__main__':
    saved_files_path = sys.argv[1]
    test_model(saved_files_path=saved_files_path)