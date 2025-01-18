import copy
from utils.utils import *

from tensorflow.keras.utils import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def get_generator(
        _data: list, 
        data: dict,
        data_sizes: list,
        process: bool,
        global_pooling: str,
        model_opts: dict,
        data_type: str,
        combined_model: bool = False
        ):
    """ Get generator object 
    Args:
        _data [list]: data to use for training (list of numpy.ndarray)
        data [dict]: metadata dictionnary (with keys 'box', 'ped_id', etc...)
        data_sizes [list]: list of data size for each feature
        process [bool]: add additional processing to features when loading data
        global_pooling [str]: type of pooling
        model_opts [dict]: model options
        data_type [str]: data split (train, val, test)
        combined_model [bool]: if the model combines special data features
    """

    pytorch, tensorflow = False, True
    if model_opts.get("frameworks") and model_opts["frameworks"].get("pytorch"):
        pytorch, tensorflow = True, False
        batch_size = 1 # when using pytorch/huggingface, we want a generator object that
                       # will return a batch of 1. The actual batch size using
                       # model_opts['batch_size'] is specified in the huggingface
                       # TrainingArguments
    else:
        batch_size=model_opts['batch_size']

    if combined_model:
        input_type_list = get_input_type_list_for_combined_model(data_sizes, model_opts)
    else:
        input_type_list = model_opts['obs_input_type']       

    if tensorflow:
        target_data = _get_target_data(data, model_opts)
        new_data = (DataGenerator(data=_data,
                                labels=target_data,
                                data_sizes=data_sizes,
                                process=process,
                                global_pooling=global_pooling,
                                input_type_list=input_type_list,
                                batch_size=batch_size,
                                shuffle=data_type != 'test',
                                to_fit=data_type != 'test',
                                combined_model=combined_model
                            ), 
                    target_data
        )
    elif pytorch: # Return a DataGenerator for compatibility with data processing code.
                  # Pytorch's Dataset class will take care of building the batches later.
        target_data = _get_target_data(data, model_opts)
        new_data = (DataGenerator(data=_data,
                                  labels=target_data,
                                  data_sizes=data_sizes,
                                  process=process,
                                  global_pooling=global_pooling,
                                  input_type_list=input_type_list,
                                  batch_size=batch_size,
                                  shuffle=data_type != 'test',
                                  to_fit=data_type != 'test',
                                  combined_model=combined_model
                            ), 
                    target_data
        )
    else:
        raise Exception()
    return new_data

def _get_target_data(data: dict, model_opts: dict):
    if "tte" in model_opts.get("multi_objectives", []):
        normalized_tte_data = [(e - 30.0) / 31.0 if data['crossing'][i]==1 \
                              else np.array([1.0]) \
                              for (i, e) in enumerate(data['tte'])]

        normalized_tte_data = np.array(normalized_tte_data)
        target = np.concatenate((data['crossing'], normalized_tte_data), axis=1)
        return target
    elif "tte_pos" in model_opts.get("multi_objectives", []):
        tte_pos = np.squeeze(data['tte_pos'], axis=1)
        target = np.concatenate((data['crossing'], tte_pos), axis=1)
        return target
    elif model_opts["seq_type"] == "trajectory":
        return data['trajectories']
    else:
        return data['crossing']

def get_input_type_list_for_combined_model(data_sizes: list, 
                                           model_opts: dict):
    combine_indexes = model_opts.get("process_input_features", {"indexes_to_stack": []})["indexes_to_stack"]

    input_type_list = []
    merged_feature_name = ""
    
    # Create list of input features names (input_type_list)
    for idx in range(len(model_opts['obs_input_type'])):
        if idx in combine_indexes:
            merged_feature_name = f"{merged_feature_name}_{model_opts['obs_input_type'][idx]}"
            if idx+1 not in combine_indexes:
                input_type_list.append(merged_feature_name[1:])
        else:
            input_type_list.append(model_opts['obs_input_type'][idx])
    return input_type_list


# Tensorflow data loader (generator)
class DataGenerator(Sequence):

    def __init__(self,
                 data: list = None,
                 labels: np.ndarray = None,
                 data_sizes: list = None,
                 process: bool = False,
                 global_pooling: str = None,
                 input_type_list: list = None,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 to_fit: bool = True,
                 stack_feats: bool = False,
                 combined_model: bool = False):
        self.data = data
        self.labels = labels
        self.process = process
        self.global_pooling = global_pooling
        self.input_type_list = input_type_list
        self.batch_size = 1 if len(self.labels) < batch_size else batch_size        
        self.data_sizes = data_sizes
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.stack_feats = stack_feats
        self.indices = None
        self.combined_model = combined_model
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.data[0])/self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.data[0]))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        indices = self.indices[index*self.batch_size: (index+1)*self.batch_size]

        X = self._generate_X(indices)
        if self.to_fit:
            y = self._generate_y(indices)
            return X, y
        else: # when testing
            if self.combined_model:
                return [X]
            else:
                return X

    def _get_img_features(self, cached_path: str):
        img_features = open_pickle_file(cached_path)
        if self.process:
            if self.global_pooling == 'max':
                img_features = np.squeeze(img_features)
                img_features = np.amax(img_features, axis=0)
                img_features = np.amax(img_features, axis=0)
            elif self.global_pooling == 'avg':
                img_features = np.squeeze(img_features)
                img_features = np.average(img_features, axis=0)
                img_features = np.average(img_features, axis=0)
            else:
                img_features = img_features.ravel()        
        return img_features

    def _generate_X(self, indices: np.ndarray):
        X = []
        for input_type_idx, input_type in enumerate(self.input_type_list):
            features_batch = np.empty((self.batch_size, *self.data_sizes[input_type_idx]))
            num_ch = features_batch.shape[-1]//len(self.data[input_type_idx][0])
            for i, index in enumerate(indices):
                if isinstance(self.data[input_type_idx][index][0], str): # if str, we have an image path
                    cached_path_list = self.data[input_type_idx][index]
                    for j, cached_path in enumerate(cached_path_list):
                        if 'optical_flow' in input_type:
                            img_features = read_flow_file(cached_path)
                        else:
                            img_features = self._get_img_features(cached_path)

                        if len(cached_path_list) == 1:
                            # for static model if only one image in the sequence
                            features_batch[i, ] = img_features
                        else:
                            if self.stack_feats and 'optical_flow' in input_type:
                                features_batch[i,...,j*num_ch:j*num_ch+num_ch] = img_features
                            else:
                                features_batch[i, j, ] = img_features
                else:
                    features_batch[i, ] = self.data[input_type_idx][index]
            X.append(features_batch)
        return X

    def _generate_y(self, indices: np.ndarray):
        return np.array(self.labels[indices])


def get_static_context_data(self, 
                            model_opts: dict, 
                            data: dict, 
                            data_gen_params: dict, 
                            feature_type: str,
                            submodels_paths: dict = None):
    """ Get static context data (i.e. frame at time t-15 or at time t)"""
    
    # Get some settings from config
    concatenate_frames = "concatenate_frames" in model_opts and model_opts["concatenate_frames"]["enabled"]
    add_optical_flow = "flow_optical" in feature_type
    data_gen_params['concatenate_frames'] = concatenate_frames
    data_gen_params['feature_type'] = feature_type
    data_gen_params['is_feature_static'] = True
    get_first_static_feature_instead_of_last = (feature_type == "scene_context_with_segmentation_v5" \
                                                or "previous" in feature_type \
                                                    or model_opts["model"]=="VanMultiscale" and feature_type=="local_context")

    static_index = -1
    if get_first_static_feature_instead_of_last:
        static_index = 0

    # Keep latest element in sequence (model will be run on one frame)
    full_bbox_sequences = None
    full_rel_bbox_seq = None
    full_veh_speed = None
    data_cpy = copy.deepcopy(data)
    if not concatenate_frames and not add_optical_flow:
        for k, v in data.items():
            if 'act' not in k and v.size != 0:
                if data_gen_params["crop_type"] == "ped_overlays":
                    if k=="box_org":
                        full_bbox_sequences = data_cpy['box_org']
                    if k=="box":
                        full_rel_bbox_seq = data_cpy['box']
                    if k in ["speed", "veh_speed"]:
                        full_veh_speed = data_cpy["veh_speed"] if "veh_speed" in data_cpy else data_cpy["speed"]
                if len(v.shape) == 3:
                    data_cpy[k] = np.expand_dims(v[:, static_index, :], axis=1)
                else:
                    data_cpy[k] = np.expand_dims(v[:, static_index], axis=-1)  
    
    return self.load_images_crop_and_process(data_cpy['image'],
                                             data_cpy['box_org'],
                                             data_cpy['ped_id'],
                                             full_bbox_sequences=full_bbox_sequences,
                                             full_rel_bbox_seq=full_rel_bbox_seq,
                                             full_veh_speed=full_veh_speed,
                                             model_opts=model_opts,
                                             submodels_paths=submodels_paths,
                                             **data_gen_params)  
