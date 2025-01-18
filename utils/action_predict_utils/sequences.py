import copy
import os

from datasets_data.jaad_data import JAAD
from datasets_data.pie_data import PIE


def get_trajectory_sequences(configs: dict, free_memory: bool = False,
                             compute_cross_dataset_test: bool = False):
    """ Generate trajectory sequence as a function of dataset """
    
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
    """ Combine sequences of two datasets """

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

def compute_sequences(d: dict, data_raw: dict, opts: dict, 
                      obs_length: int, time_to_event: list, olap_res: int,
                      add_normalized_abs_box: bool = False, 
                      add_box_center_speed: bool = False,
                      action_predict_obj_ref = None):
    """ Compute sequences (t=16) from pedestrian tracks
    Args:
        d [dict]: sequence data dictionary
        data_raw [dict]: raw data dictionary
        opts [dict]: model opts
        obs_length [int]: observation period length
        time_to_event [list]: list of two time values which consist of the start and
                              the end of the prediction horizon
                              (see https://doi.org/10.1016/j.neucom.2024.129105)
        olap_res [int]: overlap factor
        add_normalized_abs_box [bool]: add normalized absolute box coordinates to
                                       sequence data dictionary
        add_box_center_speed [bool]: add bounding box center speed to sequence data dictionary
    """
    trajectories = []

    if add_normalized_abs_box:
        d['normalized_abs_box_org'] = copy.deepcopy(d['box_org'])
        d = action_predict_obj_ref.compute_positions_normalization_divide_by_img_dims(
            d, 'normalized_abs_box_org', opts, replace_by_patch_id=False)
    if add_box_center_speed:
        d['box_center_speed'] = copy.deepcopy(d['center'])
        d = action_predict_obj_ref.compute_box_speed(d, k="box_center_speed")
        
    for k in d.keys():
        seqs = []
        for seq_idx, seq in enumerate(d[k]):
            if opts.get("seq_type")=="trajectory":
                TRAJECTORY_PREDICTION_LENGTH = 60
                start_idx = 0
                end_idx = len(seq) - obs_length - TRAJECTORY_PREDICTION_LENGTH
                seqs.extend([seq[i:i + obs_length] for i in
                                range(start_idx, end_idx + 1, olap_res)])
                if k == "box_org":
                    # Compute trajectory data if requested
                    speed_seq = d["speed_org"][seq_idx]
                    if add_normalized_abs_box:
                        normalized_abs_box_seq = d["normalized_abs_box_org"][seq_idx]
                        # combined_seq = [s + normalized_abs_box_seq[idx] + speed_seq[idx] for idx, s in enumerate(seq)]
                        combined_seq = [normalized_abs_box_seq[idx] for idx, s in enumerate(seq)]
                    elif add_box_center_speed:
                        box_center_speed_seq = d["box_center_speed"][seq_idx]
                        combined_seq = [s + box_center_speed_seq[idx] + speed_seq[idx] for idx, s in enumerate(seq)]
                    else:
                        combined_seq = [s + speed_seq[idx] for idx, s in enumerate(seq)]

                    # Get trajectory following observation length
                    start_idx = 0
                    end_idx = len(seq) - obs_length - TRAJECTORY_PREDICTION_LENGTH
                    trajectories.extend([combined_seq[i+obs_length-1:i+obs_length+TRAJECTORY_PREDICTION_LENGTH] \
                                        for i in range(start_idx, end_idx + 1, olap_res)])
                    for idx, t in enumerate(trajectories):
                        if len(t) != TRAJECTORY_PREDICTION_LENGTH+1:
                            raise Exception
            else:
                start_idx = len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - time_to_event[0]
                seqs.extend([seq[i:i + obs_length] for i in
                                range(start_idx, end_idx + 1, olap_res)])
            
        d[k] = seqs
    if opts.get("seq_type")=="trajectory":
        d["trajectories"] = trajectories

    for seq in data_raw['bbox']:
        start_idx = len(seq) - obs_length - time_to_event[1]
        end_idx = len(seq) - obs_length - time_to_event[0]
        d['tte'].extend([[len(seq) - (i + obs_length)] for i in
                        range(start_idx, end_idx + 1, olap_res)])
        # Get ped position at the tte frame
        d['tte_pos'].extend([[seq[-1]] for i in
                                range(start_idx, end_idx + 1, olap_res)])
    return d
