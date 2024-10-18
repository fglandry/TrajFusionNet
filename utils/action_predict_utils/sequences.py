import copy

def compute_sequences(d, data_raw, opts, obs_length, time_to_event, olap_res,
                      add_normalized_abs_box=False, 
                      add_box_center_speed=False,
                      add_pose=False,
                      action_predict_obj_ref=None):
    # olap_res = 3 # todo: remove
    trajectories = []

    if add_normalized_abs_box:
        d['normalized_abs_box_org'] = copy.deepcopy(d['box_org'])
        d = action_predict_obj_ref.compute_positions_normalization_divide_by_img_dims(
            d, 'normalized_abs_box_org', opts, replace_by_patch_id=False)
    if add_box_center_speed:
        d['box_center_speed'] = copy.deepcopy(d['center'])
        d = action_predict_obj_ref.compute_box_speed(d, k="box_center_speed")
    if add_pose:
        d['pose'] = copy.deepcopy(d['ped_id'])
        d = action_predict_obj_ref.compute_pose(d, k="pose", model_opts=opts)

                                
    for k in d.keys():
        seqs = []
        for seq_idx, seq in enumerate(d[k]):
            if opts.get("seq_type")=="trajectory":
                TRAJECTORY_PREDICTION_LENGTH = 60
                start_idx = 0 # len(seq) - obs_length - time_to_event[1]
                end_idx = len(seq) - obs_length - TRAJECTORY_PREDICTION_LENGTH # len(seq) - obs_length - time_to_event[1]
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
                    elif add_pose:
                        pose = d["pose"][seq_idx]
                        pose = [[p for i, p in enumerate(points) if i in [0,5,6,11,12,13,18,19,24,25]] for points in pose]
                        # combined_seq = [pose[idx] for idx, s in enumerate(seq)]
                        combined_seq = [s + pose[idx] + speed_seq[idx] for idx, s in enumerate(seq)]
                        test = 10
                    else:
                        combined_seq = [s + speed_seq[idx] for idx, s in enumerate(seq)]


                    # Get trajectory following observation length
                    start_idx = 0 # len(seq) - obs_length - time_to_event[1]
                    # end_idx = len(seq) - obs_length - time_to_event[1]
                    end_idx = len(seq) - obs_length - TRAJECTORY_PREDICTION_LENGTH # TODO: verify
                    trajectories.extend([combined_seq[i+obs_length-1:i+obs_length+TRAJECTORY_PREDICTION_LENGTH] \
                                        for i in range(start_idx, end_idx + 1, olap_res)]) # obs_length+16]
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