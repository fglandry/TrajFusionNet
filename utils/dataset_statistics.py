import numpy as np
from typing import Any


def get_dataset_statistics(data_train: dict, model_opts: dict, 
                           use_precomputed_values: bool = False) -> dict:
    """ Get dataset statistics for various data features (mean, std dev, etc.). 
        Statistics are only computed on training data to avoid data leakage
    Args:
        data_train [dict]: training data dictionary
        model_opts [dict]: model options dictionary
        use_precomputed_values [bool]: if set to True, precomputed values will be used
                                       instead of recomputing values
    """
    if use_precomputed_values:
        dataset_statistics = {
            "dataset_means": {
                "scene_context": [0.43140541504829, 0.41739486305595647, 0.43278804277937344],
                "local_context": [0.287280191104146, 0.2733837271647457, 0.3027266527874068],
            },
            "dataset_std_devs": {
                "scene_context": [0.29391770977924025, 0.2938884989361296, 0.2861626088201237],
                "local_context": [[0.14456805092887873, 0.14980047848430367, 0.15326681137702644]],
            }
        }

    else:
        # Calculate mean and std dev of all images in dataset
        means, std_devs = {}, {}

        for item in data_train["data"][0]:
            for t_idx, data_type in enumerate(data_train["data_params"]["data_types"]):
                # Do not calculate statistics if data type is not image-like
                if (type(item) is tuple or len(item.shape) <= 3) and \
                    "context" not in data_type:
                    continue
                else:
                    _calculate_stats_for_img_like_data(data_type, means, std_devs, item, t_idx)

        dataset_means, dataset_std_devs = {}, {}
        for data_type in data_train["data_params"]["data_types"]:
            if not data_type in means:
                print(f"WARNING: No statistics computed for data type {data_type}")
                continue

            means_t = np.asarray(means[data_type])
            dataset_means[data_type] = np.mean(means_t, axis=0).tolist()

            std_devs_t = np.asarray(std_devs[data_type])
            dataset_std_devs[data_type] = np.mean(std_devs_t, axis=0).tolist()

        dataset_statistics = {
            "dataset_means": dataset_means,
            "dataset_std_devs": dataset_std_devs
        }

    # Calculate statistics for trajectory features
    calculate_stats_for_trajectory_data(data_train["data"][0],
                                        data_train["data"][1],
                                        dataset_statistics,
                                        model_opts,
                                        use_precomputed_values=True)

    return dataset_statistics


def _calculate_stats_for_img_like_data(data_type, means, std_devs, item, t_idx):
    if data_type not in means:
        means[data_type] = []
        std_devs[data_type] = []
    
    img = item[0][t_idx]
    img = _format_img(img)

    num_channels = img.shape[-1]
    mean = [np.mean(img[...,i]) for i in range(num_channels)]
    std_dev = [np.std(img[...,i]) for i in range(num_channels)]
    means[data_type].append(mean)
    std_devs[data_type].append(std_dev)


def calculate_stats_for_trajectory_data(data: Any, labels: np.ndarray, 
                                        dataset_statistics: dict, model_opts: dict,
                                        include_labels: bool = True,
                                        use_precomputed_values: bool = True):
    
    if use_precomputed_values:
        dataset_statistics["dataset_maxs"], dataset_statistics["dataset_mins"] = {}, {}

        # Get stats for 'box' + 'speed'
        # Here, statistics include labels (pred_len=60)
        if model_opts["dataset_full"] == "jaad_all":
            
            dataset_statistics["dataset_means"]["trajectory"] = [-7.817597278751057, -2.797630704496746, 0.7966554592107592, 15.765265538056195, 2.4889659647260074]
            dataset_statistics["dataset_std_devs"]["trajectory"] = [161.24783689412033, 14.863155394437634, 161.40917654572766, 29.585598109148503, 1.4469956924143323]
            dataset_statistics["dataset_maxs"]["trajectory"] = [1828.0, 126.0, 1864.0, 350.0, 4.0]
            dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0]
        
        elif model_opts["dataset_full"] == "jaad_beh":

            dataset_statistics["dataset_means"]["trajectory"] = [-17.27282171389908, -6.375631717282119, 0.8653072058708721, 30.30301819033492, 2.7640398678012854]
            dataset_statistics["dataset_std_devs"]["trajectory"] = [225.56754977876653, 17.441062982192516, 225.14992865469864, 39.937830636813565, 1.2849233429522937]
            dataset_statistics["dataset_maxs"]["trajectory"] = [1479.0, 86.0, 1620.0, 350.0, 4.0]
            dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0]
        
        elif model_opts["dataset_full"] == "pie":

            dataset_statistics["dataset_means"]["trajectory"] = [-0.5568219597392173, -3.7699375720990185, 4.299549196734085, 9.470286313861626, 6.55515245931165]
            dataset_statistics["dataset_std_devs"]["trajectory"] = [135.28868689619418, 16.119308817130346, 136.77957278484664, 23.87334432431553, 9.758136061619542]
            dataset_statistics["dataset_maxs"]["trajectory"] = [1672.3000000000002, 221.32, 1728.14, 285.05999999999995, 54.00958464000001]
            dataset_statistics["dataset_mins"]["trajectory"] = [-1575.76, -407.0899999999999, -1589.8, -217.10000000000002, 0.0]

    else: # compute statistics
        traj_np = []
        for item in data:
            traj_values = item[0][-1]
            traj_np.append(traj_values.tolist())
        traj_np = np.array(traj_np).squeeze(axis=1)
        traj_np = np.reshape(traj_np, (traj_np.shape[0]*traj_np.shape[1], traj_np.shape[-1]))

        if include_labels:
            labels_np = []
            for item in labels:
                labels_np.append(item.tolist())
            labels_np = np.array(labels_np)
            labels_np = np.reshape(labels_np, 
                                   (labels_np.shape[0]*labels_np.shape[1], labels_np.shape[-1]))
            traj_np = np.concatenate((traj_np, labels_np), axis=0)

        means = np.mean(traj_np, axis=0)
        std_devs = np.std(traj_np, axis=0)
        maxs = np.max(traj_np, axis=0)
        mins = np.min(traj_np, axis=0)

        dataset_statistics["dataset_means"]["trajectory"] = list(means)
        dataset_statistics["dataset_std_devs"]["trajectory"] = list(std_devs)
        dataset_statistics["dataset_maxs"], dataset_statistics["dataset_mins"] = {}, {}
        dataset_statistics["dataset_maxs"]["trajectory"] = list(maxs)
        dataset_statistics["dataset_mins"]["trajectory"] = list(mins)


def _format_img(img: np.ndarray):
    img = img / 255 # Normalize to a value between 0 and 1
    if img.shape[0] == 1 and img.shape[1] == 1:
        img = np.squeeze(img, axis=(0,1))
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
    return img
