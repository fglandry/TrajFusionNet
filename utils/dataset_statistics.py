import numpy as np

def get_dataset_statistics(data_train, model_opts, debug=False) -> dict:
    """ Get statistics for image data types. Statistics are only 
        computed on training data to avoid data leakage
    """
    if debug:
        """
        return {
            "dataset_means": {
                "scene_context": [0.43140541504829, 0.41739486305595647, 0.43278804277937344],
                "scene_context_with_segmentation": [0.43026995289945114, 0.41615336416516974, 0.4316534206674368, 0.02056452516315074],
            },
            "dataset_std_devs": {
                "scene_context": [0.29391770977924025, 0.2938884989361296, 0.2861626088201237],
                "scene_context_with_segmentation": [0.2942453577723304, 0.2942248280402507, 0.2865064567576991, 0.01720533010166126]
            }
        }
        return {
            "dataset_means": {
                "scene_context": [0.43140541504829, 0.41739486305595647, 0.43278804277937344],
                "local_context": [0.287280191104146, 0.2733837271647457, 0.3027266527874068],
            },
            "dataset_std_devs": {
                "scene_context": [0.29391770977924025, 0.2938884989361296, 0.2861626088201237],
                "scene_context_with_segmentation": [[0.14456805092887873, 0.14980047848430367, 0.15326681137702644]]
            }
        }
        """
        dataset_statistics = {
            "dataset_means": {
                "scene_context": [0.43140541504829, 0.41739486305595647, 0.43278804277937344],
                "local_context": [0.287280191104146, 0.2733837271647457, 0.3027266527874068],
                "scene_context_with_segmentation_v0": [0.019067334553108873, 0.019067334553108873, 0.019067334553108873],
                "scene_context_with_segmentation_v3": [0.5877854690263202, 0.4941187362881982, 0.4422615371333348]
            },
            "dataset_std_devs": {
                "scene_context": [0.29391770977924025, 0.2938884989361296, 0.2861626088201237],
                "local_context": [[0.14456805092887873, 0.14980047848430367, 0.15326681137702644]],
                "scene_context_with_segmentation_v0": [0.016265704760681676, 0.016265704760681676, 0.016265704760681676],
                "scene_context_with_segmentation_v3": [0.19343672940732676, 0.14883532650119746, 0.2593562639974769]
            }
        }

    else:
        # Calculate mean and std dev of all images in dataset
        means, std_devs = {}, {}

        for item in data_train["data"][0]:
            #if len(data_train["data_params"]["data_types"]) > 1:
            #    raise Exception
            for t_idx, data_type in enumerate(data_train["data_params"]["data_types"]):
                # Do not calculate statistics if data type is not image-like
                # TODO: use better logic
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

    # Calculate mean and std dev of trajectory features
    calculate_stats_for_trajectory_data(data_train["data"][0],
                                        data_train["data"][1],
                                        dataset_statistics)

    return dataset_statistics

def _calculate_stats_for_img_like_data(data_type, means, std_devs, item, t_idx):
    if data_type not in means:
        means[data_type] = []
        std_devs[data_type] = []
    
    img = item[0][t_idx]
    img = _format_img(img)

    #img = item[0][0]
    num_channels = img.shape[-1]
    mean = [np.mean(img[...,i]) for i in range(num_channels)]
    std_dev = [np.std(img[...,i]) for i in range(num_channels)]
    means[data_type].append(mean)
    std_devs[data_type].append(std_dev)
    #np.concatenate(means, axis=0)
    #np.concatenate(std_devs, axis=0)

def calculate_stats_for_trajectory_data(data, labels, dataset_statistics,
                                        include_labels=True,
                                        debug=True):
    
    if debug:
        dataset_statistics["dataset_maxs"], dataset_statistics["dataset_mins"] = {}, {}
        #dataset_statistics["dataset_means"]["trajectory"] = [-0.36637457009980484, -0.4659656887488598, 0.581620987878684, 1.8064563436778984, 0.11743525185289468, 0.45350332012369615, 0.6573087139027761, 0.47640759819841844, 0.7747439657553944, 0.015957067188939973, 0.08343634785066255, 6.804972165127722]
        #dataset_statistics["dataset_std_devs"]["trajectory"] = [26.58546227714022, 4.169880899944929, 26.70766158537236, 5.514246665263999, 0.08418267617663579, 0.2392682410176421, 0.04554404139545749, 0.24035221273294288, 0.061561051754394865, 3.091438807390166, 0.8245535068724321, 10.015985128031387]
        #dataset_statistics["dataset_maxs"]["trajectory"] = [372.64000000000004, 133.53000000000003, 320.46000000000004, 154.43999999999994, 0.6192685185185185, 0.99546875, 0.7741296296296296, 1.0, 1.0, 36.61500000000012, 128.0150000000002, 54.00958464000001]
        #dataset_statistics["dataset_mins"]["trajectory"] = [-324.3700000000001, -66.84000000000003, -346.69999999999993, -69.56000000000006, 0.009990740740740706, 0.0, 0.38049074074074074, 0.00321875, 0.6289166666666667, -59.49750000000006, -42.565000000000055, 0.0]
        
        
        # JAAD_ALL DATASET ============================================================

        # Get stats for 'box' + 'speed' -----------------------------------------------
        # Here, statistics include labels (pred_len=60):
        
        dataset_statistics["dataset_means"]["trajectory"] = [-7.817597278751057, -2.797630704496746, 0.7966554592107592, 15.765265538056195, 2.4889659647260074]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [161.24783689412033, 14.863155394437634, 161.40917654572766, 29.585598109148503, 1.4469956924143323]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1828.0, 126.0, 1864.0, 350.0, 4.0]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0]
        

        # Get stas for 'box' + 'speed' + 'normalized_abs_box -----------------------------------------------
        # Here, statistics include labels (pred_len=60):
        """
        dataset_statistics["dataset_means"]["trajectory"] = [-7.817597278751057, -2.797630704496746, 0.7966554592107592, 15.765265538056195, 0.46580601126806626, 0.5881889890930191, 0.49563012446195803, 0.7114383036465168, 2.7137792721052323]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [161.24783689412033, 14.863155394437634, 161.40917654572766, 29.585598109148503, 0.2441041362407904, 0.05964588498939262, 0.24599167728011406, 0.095895595316637, 1.350200146298904]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1828.0, 126.0, 1864.0, 350.0, 0.9979166666666667, 0.7694444444444445, 0.9994791666666667, 0.9990740740740741, 4.0]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0, 0.31203703703703706, 0.0078125, 0.4546296296296296, 0.0]
        """

        # JAAD_BEH DATASET ============================================================
        
        # Get stats for 'box' + 'speed' -----------------------------------------------
        # Here, statistics include labels (pred_len=60):
        """
        dataset_statistics["dataset_means"]["trajectory"] = [-17.27282171389908, -6.375631717282119, 0.8653072058708721, 30.30301819033492, 2.7640398678012854]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [225.56754977876653, 17.441062982192516, 225.14992865469864, 39.937830636813565, 1.2849233429522937]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1479.0, 86.0, 1620.0, 350.0, 4.0]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0]
        """

        """
        dataset_statistics["dataset_means"]["trajectory"] = [-7.817597278751057, -2.797630704496746, 0.7966554592107592, 15.765265538056195, 2.7137792721052323]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [161.24783689412033, 14.863155394437634, 161.40917654572766, 29.585598109148503, 1.350200146298904]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1828.0, 126.0, 1864.0, 350.0, 4.0]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0]
        """
        
        # Get stats for 'box' + 'speed' + 'normalized_abs_box -----------------------------------------------
        # Here, statistics include labels (pred_len=60):
        """
        dataset_statistics["dataset_means"]["trajectory"] = [-17.27282171389908, -6.375631717282119, 0.8653072058708721, 30.30301819033492, 0.43621347078003997, 0.5830201950981644, 0.4830560917454304, 0.7769123099210942, 2.7640398678012854]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [225.56754977876653, 17.441062982192516, 225.14992865469864, 39.937830636813565, 0.20261384396297444, 0.062278674908816625, 0.2058122991715098, 0.10028360265909717, 1.2849233429522937]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1479.0, 86.0, 1620.0, 350.0, 0.9895833333333334, 0.7694444444444445, 0.9994791666666667, 0.9990740740740741, 4.0]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1654.0, -189.0, -1655.0, -120.0, 0.0, 0.31203703703703706, 0.010416666666666666, 0.462037037037037, 0.0]
        """

        # PIE DATASET =================================================================
        
        # Get stats for 'box' + 'speed' -----------------------------------------------
        # Here, statistics include labels (pred_len=60):
        """
        dataset_statistics["dataset_means"]["trajectory"] = [-0.5568219597392173, -3.7699375720990185, 4.299549196734085, 9.470286313861626, 6.55515245931165]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [135.28868689619418, 16.119308817130346, 136.77957278484664, 23.87334432431553, 9.758136061619542]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1672.3000000000002, 221.32, 1728.14, 285.05999999999995, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1575.76, -407.0899999999999, -1589.8, -217.10000000000002, 0.0]
        """

        """
        # Here, statistics include labels (pred_len=75):

        dataset_statistics["dataset_means"]["trajectory"] = [-1.1114488312114852, -3.900570005393131, 4.382427352230639, 10.738248423248757, 6.201281409855358]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [154.028803301661, 17.441748338477698, 155.52659888485582, 26.68904158987231, 9.465415248908121]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1858.12, 243.60999999999996, 1853.8, 298.62, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1700.44, -374.34999999999997, -1731.75, -231.10000000000014, 0.0]
        """

        # Here, statistics don't include labels:
        """
        dataset_statistics["dataset_means"]["trajectory"] = [-0.36637457009980484, -0.4659656887488598, 0.581620987878684, 1.8064563436778984, 6.804972165127722]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [26.58546227714022, 4.169880899944929, 26.70766158537236, 5.514246665263999, 10.015985128031387]
        dataset_statistics["dataset_maxs"]["trajectory"] = [372.64000000000004, 133.53000000000003, 320.46000000000004, 154.43999999999994, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-324.3700000000001, -66.84000000000003, -346.69999999999993, -69.56000000000006, 0.0]
        """

        # Get stats for 'box_center_speed' only
        """
        dataset_statistics["dataset_means"]["trajectory"] = [0.015957067188939973, 0.08343634785066255]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [3.091438807390166, 0.8245535068724321]
        dataset_statistics["dataset_maxs"]["trajectory"] = [36.61500000000012, 128.0150000000002]
        dataset_statistics["dataset_mins"]["trajectory"] = [-59.49750000000006, -42.565000000000055]
        """

        # Get stats for 'box_center_speed' AND 'normalized_abs_box'
        # Here, statistics include labels (pred_len=60):
        """
        dataset_statistics["dataset_means"]["trajectory"] = [-0.5568219597392173, -3.7699375720990185, 4.299549196734085, 9.470286313861626, 0.4533823029685946, 0.6542094064638576, 0.4783747463201587, 0.7820043550188241, 6.55515245931165]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [135.28868689619418, 16.119308817130346, 136.77957278484664, 23.87334432431553, 0.24557459621324632, 0.04657779813118662, 0.247045692814803, 0.06424173232641271, 9.758136061619542]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1672.3000000000002, 221.32, 1728.14, 285.05999999999995, 0.9975, 0.7908796296296297, 1.0, 1.0, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1575.76, -407.0899999999999, -1589.8, -217.10000000000002, 0.0, 0.33631481481481484, 0.00321875, 0.6289166666666667, 0.0]
        """

        # Here, statistics don't include labels (pred_len=60):
        """
        dataset_statistics["dataset_means"]["trajectory"] = [-0.36637457009980484, -0.4659656887488598, 0.581620987878684, 1.8064563436778984, 0.11743525185289468, 0.45350332012369615, 0.6573087139027761, 0.47640759819841844, 0.015957067188939973, 0.08343634785066255, 6.804972165127722]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [26.58546227714022, 4.169880899944929, 26.70766158537236, 5.514246665263999, 0.08418267617663579, 0.2392682410176421, 0.04554404139545749, 0.24035221273294288, 3.091438807390166, 0.8245535068724321, 10.015985128031387]
        dataset_statistics["dataset_maxs"]["trajectory"] = [372.64000000000004, 133.53000000000003, 320.46000000000004, 154.43999999999994, 0.6192685185185185, 0.99546875, 0.7741296296296296, 1.0, 36.61500000000012, 128.0150000000002, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-324.3700000000001, -66.84000000000003, -346.69999999999993, -69.56000000000006, 0.009990740740740706, 0.0, 0.38049074074074074, 0.00321875, -59.49750000000006, -42.565000000000055, 0.0]
        """

        """
        # Here, statistics include labels (pred_len=50):
        dataset_statistics["dataset_means"]["trajectory"] = [-0.19605875548224377, -3.619018273308329, 4.174967956771964, 8.51623622403922, 6.775887975342747]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [121.98857066073953, 15.0458308415057, 123.44218588487205, 21.741040264776423, 9.931103399029672]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1612.6, 210.61999999999995, 1632.42, 284.2399999999999, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1506.97, -327.83000000000004, -1539.82, -207.20000000000005, 0.0]
        """

        """
        # Get stats for 'box' + 'veh_speed' AND 'normalized_abs_box'
        # Including labels
        dataset_statistics["dataset_means"]["trajectory"] = [-0.5568219597392173, -3.7699375720990185, 4.299549196734085, 9.470286313861626, 0.4533823029685946, 0.6542094064638576, 0.4783747463201587, 0.7820043550188241, 6.55515245931165]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [135.28868689619418, 16.119308817130346, 136.77957278484664, 23.87334432431553, 0.24557459621324632, 0.04657779813118662, 0.247045692814803, 0.06424173232641271, 9.758136061619542]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1672.3000000000002, 221.32, 1728.14, 285.05999999999995, 0.9975, 0.7908796296296297, 1.0, 1.0, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1575.76, -407.0899999999999, -1589.8, -217.10000000000002, 0.0, 0.33631481481481484, 0.00321875, 0.6289166666666667, 0.0]
        """

        """
        # Here, statistics don't include labels
        dataset_statistics["dataset_means"]["trajectory"] = [-0.36637457009980484, -0.4659656887488598, 0.581620987878684, 1.8064563436778984, 0.11743525185289468, 0.45350332012369615, 0.6573087139027761, 0.47640759819841844, 6.804972165127722]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [26.58546227714022, 4.169880899944929, 26.70766158537236, 5.514246665263999, 0.08418267617663579, 0.2392682410176421, 0.04554404139545749, 0.24035221273294288, 10.015985128031387]
        dataset_statistics["dataset_maxs"]["trajectory"] = [372.64000000000004, 133.53000000000003, 320.46000000000004, 154.43999999999994, 0.6192685185185185, 0.99546875, 0.7741296296296296, 1.0, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-324.3700000000001, -66.84000000000003, -346.69999999999993, -69.56000000000006, 0.009990740740740706, 0.0, 0.38049074074074074, 0.00321875, 0.0]
        """
        
        """
        dataset_statistics["dataset_means"]["trajectory"] = [0.11743525185289468, 0.45350332012369615, 0.6573087139027761, 0.47640759819841844]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [0.08418267617663579, 0.2392682410176421, 0.04554404139545749, 0.24035221273294288]
        dataset_statistics["dataset_maxs"]["trajectory"] = [0.6192685185185185, 0.99546875, 0.7741296296296296, 1.0]
        dataset_statistics["dataset_mins"]["trajectory"] = [0.009990740740740706, 0.0, 0.38049074074074074, 0.00321875]
        """

        """
        # Get stats for 'normalized_abs_box' only
        # Including labels
        dataset_statistics["dataset_means"]["trajectory"] = [0.4533823029685946, 0.6542094064638576, 0.4783747463201587, 0.7820043550188241]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [0.24557459621324632, 0.04657779813118662, 0.247045692814803, 0.06424173232641271]
        dataset_statistics["dataset_maxs"]["trajectory"] = [0.9975, 0.7908796296296297, 1.0, 1.0]
        dataset_statistics["dataset_mins"]["trajectory"] = [0.0, 0.33631481481481484, 0.00321875, 0.6289166666666667]
        """
        """
        # Get stats for 'box' + 'veh_speed' AND 'pose'
        # Including labels
        dataset_statistics["dataset_means"]["trajectory"] = [-0.5568219597392173, -3.7699375720990185, 4.299549196734085, 9.470286313861626, 0.003550792011275119, -0.0091492031315951, -0.051244509295147116, 0.06789969603904114, -0.11137654328545059, -0.022230101072390818, -0.051415528087193, 0.0254877782419563, -0.08652261516016274, 0.019509162372909977, -0.05727504213422123, -0.01700033864261815, -0.045183280935369614, -0.6096585536173024, -0.4911827896749532, -0.5039452490968922, -0.2948595435030048, -0.2975388338573535, -0.17536217249919825, -0.18592136647950336, -0.07262964232935883, -0.06516935537294899, 0.28083662794272457, 0.2791936263268543, 0.6883216503984892, 0.6866982278960365, 6.55515245931165]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [135.28868689619418, 16.119308817130346, 136.77957278484664, 23.87334432431553, 0.4325399717161851, 0.4206353604282518, 0.4145936537219253, 0.5556660047890032, 0.5603259419071277, 0.5474116208201766, 0.5440026779950597, 0.3958187919789144, 0.40918641930317634, 0.3598325514559303, 0.37287864290621986, 0.452615847735729, 0.4539461340924975, 0.34088836873771183, 0.2654001389941785, 0.24720147690689018, 0.3403207825448186, 0.3325769439062082, 0.39215386094955235, 0.3907348104812882, 0.3109437017289478, 0.3080797055631934, 0.2810464638807129, 0.278802567367444, 0.2474655814628918, 0.2509620118275534, 9.758136061619542]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1672.3000000000002, 221.32, 1728.14, 285.05999999999995, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9790940766550522, 0.9642857142857143, 0.9574468085106383, 0.9791666666666666, 0.9790575916230366, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1575.76, -407.0899999999999, -1589.8, -217.10000000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
        """
        """
        # Get stats for 'box' + 'veh_speed' AND 'pose' (10 keypoints)
        # Including labels
        dataset_statistics["dataset_means"]["trajectory"] = [-0.5568219597392173, -3.7699375720990185, 4.299549196734085, 9.470286313861626, 0.003550792011275119, -0.011704036567403596, -0.05103098052677972, 0.049496981159384515, -0.09746174788628446, -0.1348570916595387, -0.07485944706041409, -0.017445919728579687, 0.0678678251669309, 0.15017132624131618, 6.55515245931165]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [135.28868689619418, 16.119308817130346, 136.77957278484664, 23.87334432431553, 0.4325399717161851, 0.4477548155278144, 0.4422137290970258, 0.538610110787377, 0.5419000031067103, 0.5623587874674537, 0.5220728902892208, 0.4041670818506781, 0.4874945632702371, 0.4312069653432559, 9.758136061619542]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1672.3000000000002, 221.32, 1728.14, 285.05999999999995, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9790940766550522, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1575.76, -407.0899999999999, -1589.8, -217.10000000000002, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
        """

        """
        # Get stats for 'pose' only (10 keypoints)
        # Including labels
        dataset_statistics["dataset_means"]["trajectory"] = [0.003550792011275119, -0.022230101072390818, -0.051415528087193, -0.01700033864261815, -0.045183280935369614, -0.6096585536173024, -0.17536217249919825, -0.18592136647950336, 0.6883216503984892, 0.6866982278960365]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [0.4325399717161851, 0.5474116208201766, 0.5440026779950597, 0.452615847735729, 0.4539461340924975, 0.34088836873771183, 0.39215386094955235, 0.3907348104812882, 0.2474655814628918, 0.2509620118275534]
        dataset_statistics["dataset_maxs"]["trajectory"] = [0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9722222222222222, 0.9790940766550522, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666, 0.9791666666666666]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
        """

        """
        # Get stats for 'box' + 'veh_speed' AND 'box_center_speed'
        # Including labels
        dataset_statistics["dataset_means"]["trajectory"] = [-0.5568219597392173, -3.7699375720990185, 4.299549196734085, 9.470286313861626, 0.08852127997524409, 0.06437680793643977, 6.55515245931165]
        dataset_statistics["dataset_std_devs"]["trajectory"] = [135.28868689619418, 16.119308817130346, 136.77957278484664, 23.87334432431553, 3.913892919657201, 0.8891769611605792, 9.758136061619542]
        dataset_statistics["dataset_maxs"]["trajectory"] = [1672.3000000000002, 221.32, 1728.14, 285.05999999999995, 76.0899999999998, 128.0150000000002, 54.00958464000001]
        dataset_statistics["dataset_mins"]["trajectory"] = [-1575.76, -407.0899999999999, -1589.8, -217.10000000000002, -76.76750000000001, -42.565000000000055, 0.0]
        """

        # Get stats for 'box' + 'speed' -----------------------------------------------
        # Here, pred_len=30 at the beginning + end of the prediction horizon; statistics include labels.
        """
        dataset_statistics["dataset_means"]["trajectory"] = {
            [-0.7525799485353427, -1.6542208088518515, 1.970245361116053, 5.305364862712205, 6.666527738683615]
        }
        dataset_statistics["dataset_std_devs"]["trajectory"] = {
            [76.22710860030644, 9.304785382572415, 76.84046494799227, 13.771352751791904, 9.88295057427375],
        }
        dataset_statistics["dataset_maxs"]["trajectory"] = {
            [899.4100000000001, 178.2, 918.3199999999999, 166.58000000000004, 54.00958464000001],
        }
        dataset_statistics["dataset_mins"]["trajectory"] = {
            [-920.3999999999999, -129.12, -888.21, -145.08000000000015, 0.0],
        }
        """

    else:
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

    test = 10


def _format_img(img: np.ndarray):
    img = img / 255 # Normalize to a value between 0 and 1
    if img.shape[0] == 1 and img.shape[1] == 1:
        img = np.squeeze(img, axis=(0,1))
    if len(img.shape) < 3:
        img = np.expand_dims(img, axis=2)
    return img

