import math
import time
import yaml
import copy
import cv2

#import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.layers import GRU, LSTM, RNN
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import regularizers
from tensorflow.keras.utils import custom_object_scope
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

from models.custom_layers import CustomAttention, T2V
from utils.action_predict_utils.trajectory_overlays import TrajectoryOverlays
from utils.action_predict_utils.sequences import compute_sequences
from utils.data_load import get_generator, get_static_context_data
from utils.dataset_statistics import get_dataset_statistics
from utils.optical_flow import process_optical_flow_v2
from utils.utils import *

class ActionPredict(object):
    """
        A base interface class for creating prediction models
    """

    def __init__(self,
                 global_pooling='avg',
                 regularizer_val=0.0001,
                 backbone='vgg16',
                 **kwargs):
        """
        Class init function
        Args:
            global_pooling: Pooling method for generating convolutional features
            regularizer_val: Regularization value for training
            backbone: Backbone for generating convolutional features
        """
        # Network parameters
        self._regularizer_value = regularizer_val
        self._regularizer = regularizers.l2(regularizer_val)
        self._global_pooling = global_pooling
        self._backbone = backbone
        self._generator = None # use data generator for train/test

        # Framework-related settings (tensorflow, pytorch, etc. )
        self.model_configs = kwargs["model_opts"]
        del kwargs["model_opts"] # to avoid downstream errors when training models
        frameworks = self.model_configs["frameworks"] if "frameworks" in self.model_configs else {}
        self.is_tensorflow = not frameworks.get("pytorch") if frameworks else True
        #self.is_tensorflow = False # to remove
        if self.is_tensorflow:
            # Create a strategy for multi-gpu data parallelism (tensorflow models)
            gpus_found = tf.config.list_physical_devices('GPU')
            print(f"The following devices have been found on the machine: {gpus_found}")

            if gpus_found:
                self.multi_gpu_strategy = tf.distribute.MirroredStrategy()
            else:
                self.multi_gpu_strategy = tf.distribute.get_strategy()
    
    def get_concatenated_image(self, 
                                imgs_to_concatenate, 
                                img_seq, 
                                regen_data, 
                                target_dim):
        
        nb_imgs_to_concat = len(img_seq)
        assert nb_imgs_to_concat == 4

        # Use the save path of the first image in sequence and append 'concat'
        # to its path name to obtain the concatenated img save path
        first_img_save_path = img_seq[0]
        path_left, ext = first_img_save_path.rsplit(".", 1)[0], first_img_save_path.rsplit(".", 1)[1]
        new_img_save_path = path_left + "_concat." + ext

        file_already_exists = os.path.exists(new_img_save_path) and not regen_data
        if file_already_exists:
            img_seq = []
            if self._generator:
                img_seq.append(new_img_save_path)
            else:
                img_features = open_pickle_file(new_img_save_path)
                img_seq.append(img_features)
        else:
            # Generate concatenated image features
            imgs_to_concatenate = imgs_to_concatenate if self._generator else img_seq

            # Change all images (to concatenate together) to the same dimensions
            heights = [img.shape[0] for img in imgs_to_concatenate]
            if len(set(heights)) != 1:
                #min_height = min(imgs_to_concatenate, key=lambda x: x.shape[0]).shape[0]
                min_height = min(heights)
                for i in range(len(imgs_to_concatenate)):
                    imgs_to_concatenate[i] = img_pad(imgs_to_concatenate[i], mode='pad_resize', size=min_height)
                
            img_top = np.concatenate((imgs_to_concatenate[0], imgs_to_concatenate[1]), axis=1)
            img_bottom = np.concatenate((imgs_to_concatenate[3], imgs_to_concatenate[2]), axis=1)
            new_img_data = np.concatenate((img_top, img_bottom), axis=0)

            new_img_features = img_pad(new_img_data, mode='pad_resize', size=target_dim[0])
            # new_img_features = cv2.resize(new_img_data, target_dim)
            # show_image(new_img_features)
            
            # Save concat img to disk
            with open(new_img_save_path, 'wb') as fid:
                pickle.dump(new_img_features, fid, pickle.HIGHEST_PROTOCOL)
            
            # Build img_seq
            img_seq = []
            if self._generator:
                img_seq.append(new_img_save_path)
            else:
                img_seq.append(new_img_features)
        
        return img_seq

    # Processing images and generate features
    def load_images_crop_and_process(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     full_bbox_sequences=None,
                                     full_rel_bbox_seq=None,
                                     full_veh_speed=None,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     feature_type='',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     process=False,
                                     regen_data=False,
                                     concatenate_frames=False,
                                     is_feature_static=False,
                                     store_data_only=False,
                                     model_opts=None,
                                     debug=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            feature_type: e.g. local_context, box, speed, pose, etc.
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            process: Whether process the raw images using a neural network
            regen_data: Whether regenerate visual features. This will overwrite the cached features
            concatenate_frames: add concatenated frames feature to returned sequence
            is_feature_static: whether the feature is static (we should only keep the last frame)
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """
        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
              \nsave_path={}, ".format(data_type, crop_type, crop_mode,
                                       save_path))
        preprocess_dict = {'vgg16': vgg16.preprocess_input, 'resnet50': resnet50.preprocess_input}
        backbone_dict = {'vgg16': vgg16.VGG16, 'resnet50': resnet50.ResNet50}

        preprocess_input = preprocess_dict.get(self._backbone, None)
        if process:
            assert (self._backbone in ['vgg16', 'resnet50']), "{} is not supported".format(self._backbone)

        convnet = backbone_dict[self._backbone](input_shape=(224, 224, 3),
                                                include_top=False, weights='imagenet') if process else None
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            img_seq, imgs_to_concatenate = [], []
            img_save_path = ""
            prev_img_list, prev_img_data = [], None

            # For each image in sequence
            seq_idx = 0
            for imp, b, p in zip(seq, bbox_seq[i], pid):

                img_data, img_features, flip_image = None, None, False
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                img_save_folder = os.path.join(save_path, set_id, vid_id)

                # Modify the path depending on crop mode
                if crop_type == 'none':
                    img_save_path = os.path.join(img_save_folder, img_name + '.pkl')
                else:
                    img_save_path = os.path.join(img_save_folder, img_name + '_' + p[0] + '.pkl')

                # Check whether the file exists
                file_already_exists = os.path.exists(img_save_path) and not regen_data
                if file_already_exists and not concatenate_frames:
                    # When the 'concatenate_frames' option is enabled, we need to recompute
                    # the image features for all images
                    if not self._generator:
                        img_features = open_pickle_file(img_save_path)
                else:
                    if 'flip' in imp:
                        imp = imp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        img_data = cv2.imread(imp)
                        show_image(img_data) if debug else None
                        img_features = cv2.resize(img_data, target_dim)
                        if flip_image:
                            img_features = cv2.flip(img_features, 1)
                        #show_image(img_features) if debug else None
                    else:
                        img_data = cv2.imread(imp)
                        #show_image(img_data) if debug else None
                        if flip_image:
                            img_data = cv2.flip(img_data, 1)
                        if crop_type == 'bbox':
                            img_features = crop_bbox(img_data, b, crop_mode, target_dim)
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = img_data[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                            #img_features = img_pad(cropped_image, mode='pad_resize', size=300)
                            show_image(img_features) if debug else None
                        elif 'surround' in crop_type:
                            b_org = list(map(int, b[0:4])).copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, img_data.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            img_features = img_data.copy()
                            img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                            cropped_image = img_features[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            img_features = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                            show_image(img_features) if debug else None
                        elif 'ped_overlays' in crop_type:
                            img_features = TrajectoryOverlays(model_opts).compute_trajectory_overlays(
                                img_data, feature_type,
                                full_bbox_sequences, full_rel_bbox_seq, 
                                full_veh_speed, i)
                            img_features = cv2.resize(img_features, target_dim)
                            show_image(img_features) if debug else None
                        elif 'remove_ped' in crop_type:
                            b_org = list(map(int, b[0:4])).copy()
                            img_features = img_data.copy()
                            img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = 128
                            img_features = cv2.resize(img_features, target_dim)
                            show_image(img_features) if debug else None
                        elif 'keep_ped' in crop_type:
                            img_features = img_data.copy()
                            img_features = cv2.resize(img_features, target_dim)
                            show_image(img_features) if debug else None
                        elif crop_type == 'bbox_resize':
                            img_features = crop_bbox(img_data, b, crop_mode, target_dim, skip_padding=True)
                            img_features = cv2.resize(img_features, target_dim)
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))
                        
                    if 'flow_optical' in feature_type:
                        img_features = process_optical_flow_v2(img_data, target_dim, prev_img_list, feature_type)
                    if 'segmentation' in feature_type:
                        img_features = get_semantic_segmentation(img_features, img_data, target_dim, feature_type, b, crop_mode)
                        print(f"Processing {img_save_path} ...")
                    if preprocess_input is not None:
                        img_features = preprocess_input(img_features)
                    if process:
                        expanded_img = np.expand_dims(img_features, axis=0)
                        img_features = convnet.predict(expanded_img)
                    # Save the file
                    if not os.path.exists(img_save_folder):
                        os.makedirs(img_save_folder)
                    with open(img_save_path, 'wb') as fid:
                        pickle.dump(img_features, fid, pickle.HIGHEST_PROTOCOL)

                # if using the generator save the cached features path and size of the features                                   
                if process and not self._generator:
                    if self._global_pooling == 'max':
                        img_features = np.squeeze(img_features)
                        img_features = np.amax(img_features, axis=0)
                        img_features = np.amax(img_features, axis=0)
                    elif self._global_pooling == 'avg':
                        img_features = np.squeeze(img_features)
                        img_features = np.average(img_features, axis=0)
                        img_features = np.average(img_features, axis=0)
                    else:
                        img_features = img_features.ravel()

                # Some steps to do at the end of each iteration...
                prev_img_list.append(img_data)
                seq_idx = seq_idx + 1

                # Add extracted image to sequence
                if is_feature_static and not concatenate_frames and seq_idx < len(seq):
                    continue # only keep image from last frame in sequence
                if self._generator:
                    img_seq.append(img_save_path)
                    if concatenate_frames:
                        imgs_to_concatenate.append(cropped_image) # Get cropped images before any resize
                else:
                    img_seq.append(img_features)

            # endif - for each image in sequence

            if concatenate_frames:
                img_seq = self.get_concatenated_image(imgs_to_concatenate, img_seq, regen_data, target_dim)
                
            if not store_data_only:
                sequences.append(img_seq)
        if not store_data_only:
            sequences = np.array(sequences)

        # compute size of the features after the processing
        if self._generator:
            with open(sequences[0][0], 'rb') as fid:
                feat_shape = pickle.load(fid).shape
            if process:
                if self._global_pooling in ['max', 'avg']:
                    feat_shape = feat_shape[-1]
                else:
                    feat_shape = np.prod(feat_shape)
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]

        return sequences, feat_shape

        # Processing images anf generate features

    def get_optical_flow(self, img_sequences, bbox_sequences,
                                     ped_ids, save_path,
                                     data_type='train',
                                     crop_type='none',
                                     crop_mode='warp',
                                     crop_resize_ratio=2,
                                     target_dim=(224, 224),
                                     regen_data=False):
        """
        Generate visual feature sequences by reading and processing images
        Args:
            img_sequences: Sequences of image na,es
            bbox_sequences: Sequences of bounding boxes
            ped_ids: Sequences of pedestrian ids
            save_path: Path to the root folder to save features
            data_type: The type of features, train/test/val
            crop_type: The method to crop the images.
            Options are 'none' (no cropping)
                        'bbox' (crop using bounding box coordinates),
                        'context' (A region containing pedestrian and their local surround)
                        'surround' (only the region around the pedestrian. Pedestrian appearance
                                    is suppressed)
            crop_mode: How to resize ond/or pad the corpped images (see utils.img_pad)
            crop_resize_ratio: The ratio by which the image is enlarged to capture the context
                               Used by crop types 'context' and 'surround'.
            target_dim: Dimension of final visual features
            regen_data: Whether regenerate visual features. This will overwrite the cached features
        Returns:
            Numpy array of visual features
            Tuple containing the size of features
        """

        # load the feature files if exists
        print("Generating {} features crop_type={} crop_mode={}\
               \nsave_path={}, ".format(data_type, crop_type, crop_mode, save_path))
        sequences = []
        bbox_seq = bbox_sequences.copy()
        i = -1
        # flow size (h,w)
        flow_size = read_flow_file(img_sequences[0][0].replace('images', 'optical_flow').replace('png', 'flo')).shape
        img_size = cv2.imread(img_sequences[0][0]).shape
        # A ratio to adjust the dimension of bounding boxes (w,h)
        box_resize_coef = (flow_size[1]/img_size[1], flow_size[0]/img_size[0])

        for seq, pid in zip(img_sequences, ped_ids):
            i += 1
            update_progress(i / len(img_sequences))
            flow_seq = []
            for imp, b, p in zip(seq, bbox_seq[i], pid):
                flip_image = False
                set_id = imp.split('/')[-3]
                vid_id = imp.split('/')[-2]
                img_name = imp.split('/')[-1].split('.')[0]
                optflow_save_folder = os.path.join(save_path, set_id, vid_id)
                ofp = imp.replace('images', 'optical_flow').replace('png', 'flo')
                # Modify the path depending on crop mode
                if crop_type == 'none':
                    optflow_save_path = os.path.join(optflow_save_folder, img_name + '.flo')
                else:
                    optflow_save_path = os.path.join(optflow_save_folder, img_name + '_' + p[0] + '.flo')

                # Check whether the file exists
                if os.path.exists(optflow_save_path) and not regen_data:
                    if not self._generator:
                        ofp_data = read_flow_file(optflow_save_path)
                else:
                    if 'flip' in imp:
                        ofp = ofp.replace('_flip', '')
                        flip_image = True
                    if crop_type == 'none':
                        ofp_image = read_flow_file(ofp)
                        ofp_data = cv2.resize(ofp_image, target_dim)
                        if flip_image:
                            ofp_data = cv2.flip(ofp_data, 1)
                    else:
                        ofp_image = read_flow_file(ofp)
                        # Adjust the size of bbox according to the dimensions of flow map
                        b = list(map(int, [b[0] * box_resize_coef[0], b[1] * box_resize_coef[1],
                                           b[2] * box_resize_coef[0], b[3] * box_resize_coef[1]]))
                        if flip_image:
                            ofp_image = cv2.flip(ofp_image, 1)
                        if crop_type == 'bbox':
                            cropped_image = ofp_image[b[1]:b[3], b[0]:b[2], :]
                            ofp_data = img_pad(cropped_image, mode=crop_mode, size=target_dim[0])
                        elif 'context' in crop_type:
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, ofp_image.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            cropped_image = ofp_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            ofp_data = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        elif 'surround' in crop_type:
                            b_org = b.copy()
                            bbox = jitter_bbox(imp, [b], 'enlarge', crop_resize_ratio)[0]
                            bbox = squarify(bbox, 1, ofp_image.shape[1])
                            bbox = list(map(int, bbox[0:4]))
                            ofp_image[b_org[1]:b_org[3], b_org[0]: b_org[2], :] = 0
                            cropped_image = ofp_image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                            ofp_data = img_pad(cropped_image, mode='pad_resize', size=target_dim[0])
                        else:
                            raise ValueError('ERROR: Undefined value for crop_type {}!'.format(crop_type))

                    # Save the file
                    if not os.path.exists(optflow_save_folder):
                        os.makedirs(optflow_save_folder)
                    write_flow(ofp_data, optflow_save_path)

                # if using the generator save the cached features path and size of the features
                if self._generator:
                    flow_seq.append(optflow_save_path)
                else:
                    flow_seq.append(ofp_data)
            sequences.append(flow_seq)
        sequences = np.array(sequences)
        # compute size of the features after the processing
        if self._generator:
            feat_shape = read_flow_file(sequences[0][0]).shape
            if not isinstance(feat_shape, tuple):
                feat_shape = (feat_shape,)
            feat_shape = (np.array(bbox_sequences).shape[1],) + feat_shape
        else:
            feat_shape = sequences.shape[1:]
        return sequences, feat_shape

    def get_data_sequence(self, data_type, data_raw, opts):
        """
        Generates raw sequences from a given dataset
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            opts:  Options for generating data samples
        Returns:
            A list of data samples extracted from raw data
            Positive and negative data counts
        """
        print('\n#####################################')
        print('Generating raw data')
        print('#####################################')
        d = {'center': data_raw['center'].copy(),
             'box': data_raw['bbox'].copy(),
             'ped_id': data_raw['pid'].copy(),
             'crossing': data_raw.get('activities', []).copy(),
             'image': data_raw['image'].copy()}

        balance = opts['balance_data'] if data_type == 'train' else False
        obs_length = opts['obs_length']
        time_to_event = opts['time_to_event']
        normalize = opts['normalize_boxes']

        try:
            d['speed'] = data_raw['obd_speed'].copy()
        except KeyError:
            d['speed'] = data_raw['vehicle_act'].copy()
            print('Jaad dataset does not have speed information')
            print('Vehicle actions are used instead')
        if balance:
            self.balance_data_samples(d, data_raw['image_dimension'][0])
        d['box_org'] = d['box'].copy()
        d['speed_org'] = d['speed'].copy()
        d['tte'] = []
        d['tte_pos'] = []
        if opts.get("seq_type") == "trajectory":
            d["trajectories"] = []

        if isinstance(time_to_event, int):
            for k in d.keys():
                for i in range(len(d[k])):
                    d[k][i] = d[k][i][- obs_length - time_to_event:-time_to_event]
            d['tte'] = [[time_to_event]]*len(data_raw['bbox'])
        else:
            overlap = opts['overlap'] # if data_type == 'train' else 0.0
            olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
            olap_res = 1 if olap_res < 1 else olap_res
            
            # this is where sequences (t=16) are created
            d = compute_sequences(d, data_raw, opts, obs_length, time_to_event, olap_res,
                                  action_predict_obj_ref=self)
            
            test = 10
        if normalize:
            d = self.normalize_sequence_data(d, opts)
        else:
            for k in d.keys():
                d[k] = np.array(d[k])

        if opts["seq_type"] == "crossing":
            d['crossing'] = np.array(d['crossing'])[:, 0, :]
            pos_count = np.count_nonzero(d['crossing'])
            neg_count = len(d['crossing']) - pos_count
        else:
            pos_count, neg_count = 0, 0
        print("Negative {} and positive {} sample counts".format(neg_count, pos_count))

        ts_config = opts.get("trajectory_subsequence")
        if ts_config and ts_config.get("enabled"):
            get_subsequence_from_sequence(d, strategy=ts_config["strategy"], 
                                          n=ts_config["n"])
        return d, neg_count, pos_count
    
    def normalize_sequence_data(self, d, opts,
                                normalization_type="relative_subtract"):
        """ Args:
        normalization_type [str]: can be any of:
            "relative_subtract": subtract first position in sequence from 
                                 all sequence elements
            "divide_by_img_dims": divide positions by image dimensions
        """
        for k in d.keys():
            self.compute_positions(d, k, opts, normalization_type)
        if "abs_box" in opts["obs_input_type"]:
            self.compute_positions(
                d, "abs_box", opts, normalization_type)
        if "normalized_abs_box" in opts["obs_input_type"]:
            self.compute_positions(
                d, "normalized_abs_box", opts, normalization_type)
        if "box_speed" in opts["obs_input_type"]:
            self.compute_positions(
                d, "box_speed", opts, normalization_type)
        if "box_center_speed" in opts["obs_input_type"]:
            self.compute_positions(
                d, "box_center_speed", opts, normalization_type)
        if "box_center_acceleration" in opts["obs_input_type"]:
            self.compute_positions(
                d, "box_center_acceleration", opts, normalization_type)
        if "box_height" in opts["obs_input_type"]:
            self.compute_positions(
                d, "box_height", opts, normalization_type)
        if "veh_speed" in opts["obs_input_type"]:
            self.compute_veh_speed(d, "veh_speed", opts["dataset"])
        if "veh_acceleration" in opts["obs_input_type"]:
            self.compute_positions(
                d, "veh_acceleration", opts, normalization_type)
        for k in d.keys():
            d[k] = np.array(d[k])
        return d
    
    def compute_positions(self, d, k, opts, normalization_type,
                          compute_tte_pos=False, 
                          add_normalized_abs_box=False, 
                          add_box_center_speed=False,
                          add_pose=False, 
                          debug=False):
        #is_key_std_box = (k == 'box' or k == 'box_org')
        if (('box' not in k and 'veh' not in k and "tte" not in k and "trajectories" not in k) \
            or k=='box_org') and k != 'center':
            for i in range(len(d[k])):
                d[k][i] = d[k][i][1:]
        #elif k == 'box_org':
        #    return
        else:
            # len(d[k]) is the number of tracks
            if k == 'tte' or k == 'normalized_abs_box_org' or k == 'box_center_speed_org':
                return
            elif k == 'tte_pos':
                if not compute_tte_pos:
                    return
                for i in range(len(d[k])):
                    d[k][i] = np.subtract(d[k][i][0:], d["box_org"][i][-1]).tolist()
                    d = self.compute_positions_normalization_divide_by_img_dims(d, k, opts)
            elif k == 'trajectories':
                """
                d = self.compute_positions_normalization_divide_by_img_dims(d, k, opts,
                                                                            replace_by_patch_id=False)
                """
                for i in range(len(d[k])):
                    box_rel_coords = [el[:4] for el in d[k][i][1:]]
                    speed_vals = [[el[-1]] for el in d[k][i][1:]]

                    original_trajectory = d[k][i][1:]

                    # subtract origin coord from box_rel_coords
                    d[k][i] = np.subtract(box_rel_coords, d["box_org"][i][0]).tolist()

                    if add_normalized_abs_box:
                        normalized_abs_box = [[el[4:8]] for el in original_trajectory]
                        # normalized_abs_box = [[el] for el in original_trajectory]
                        # d[k][i] = [c + normalized_abs_box[idx][0] + speed_vals[idx] for idx, c in enumerate(d[k][i])]
                        d[k][i] = [normalized_abs_box[idx][0] for idx, c in enumerate(d[k][i])]
                        test = 10
                    elif add_box_center_speed:
                        box_center_speed = [[el[4:6]] for el in original_trajectory]
                        # d[k][i] = [box_center_speed[idx][0] for idx, c in enumerate(d[k][i])]
                        d[k][i] = [c + box_center_speed[idx][0] + speed_vals[idx] for idx, c in enumerate(d[k][i])]
                    elif add_pose:
                        #pose_keypoints = [el[4:14] for el in original_trajectory]
                        pose_keypoints = [el for el in original_trajectory]
                        d[k][i] = [c + pose_keypoints[idx] + speed_vals[idx] for idx, c in enumerate(d[k][i])]
                        #d[k][i] = [pose_keypoints[idx] for idx, c in enumerate(d[k][i])]
                        test = 10
                    else:
                        d[k][i] = np.subtract(box_rel_coords, d["box_org"][i][0]).tolist()
                        d[k][i] = [c + speed_vals[idx] for idx, c in enumerate(d[k][i])]
            elif (k == 'box' and normalization_type == "relative_subtract") \
                or k == 'center':
                for i in range(len(d[k])):
                    d[k][i] = np.subtract(d[k][i][1:], d[k][i][0]).tolist()
            elif (k == 'box' and normalization_type == "divide_by_img_dims"):
                d = self.compute_positions_normalization_divide_by_img_dims(
                    d, k, opts)
            elif k == 'abs_box':
                d['abs_box'] = copy.deepcopy(d['box_org'])
            elif k == 'box_height':
                d['box_height'] = copy.deepcopy(d['box_org'])
                d = self.compute_box_height(d, k)
                d = self.compute_positions_normalization_divide_by_img_dims(d, k, opts)
            elif k == 'normalized_abs_box':
                d['normalized_abs_box'] = copy.deepcopy(d['box_org'])
                d = self.compute_positions_normalization_divide_by_img_dims(d, k, opts,
                                                                            replace_by_patch_id=False)
            elif k == 'box_speed':
                d['box_speed'] = copy.deepcopy(d['box_org'])
                d = self.compute_box_speed(d, k)
            elif k == 'box_center_speed':
                d['box_center_speed'] = copy.deepcopy(d['center'])
                d = self.compute_box_speed(d, k)
            elif k == 'box_center_acceleration':
                if not 'box_center_speed' in d:
                    raise Exception
                d['box_center_acceleration'] = copy.deepcopy(d['center'])
                d = self.compute_box_speed(d, k)
            elif k == 'veh_acceleration':
                if not 'veh_speed' in d:
                    raise Exception
                d['veh_acceleration'] = copy.deepcopy(d['veh_speed'])
                d = self.compute_box_speed(d, k, base_k='veh_speed')
            else:
                raise Exception
        
    def compute_positions_normalization_divide_by_img_dims(self, d, k, opts, shortcut=True,
                                                           second_step_normalize=True,
                                                           replace_by_patch_id=False):
        feature_folder, _ = get_path(save_folder=f"box_{k}",
                                     dataset=opts['dataset'],
                                     save_root_folder='data/features')
        for i in range(len(d[k])):
            if shortcut:
                img_shape = (1080, 1920)
                d = self._divide_sequence_by_img_dims(img_shape, d, k, i,
                                                      replace_by_patch_id=replace_by_patch_id,
                                                      second_step_normalize=second_step_normalize)
            else:
                img_path = d["image"][i][0]
                ped_id = d["ped_id"][i][0][0]
                feat_file_name = "/".join(img_path.rsplit("/", 2)[1:]).replace(".png", "")
                feat_file_name = f"{feat_file_name}_{ped_id}.pkl"
                feat_path = os.path.join(feature_folder, feat_file_name)
                file_already_exists = os.path.exists(feat_path)
                if file_already_exists:
                    feat_data = open_pickle_file(feat_path)
                    d[k][i] = feat_data
                else:
                    img_data = cv2.imread(img_path)
                    img_shape = img_data.shape
                    if img_shape[0] != 1080 or img_shape[1] != 1920:
                        test = 10
                    d = self._divide_sequence_by_img_dims(img_shape, d, k, i)
                    save_data_in_pkl(feat_path.rsplit("/", 1)[0], 
                                        feat_path, d[k][i])
        return d
    
    def _divide_sequence_by_img_dims(self, img_shape, d, k, i,
                                     second_step_normalize=False,
                                     replace_by_patch_id=False):
        if k == "box_height":
            for seq_idx, seq_el in enumerate(d[k][i]):
                position = seq_el.copy()
                position = position[0] / img_shape[0] if second_step_normalize else position[0]# x
                d[k][i][seq_idx] = [position]
        else:
            # For all other data types...
            for seq_idx, seq_el in enumerate(d[k][i]):
                position = seq_el.copy()
                if second_step_normalize:
                    position[0] = position[0] / img_shape[1] # y
                    position[1] = position[1] / img_shape[0] # x
                    position[2] = position[2] / img_shape[1] # y
                    position[3] = position[3] / img_shape[0] # x
                d[k][i][seq_idx] = position
                #if not all(i<=1.0 for i in position):
                #    raise Exception
                if k == "normalized_abs_box" and replace_by_patch_id:
                    dim = 4
                    center_point = [(position[1]+position[3])/2, (position[0]+position[2])/2] # [x, y]
                    row = math.ceil(dim * center_point[0])
                    column = math.ceil(dim * center_point[1])
                    patch_id = (row-1)*dim + (column-1)
                    d[k][i][seq_idx] = [patch_id / (dim*dim)] 
        if k not in ["normalized_abs_box", "normalized_abs_box_org", 
                     "box_center_speed_org", "box_height", "tte_pos"]: # todo, by default the sequence length should be kept the same
            d[k][i] = d[k][i][1:] # leave out first element in sequence
        return d
    
    def compute_box_speed(self, d, k, base_k=None):
        if not base_k:
            if k == "box_center_speed":
                base_k = "center"
            elif k == "box_center_acceleration":
                base_k = "box_center_speed"
            elif k == "box_speed":
                base_k = "box_org"
            else:
                base_k = "box"
        for i in range(len(d[k])):
            for seq_idx, seq_el in enumerate(d[k][i]):
                if seq_idx == 0:
                    # forward looking speed
                    d[k][i][seq_idx] = np.subtract(d[base_k][i][seq_idx+1],
                                                   d[base_k][i][seq_idx]).tolist()
                elif seq_idx == 1:
                    d[k][i][seq_idx] = np.subtract(d[base_k][i][seq_idx],
                                                   d[base_k][i][seq_idx-1]).tolist()
                elif seq_idx >= 2:
                    # estimation of speed with second order polynomial
                    fi = np.multiply(3, d[base_k][i][seq_idx])
                    fi_minus1 = np.multiply(4, d[base_k][i][seq_idx-1])
                    fi_minus2 = d[base_k][i][seq_idx-2]
                    num = np.add(np.subtract(fi, fi_minus1), fi_minus2)
                    d[k][i][seq_idx] = np.divide(num, 2).tolist()
        return d
    
    def compute_box_height(self, d, k):
        for i in range(len(d[k])):
            for seq_idx, seq_el in enumerate(d[k][i]):
                # d[k][i][seq_idx] = [seq_el[2] - seq_el[0]]
                d[k][i][seq_idx] = [seq_el[3] - seq_el[1]]
                test = 10
        return d
    
    def compute_pose(self, d, k, model_opts):
        save_path, _ = get_path(save_folder='poses',
                                dataset=model_opts["dataset"],
                                save_root_folder='data/features')
        for i in tqdm(range(len(d[k]))):
            for seq_idx, seq_el in enumerate(d[k][i]):
                imp = d["image"][i][seq_idx]
                b = d["box"][i][seq_idx]
                p = d["ped_id"][i][seq_idx]
                _, pose_features = create_pose_from_img_path(
                    imp, b, p, None, None, model_opts, save_path)
                d[k][i][seq_idx] = pose_features
        return d
    
    def compute_veh_speed(self, d, k, dataset):
        """ After swapping encodings, new values should be:
            0: stopped
            1: decelerating
            2: moving slow
            3: moving fast
            4: accelerating
        """
        
        def swap_speed_encodings(seq, a, b):
            seq[seq == a] = -1
            seq[seq == b] = a
            seq[seq == -1] = b
            return seq

        d['veh_speed'] = copy.deepcopy(d['speed'])
        if dataset in ["jaad_all", "jaad_beh"]:
            for i in range(len(d[k])):
                seq = np.asarray(d[k][i])
                seq = swap_speed_encodings(seq, 1, 2)
                seq = swap_speed_encodings(seq, 1, 3)
                d[k][i] = seq.tolist()
        return d

    def balance_data_samples(self, d, img_width, balance_tag='crossing'):
        """
        Balances the ratio of positive and negative data samples. The less represented
        data type is augmented by flipping the sequences
        Args:
            d: Sequence of data samples
            img_width: Width of the images
            balance_tag: The tag to balance the data based on
        """
        print("Balancing with respect to {} tag".format(balance_tag))
        gt_labels = [gt[0] for gt in d[balance_tag]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                gt_augment = 1
            else:
                gt_augment = 0

            num_samples = len(d[balance_tag])
            for i in range(num_samples):
                if d[balance_tag][i][0][0] == gt_augment:
                    for k in d:
                        if k == 'center':
                            flipped = d[k][i].copy()
                            flipped = [[img_width - c[0], c[1]]
                                       for c in flipped]
                            d[k].append(flipped)
                        if k == 'box':
                            flipped = d[k][i].copy()
                            flipped = [np.array([img_width - b[2], b[1], img_width - b[0], b[3]])
                                       for b in flipped]
                            d[k].append(flipped)
                        if k == 'image':
                            flipped = d[k][i].copy()
                            flipped = [im.replace('.png', '_flip.png') for im in flipped]
                            d[k].append(flipped)
                        if k in ['speed', 'ped_id', 'crossing', 'walking', 'looking']:
                            d[k].append(d[k][i].copy())

            gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(gt_labels))
            num_neg_samples = len(gt_labels) - num_pos_samples
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(42)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]

            # update the data
            for k in d:
                seq_data_k = d[k]
                d[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in d[balance_tag]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(d[balance_tag]) - num_pos_samples))

    def _get_aux_name(self, model_opts):
        process = model_opts.get('process', True)
        aux_name = [self._backbone]
        if not process:
            aux_name.append('raw')
        aux_name = '_'.join(aux_name).strip('_')
        return aux_name
    
    def get_context_data(self, model_opts, data, data_type, feature_type):
        print('\n#####################################')
        print('Generating {} {}'.format(feature_type, data_type))
        print('#####################################')
        process = model_opts.get('process', True)
        aux_name = self._get_aux_name(model_opts)
        eratio = model_opts['enlarge_ratio']
        dataset = model_opts['dataset']

        data_gen_params = {'data_type': data_type, 'crop_type': 'none',
                           'target_dim': model_opts.get('target_dim', (224, 224))}
        if 'local_box' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif 'local_context' in feature_type:
            data_gen_params['crop_type'] = 'context'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'surround' in feature_type:
            data_gen_params['crop_type'] = 'surround'
            data_gen_params['crop_resize_ratio'] = eratio
        elif 'with_ped_overlays' in feature_type: # added
            data_gen_params['crop_type'] = 'ped_overlays'
        elif 'with_ped' in feature_type: # added:
            data_gen_params['crop_type'] = 'keep_ped'
        elif 'scene_context' in feature_type and 'segmentation' not in feature_type: # added
            data_gen_params['crop_type'] = 'remove_ped'
        elif 'ped_segmentation' in feature_type:
            data_gen_params['crop_type'] = 'bbox'
            data_gen_params['crop_mode'] = 'pad_resize'
        elif 'bbox' in feature_type:
            data_gen_params['crop_type'] = 'bbox_resize'
        save_folder_name = feature_type
        if 'optical_flow' not in feature_type:
            save_folder_name = '_'.join([feature_type, aux_name])
            if 'local_context' in feature_type or 'surround' in feature_type:
                save_folder_name = '_'.join([save_folder_name, str(eratio)])
        data_gen_params['save_path'], _ = get_path(save_folder=save_folder_name,
                                                   dataset=dataset, save_root_folder='data/features')
        # Get context data
        if 'scene_context' in feature_type and feature_type != "scene_context_non_static":
            return get_static_context_data(
                self, model_opts, data, 
                data_gen_params, feature_type
            )
        if 'optical_flow' in feature_type:
            return self.get_optical_flow(data['image'],
                                         data['box_org'],
                                         data['ped_id'],
                                         **data_gen_params)
        else:
            return self.load_images_crop_and_process(data['image'],
                                                     data['box_org'],
                                                     data['ped_id'],
                                                     process=process,
                                                     feature_type=feature_type,
                                                     **data_gen_params)

    def get_data(self, data_type, data_raw, model_opts,
                 combined_model=False):
        """
        Generates data train/test/val data
        Args:
            data_type: Split type of data, whether it is train, test or val
            data_raw: Raw tracks from the dataset
            model_opts: Model options for generating data
        Returns:
            A dictionary containing, data, data parameters used for model generation,
            effective dimension of data (the number of rgb images to be used calculated accorfing
            to the length of optical flow window) and negative and positive sample counts
        """

        self._generator = model_opts.get('generator', False)
        data_type_sizes_dict = {}
        process = model_opts.get('process', True)
        dataset = model_opts['dataset']
        data, neg_count, pos_count = self.get_data_sequence(data_type, data_raw, model_opts)

        data_type_sizes_dict['box'] = data['box'].shape[1:]
        if 'speed' in data.keys():
            data_type_sizes_dict['speed'] = data['speed'].shape[1:]

        # Store the type and size of each image
        _data = []
        data_sizes = []
        data_types = []

        for d_type in model_opts['obs_input_type']:
            if 'local' in d_type or 'context' in d_type or 'bbox' in d_type:
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
                test = 10
            elif 'segmentation' in d_type: # todo: verify if this gets called
                features, feat_shape = self.get_context_data(model_opts, data, data_type, d_type)
            elif 'pose' in d_type:
                path_to_pose, _ = get_path(save_folder='poses',
                                           dataset=dataset,
                                           save_root_folder='data/features')
                features, _ = get_pose(model_opts,
                                    data['image'],
                                    data['box_org'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=path_to_pose,
                                    dataset=model_opts['dataset'])
                feat_shape = features.shape[1:]
            elif 'segm_map_features' in d_type:
                raise Exception("Not implemented in this repo")
            elif 'segm_map_pixels' in d_type:
                raise Exception("Not implemented in this repo")
            elif 'segm_map_occurences' in d_type:
                raise Exception("Not implemented in this repo")
            elif d_type == 'scene_graph':
                raise Exception("Not implemented in this repo")
            elif d_type == 'scene_graph_previous':
                raise Exception("Not implemented in this repo")
            else:
                features = data[d_type]
                feat_shape = features.shape[1:]

            # Get feature descriptors
            if "ped_scene_segmentation" in d_type or "hog_descriptor" in d_type:
                if d_type == "ped_scene_segmentation":
                    descriptor_type = "ped_shape_descriptor"
                elif d_type == "bbox_hog_descriptor":
                    descriptor_type = "hog_descriptor"
                file_path, _ = get_path(save_folder=descriptor_type,
                                        dataset=dataset,
                                        save_root_folder='data/features')
                # Despite the function name, here we're getting pedestrian
                # shape descriptors (and not pose)
                features, feat_shape = get_pose(model_opts,
                                    features,
                                    data['box_org'],
                                    data['ped_id'],
                                    data_type=data_type,
                                    file_path=file_path,
                                    dataset=model_opts['dataset'],
                                    descriptor_type=descriptor_type,
                                    prev_features=_data,
                                    do_not_keep_in_memory=True)
                #feat_shape = features.shape[1:]

            _data.append(features)
            data_sizes.append(feat_shape)
            data_types.append(d_type)

        # Reshape input if specified in configs
        if "process_input_features" in model_opts and model_opts["process_input_features"]["enabled"]:
            _data, data_sizes = self._process_input_features(_data, data_sizes, model_opts)
        
        # create the final data file to be returned
        if self._generator:
            _data = get_generator(_data=_data,
                                    data=data,
                                    data_sizes=data_sizes,
                                    process=process,
                                    global_pooling=self._global_pooling,
                                    model_opts=model_opts,
                                    data_type=data_type,
                                    combined_model=combined_model
                    )
        else:
            _data = (_data, data['crossing'])

        return {'data': _data,
                'ped_id': data['ped_id'],
                'image': data['image'],
                'tte': data['tte'],
                'data_params': {'data_types': data_types, 'data_sizes': data_sizes},
                'count': {'neg_count': neg_count, 'pos_count': pos_count}}

    def _process_input_features(self, data, data_sizes, model_opts):

        def _stack_features(data, data_sizes, indexes_to_stack):

            if not indexes_to_stack:
                return data, data_sizes

            # Determine input shape when "index_to_use_for_input_shape" is set
            if "process_input_features" in model_opts and model_opts["process_input_features"].get("index_to_use_for_input_shape"):
                feat_size_index = model_opts["process_input_features"]["index_to_use_for_input_shape"]
                special_input_shape = data_sizes[feat_size_index][-1]

            # Stack data sizes
            nb_stacked_features = 0
            for i in indexes_to_stack:
                nb_stacked_features += data_sizes[i][-1]
            
            # Stack data itself
            _data = [data[i] for i in indexes_to_stack]
            # This could obviously be written in a better way, but I'm not
            # sure how to go about it
            if len(_data) == 1:
                _data = np.c_[_data[0]]
            elif len(_data) == 2:
                _data = np.c_[_data[0], _data[1]]
            elif len(_data) == 3:
                _data = np.c_[_data[0], _data[1], _data[2]]
            elif len(_data) == 4:
                _data = np.c_[_data[0], _data[1], _data[2], _data[3]]
            elif len(_data) == 5:
                _data = np.c_[_data[0], _data[1], _data[2], _data[3], _data[4]]
            elif len(_data) == 6:
                _data = np.c_[_data[0], _data[1], _data[2], _data[3], _data[4], _data[5]]
            elif len(_data) == 7:
                _data = np.c_[_data[0], _data[1], _data[2], _data[3], _data[4], _data[5], _data[6]]
            else:
                raise Exception(f"Number of observation input types ({len(_data)}) is not supported")
            
            # todo: revisit
            sequence_len = data_sizes[-1][0]
            for index in sorted(indexes_to_stack, reverse=True):
                del data[index]
                del data_sizes[index]
            data.append(_data)
            
            # Determine data_sizes
            if "process_input_features" in model_opts and model_opts["process_input_features"].get("index_to_use_for_input_shape"):
                data_sizes.append((sequence_len, special_input_shape))
            else:    
                data_sizes.append((sequence_len, nb_stacked_features))
            
            return data, data_sizes
        
        def _convert_features_to_static(data, data_sizes, indexes_to_convert_to_static):
            for idx in indexes_to_convert_to_static:
                static_data = list(data[idx])
                for row_idx in range(len(static_data)):
                    static_data[row_idx] = [static_data[row_idx][-1]] # keep last frame
                data[idx] = np.array(static_data)
                
                # Update feature size
                feature_size = list(data_sizes[idx])
                data_sizes[idx] = tuple(feature_size[1:]) 
                
            return data, data_sizes

        """
        # When data contains image paths, expand the last dimension
        # to make each data type the same shape
        for i in range(len(_data)):
            if len(_data[i].shape) == 2:
                _data[i] = np.expand_dims(_data[i], -1)
        """

        # Stack some features into one feature
        if model_opts["process_input_features"].get("indexes_to_stack"):
            indexes_to_stack = model_opts["process_input_features"]["indexes_to_stack"]
        else:
            #indexes_to_stack = list(range(len(data))) # stack all features
            indexes_to_stack = []
        data, data_sizes = _stack_features(data, data_sizes, indexes_to_stack)

        # Change data for some features to static (keep last frame)
        if model_opts["process_input_features"].get("static_features_indexes"):
            indexes_to_convert_to_static = model_opts["process_input_features"]["static_features_indexes"] 
            data, data_sizes = _convert_features_to_static(data, data_sizes, indexes_to_convert_to_static)

        return data, data_sizes
        

    def log_configs(self, config_path, batch_size, epochs,
                    lr, model_opts):

        # TODO: Update config by adding network attributes
        """
        Logs the parameters of the model and training
        Args:
            config_path: The path to save the file
            batch_size: Batch size of training
            epochs: Number of epochs for training
            lr: Learning rate of training
            model_opts: Data generation parameters (see get_data)
        """
        # Save config and training param files
        with open(config_path, 'wt') as fid:
            yaml.dump({'model_opts': model_opts, 
                       'train_opts': {'batch_size':batch_size, 'epochs': epochs, 'lr': lr}},
                       fid, default_flow_style=False)
        # with open(config_path, 'wt') as fid:
        #     fid.write("####### Model options #######\n")
        #     for k in opts:
        #         fid.write("%s: %s\n" % (k, str(opts[k])))

        #     fid.write("\n####### Network config #######\n")
        #     # fid.write("%s: %s\n" % ('hidden_units', str(self._num_hidden_units)))
        #     # fid.write("%s: %s\n" % ('reg_value ', str(self._regularizer_value)))

        #     fid.write("\n####### Training config #######\n")
        #     fid.write("%s: %s\n" % ('batch_size', str(batch_size)))
        #     fid.write("%s: %s\n" % ('epochs', str(epochs)))
        #     fid.write("%s: %s\n" % ('lr', str(lr)))

        print('Wrote configs to {}'.format(config_path))

    def class_weights(self, apply_weights, sample_count,
                      huggingface=False):
        """
        Computes class weights for imbalanced data used during training
        Args:
            apply_weights: Whether to apply weights
            sample_count: Positive and negative sample counts
        Returns:
            A dictionary of class weights or None if no weights to be calculated
        """

        total = sample_count['neg_count'] + sample_count['pos_count']
        if not apply_weights or not total:
            return None
        # formula from sklearn
        #neg_weight = (1 / sample_count['neg_count']) * (total) / 2.0
        #pos_weight = (1 / sample_count['pos_count']) * (total) / 2.0
        
        # use simple ratio
        neg_weight = sample_count['pos_count']/total
        pos_weight = sample_count['neg_count']/total

        print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
        if huggingface:
            return [neg_weight, pos_weight]
        return {0: neg_weight, 1: pos_weight}

    def get_callbacks(self, learning_scheduler, model_path):
        """
        Creates a list of callbacks for training
        Args:
            learning_scheduler: Whether to use callbacks
        Returns:
            A list of call backs or None if learning_scheduler is false
        """
        callbacks = None
        metric_for_choosing_best_model = "val_f1_score" # 'val_loss'

        # Set up learning schedulers
        if learning_scheduler:
            callbacks = []
            if 'early_stop' in learning_scheduler:
                default_params = {'monitor': metric_for_choosing_best_model,
                                  'min_delta': 0.0, 'patience': 5,
                                  'verbose': 1}
                default_params.update(learning_scheduler['early_stop'])
                callbacks.append(EarlyStopping(**default_params))

            if 'plateau' in learning_scheduler:
                default_params = {'monitor': metric_for_choosing_best_model,
                                  'factor': 0.2, 'patience': 5,
                                  'min_lr': 1e-08, 'verbose': 1}
                default_params.update(learning_scheduler['plateau'])
                callbacks.append(ReduceLROnPlateau(**default_params))

            if 'checkpoint' in learning_scheduler:
                default_params = {'filepath': model_path, 'monitor': metric_for_choosing_best_model,
                                  'save_best_only': True, 'save_weights_only': False,
                                  'save_freq': 'epoch', 'verbose': 2}
                if learning_scheduler['checkpoint']:
                    default_params.update(learning_scheduler['checkpoint'])
                callbacks.append(ModelCheckpoint(**default_params))

        return callbacks

    def get_optimizer(self, optimizer):
        """
        Return an optimizer object
        Args:
            optimizer: The type of optimizer. Supports 'adam', 'sgd', 'rmsprop'
        Returns:
            An optimizer object
        """
        assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
        "{} optimizer is not implemented".format(optimizer)
        if optimizer.lower() == 'adam':
            return Adam
        elif optimizer.lower() == 'sgd':
            return SGD
        elif optimizer.lower() == 'rmsprop':
            return RMSprop
        
    def get_huggingface_model(self, model_opts):
        model_str = model_opts["model"]
        model_trainers_module = getattr(getattr(
            __import__(f"models.hugging_face.model_trainers.{model_str.lower()}"),
            "hugging_face"), "model_trainers")
        model_module = getattr(
            model_trainers_module,
            model_str.lower())
        model_class = getattr(model_module, model_str)
        return model_class()
    
    def train(self, data_train,
              data_val=None,
              batch_size=32,
              epochs=60,
              lr=0.000005,
              optimizer='adam',
              learning_scheduler=None,
              model_opts=None,
              is_huggingface=False,
              free_memory=False,
              train_opts=None,
              hyperparams=None,
              test_only=False):
        """
        Trains the models
        Args:
            data_train: Training data
            data_val: Validation data
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            optimizer: Optimizer for training
            learning_scheduler: Whether to use learning schedulers
            model_opts: Model options
        Returns:
            The path to the root folder of models
        """
        learning_scheduler = learning_scheduler or {}
        # Set the path for saving models
        model_folder_name = time.strftime("%d%b%Y-%Hh%Mm%Ss")
        path_params = {'save_folder': os.path.join(self.__class__.__name__, model_folder_name),
                       'save_root_folder': 'data/models/',
                       'dataset': model_opts['dataset']}
        model_path, hg_model_path = get_path(**path_params, file_name='model.h5')


        # Read train data
        data_train = self.get_data('train', data_train, {**model_opts, 'batch_size': batch_size}) 

        if data_val is not None:
            data_val = self.get_data('val', data_val, {**model_opts, 'batch_size': batch_size})['data']
            if self._generator:
                data_val = data_val[0]

        # Use custom training functions for some models
        # Todo: use design pattern to abstract training functions no matter what framework we're using (i.e. Pytorch or Tensorflow)
        if is_huggingface:
            dataset_statistics = get_dataset_statistics(data_train, model_opts)
            model = self.get_huggingface_model(model_opts)
            class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'],
                                         huggingface=True)
            trainer = model.train(
                    data_train, data_val, batch_size, epochs, hg_model_path,
                    generator=self._generator, 
                    free_memory=free_memory,
                    dataset_statistics=dataset_statistics,
                    model_opts=model_opts,
                    train_opts=train_opts,
                    hyperparams=hyperparams,
                    class_w=class_w,
                    test_only=test_only)
              
            if free_memory: # to be tested
                free_train_and_val_memory(data_train, data_val)

            history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
            trainer["saved_files_path"] = saved_files_path

            return trainer

        # If model uses the tensorflow framework...
        if not self.is_tensorflow:
            raise Exception("The appropriate training framework is not specified in configs")

        with self.multi_gpu_strategy.scope():
            train_model, class_w, optimizer, f1_metric = \
                self._get_model_and_optimizer(data_train, model_opts, lr, optimizer)
       
        # Train the model
        train_model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', f1_metric])

        callbacks = self.get_callbacks(learning_scheduler, model_path)
        
        history = train_model.fit(x=data_train['data'][0],
                                  y=None if self._generator else data_train['data'][1],
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  validation_data=data_val,
                                  class_weight=class_w,
                                  verbose=1,
                                  callbacks=callbacks)
        if 'checkpoint' not in learning_scheduler:
            print('Train model is saved to {}'.format(model_path))
            train_model.save(model_path)

        # Save data options and configurations
        model_opts_path, _ = get_path(**path_params, file_name='model_opts.pkl')
        with open(model_opts_path, 'wb') as fid:
            pickle.dump(model_opts, fid, pickle.HIGHEST_PROTOCOL)

        config_path, _ = get_path(**path_params, file_name='configs.yaml')
        self.log_configs(config_path, batch_size, epochs,
                         lr, model_opts)

        # Save training history
        history_path, saved_files_path = get_path(**path_params, file_name='history.pkl')
        with open(history_path, 'wb') as fid:
            pickle.dump(history.history, fid, pickle.HIGHEST_PROTOCOL)

        return saved_files_path

    def _get_model_and_optimizer(self, data_train, model_opts, lr, optimizer):
        
        train_model = self.get_model(data_train['data_params'], model_opts=model_opts, data=data_train)
        class_w = self.class_weights(model_opts['apply_class_weights'], data_train['count'])
        """
        except Exception as e:
            # If an exception is caught, try to get model for a "combined" model
            train_model = self.get_model(data_train, model_opts=model_opts)
            count = data_train[list(data_train.keys())[0]]['count']
            class_w = self.class_weights(model_opts['apply_class_weights'], count)
        """
            
        optimizer = self.get_optimizer(optimizer)(lr=lr)

        # Get metrics
        f1_metric = tfa.metrics.F1Score(num_classes=1, average='macro', threshold=0.5)
        #auc_metric = tf.keras.metrics.AUC()

        return train_model, class_w, optimizer, f1_metric


    # Test Functions
    def test(self, data_test, model_path='', 
             debug=False, 
             is_huggingface=False, 
             training_result=None,
             model_opts=None,
             test_only=None):
        """
        Evaluates a given model
        Args:
            data_test: Test data
            model_path: Path to folder containing the model and options
            save_results: Save output of the model for visualization and analysis
            model_opts: options related to the model, only needed when it is a huggingface model
            test_only [bool]: is set to True, model will not be trained, only tested
        Returns:
            Evaluation metrics
        """

        if is_huggingface:
            complete_data = self.get_data('test', data_test, {**model_opts, 'batch_size': 1})
            test_data = complete_data["data"]
            model = self.get_huggingface_model(model_opts)
            return model.test(test_data, training_result, model_path, 
                              generator=self._generator, 
                              complete_data=complete_data,
                              dataset_name=model_opts["dataset_full"],
                              test_only=test_only)
            
        with open(os.path.join(model_path, 'configs.yaml'), 'r') as fid:
            opts = yaml.safe_load(fid)

        custom_objects = {
            "T2V": T2V,
            "CustomAttention": CustomAttention
        }
        with custom_object_scope(custom_objects):
            test_model = load_model(os.path.join(model_path, 'model.h5'))
        """
        if debug:
            model_dict = str(test_model.__dict__)
            with open(f"{model_path}/model.txt", 'w') as f: 
                json.dump(model_dict, f)
            with open(f"{model_path}/model.txt") as f: 
                json.dump(model_dict, f)
        """
        test_model.summary()

        test_data = self.get_data('test', data_test, {**opts['model_opts'], 'batch_size': 1})

        test_results = test_model.predict(test_data['data'][0],
                                          batch_size=1, verbose=1)
        acc = accuracy_score(test_data['data'][1], np.round(test_results))
        f1 = f1_score(test_data['data'][1], np.round(test_results))
        auc = roc_auc_score(test_data['data'][1], np.round(test_results))
        roc = roc_curve(test_data['data'][1], test_results)
        precision = precision_score(test_data['data'][1], np.round(test_results))
        recall = recall_score(test_data['data'][1], np.round(test_results))
        pre_recall = precision_recall_curve(test_data['data'][1], test_results)
        
        # THIS IS TEMPORARY, REMOVE BEFORE RELEASE
        with open(os.path.join(model_path, 'test_output.pkl'), 'wb') as picklefile:
            pickle.dump({'tte': test_data['tte'],
                         'pid': test_data['ped_id'],
                         'gt':test_data['data'][1],
                         'y': test_results,
                         'image': test_data['image']}, picklefile)
            
        with open(os.path.join(model_path, 'predictions.pkl'), 'wb') as picklefile:
            pickle.dump({'predictions': test_results,
                         'test_data': test_data['data'][1]}, picklefile)


        print('acc:{:.2f} auc:{:.2f} f1:{:.2f} precision:{:.2f} recall:{:.2f}'.format(acc, auc, f1, precision, recall))

        save_results_path = os.path.join(model_path, '{:.2f}'.format(acc) + '.yaml')

        if not os.path.exists(save_results_path):
            results = {'acc': acc,
                       'auc': auc,
                       'f1': f1,
                       'roc': roc,
                       'precision': precision,
                       'recall': recall,
                       'pre_recall_curve': pre_recall}

            with open(save_results_path, 'w') as fid:
                yaml.dump(results, fid)

        print(f"Saved model path: {model_path}") # required by train_ensemble.py

        return acc, auc, f1, precision, recall

    def get_model(self, data_params):
        """
        Generates a model
        Args:
            data_params: Data parameters to use for model generation
        Returns:
            A model
        """
        raise NotImplementedError("get_model should be implemented")

    # Auxiliary function
    def _gru(self, name='gru', r_state=False, r_sequence=False):
        """
        A helper function to create a single GRU unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the GRU
            r_sequence: Whether to return a sequence
        Return:
            A GRU unit
        """
        return GRU(units=self._num_hidden_units,
                   return_state=r_state,
                   return_sequences=r_sequence,
                   stateful=False,
                   kernel_regularizer=self._regularizer,
                   recurrent_regularizer=self._regularizer,
                   bias_regularizer=self._regularizer,
                   name=name)

    def _lstm(self, name='lstm', r_state=False, r_sequence=False):
        """
        A helper function to create a single LSTM unit
        Args:
            name: Name of the layer
            r_state: Whether to return the states of the LSTM
            r_sequence: Whether to return a sequence
        Return:
            A LSTM unit
        """
        return LSTM(units=self._num_hidden_units,
                    return_state=r_state,
                    return_sequences=r_sequence,
                    stateful=False,
                    kernel_regularizer=self._regularizer,
                    recurrent_regularizer=self._regularizer,
                    bias_regularizer=self._regularizer,
                    name=name)

    def create_stack_rnn(self, size, r_state=False, r_sequence=False):
        """
        Creates a stack of recurrent cells
        Args:
            size: The size of stack
            r_state: Whether to return the states of the GRU
            r_sequence: Whether the last stack layer to return a sequence
        Returns:
            A stacked recurrent model
        """
        cells = []
        for i in range(size):
            cells.append(self._rnn_cell(units=self._num_hidden_units,
                                        kernel_regularizer=self._regularizer,
                                        recurrent_regularizer=self._regularizer,
                                        bias_regularizer=self._regularizer, ))
        return RNN(cells, return_sequences=r_sequence, return_state=r_state)

def show_image(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)