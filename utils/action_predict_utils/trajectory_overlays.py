
import numpy as np
import torch

from models.hugging_face.model_trainers.trajectorytransformer import VanillaTransformerForForecast, get_config_for_timeseries_lib as get_config_for_trajectory_pred
from models.hugging_face.timeseries_utils import denormalize_trajectory_data, normalize_trajectory_data
from models.hugging_face.utils.semantic_segmentation import ade_palette
from utils.dataset_statistics import calculate_stats_for_trajectory_data
from utils.utils import Singleton


class TrajectoryOverlays(metaclass=Singleton):
    """ Class to compute pedestrian trajectory overlays as part of the
        Visual Attention Module (VAM)
    """

    def __init__(self,
                 model_opts: dict,
                 submodels_paths: dict = None):

        self._dataset = model_opts["dataset_full"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get dataset statistics
        self.dataset_statistics = {
            "dataset_means": {},
            "dataset_std_devs": {}
        }

        calculate_stats_for_trajectory_data(
            None, None, 
            self.dataset_statistics, model_opts,
            include_labels=True, 
            use_precomputed_values=True)

        # Get pretrained trajectory predictor -------------------------------------------
        config_for_trajectory_predictor = get_config_for_trajectory_pred(
            encoder_input_size=5, seq_len=15, hyperparams={}, pred_len=60)
        if submodels_paths:
            checkpoint = submodels_paths["traj_tf_path"]
        else:
            if self._dataset in ["pie", "combined"]:
                checkpoint = "data/models/pie/TrajectoryTransformer/13Aug2024-11h16m29s_TE22"
            elif self._dataset == "jaad_all":
                checkpoint = "data/models/jaad/TrajectoryTransformer/05Oct2024-11h30m11s_TE24"
            elif self._dataset == "jaad_beh":
                #checkpoint = "data/models/jaad/TrajectoryTransformer/20Dec2024-14h24m55s_BE26"
                checkpoint = "data/models/jaad/TrajectoryTransformer/20Nov2024-10h50m14s_TE25"
        pretrained_model = VanillaTransformerForForecast.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_trajectory_predictor,
            ignore_mismatched_sizes=True)
        
        # Make all layers untrainable
        for child in pretrained_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.traj_TF = pretrained_model

    def compute_trajectory_overlays(self, 
            img_data: np.ndarray,
            feature_type: str, 
            full_bbox_seqs: np.ndarray,
            full_rel_bbox_seqs: np.ndarray,
            full_veh_speed_seqs: np.ndarray, 
            i: int
        ):
        """ Compute pedestrian trajectory overlays, which will be added to 'img_data'.
            The overlays are obtained by predicting future pedestrian bounding boxes.
        Args:
            img_data [np.ndarray]: image data to add overlays to
            feature_type [str]: feature type to compute 
            full_bbox_seqs [np.ndarray]: all sequences of pedestrian bounding boxes 
            full_rel_bbox_seqs [np.ndarray]: all sequences of relative ped bounding boxes 
                                             (offset by subtracting initial bb)
            full_veh_speed_seqs [np.ndarray]: all sequences of vehicle speeds
            i [int]: sequence ID
        Returns:
            img_features [np.ndarray] with overlays
        """

        img_features = img_data.copy()

        bbox_sequence = full_bbox_seqs[i]
        rel_bbox_seq = full_rel_bbox_seqs[i]
        veh_speed = full_veh_speed_seqs[i]
        traj_data = np.concatenate([rel_bbox_seq, veh_speed], axis=1)
        
        # Normalize trajectory data
        trajectory_seq_norm = normalize_trajectory_data(traj_data, 
            "z_score", dataset_statistics=self.dataset_statistics)
        trajectory_seq_norm = np.expand_dims(trajectory_seq_norm, axis=0)
        trajectory_seq_norm = torch.FloatTensor(trajectory_seq_norm)

        # Run trajectory prediction
        output = self.traj_TF(
            normalized_trajectory_values=trajectory_seq_norm,
            return_logits=True)
        
        # Denormalize data
        output = output.squeeze(0).numpy()
        denormalized = denormalize_trajectory_data(
            output, "z_score", self.dataset_statistics)
        
        # re-add first bbox in original sequence to get absolute coordinates
        absolute_pred_coords = np.add(denormalized[:,0:4], bbox_sequence[0])

        # re-add speed to 'absolute_pred_coords'
        absolute_pred_coords = np.concatenate([absolute_pred_coords, 
                                               np.expand_dims(denormalized[:,-1], 1)], axis=1)

        if feature_type == "scene_context_with_ped_overlays_previous":
            # Add observed bounding boxes as overlays on image (first image in sequence)
            #color_idx = 0
            for idx, coords in enumerate(bbox_sequence):
                if idx == 0 or ((idx+1) % 5 == 0): # add first bbox and then every 5th
                    b_org = list(map(int, coords[0:4])).copy()
                    img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = \
                        np.array(ade_palette()[idx])[0:2]

        else:
            # Add predicted bounding boxes as overlays on image (last image in sequence)
            for idx, coords in enumerate(absolute_pred_coords):

                b_org = list(map(int, coords[0:4])).copy()
                
                if (idx+1) % 5 == 0: # only add each 5th box
                    img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = \
                        np.array(ade_palette()[idx+15])[0:2] # np.array(ade_palette()[color_idx])[0:2]

            # Add observed bbox at time t to forefront
            idx = len(bbox_sequence)-1
            coords = bbox_sequence[idx]
            b_org = list(map(int, coords[0:4])).copy()

            img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = \
                np.array(ade_palette()[idx])[0:2] # np.array(ade_palette()[color_idx-1])[0:2]

        return img_features
        
        if feature_type == "scene_context_with_ped_overlays_previous":
            # Add observed bounding boxes as overlays on image (first image in sequence)
            color_idx = 0
            for idx, coords in enumerate(bbox_sequence):
                if idx == 0 or ((idx+1) % 5 == 0): # add first bbox and then every 5th
                    b_org = list(map(int, coords[0:4])).copy()
                    img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = \
                        np.array(ade_palette()[color_idx])[0:2]
                    color_idx += 1

        else:
            # Add predicted bounding boxes as overlays on image (last image in sequence)
            color_idx = 4
            for idx, coords in enumerate(absolute_pred_coords):

                b_org = list(map(int, coords[0:4])).copy()
                
                if (idx+1) % 5 == 0: # only add each 5th box
                    img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = \
                        np.array(ade_palette()[color_idx])[0:2]
                    color_idx += 1

            # Add observed bbox at time t to forefront
            color_idx = 4
            idx = len(bbox_sequence)-1
            coords = bbox_sequence[idx]
            b_org = list(map(int, coords[0:4])).copy()

            img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = \
                np.array(ade_palette()[color_idx-1])[0:2]

        return img_features
        