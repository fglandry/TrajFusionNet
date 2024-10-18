import cv2
import numpy as np
import torch

from hugging_face.model_trainers.trajectorytransformer import VanillaTransformerForForecast, get_config_for_timeseries_lib as get_config_for_trajectory_pred
from hugging_face.timeseries_utils import denormalize_trajectory_data, transform_trajectory_data
from hugging_face.utils.semantic_segmentation import ade_palette
from utils.dataset_statistics import calculate_stats_for_trajectory_data
from utils.utils import Singleton


class RunningPed(metaclass=Singleton):

    def __init__(self,
                 model_opts):

        self._dataset = model_opts["dataset_full"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get dataset statistics
        self.dataset_statistics = {
            "dataset_means": {},
            "dataset_std_devs": {}
        }
        calculate_stats_for_trajectory_data(
            None, None, self.dataset_statistics,
            include_labels=True, debug=True)
        # Get rid of speed values
        #for e in self.dataset_statistics.values():
        #    e["trajectory"] = e["trajectory"][:-1]

        # Get pretrained trajectory predictor -------------------------------------------
        config_for_trajectory_predictor = get_config_for_trajectory_pred(
            encoder_input_size=5, seq_len=15, hyperparams={}, pred_len=60)
        if self._dataset in ["pie", "combined"]:
            checkpoint = "data/models/pie/TrajectoryTransformer/13Aug2024-11h16m29s_TE22"
        elif self._dataset == "jaad_all":
            checkpoint = "data/models/jaad/TrajectoryTransformer/05Oct2024-11h30m11s_TE24"
        elif self._dataset == "jaad_beh":
            checkpoint = "data/models/jaad/TrajectoryTransformer/10Aug2024-11h55m10s_TE23"
        pretrained_model = VanillaTransformerForForecast.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_trajectory_predictor,
            ignore_mismatched_sizes=True)
        
        # Make all layers untrainable
        for child in pretrained_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.traj_TF = pretrained_model

    def compute_running_ped(self, img_data,
                            feature_type, 
                            full_bbox_sequences,
                            full_rel_bbox_seq,
                            full_veh_speed, 
                            i,
                            keep_channel_with_ped=True):

        img_features = img_data.copy()

        bbox_sequence = full_bbox_sequences[i]
        rel_bbox_seq = full_rel_bbox_seq[i]
        veh_speed = full_veh_speed[i]
        traj_data = np.concatenate([rel_bbox_seq, veh_speed], axis=1)
        
        # Normalize rel box data
        rel_bbox_seq_norm = transform_trajectory_data(traj_data, 
            "z_score", dataset_statistics=self.dataset_statistics)
        rel_bbox_seq_norm = np.expand_dims(rel_bbox_seq_norm, axis=0)
        rel_bbox_seq_norm = torch.FloatTensor(rel_bbox_seq_norm) # .to(self.device)

        # Run trajectory prediction
        output = self.traj_TF(
            normalized_trajectory_values=rel_bbox_seq_norm,
            return_logits=True)
        
        # Denormalize data
        output = output.squeeze(0).numpy()
        denormalized = denormalize_trajectory_data(
            output, "z_score", self.dataset_statistics)
        abs_pred_coords = np.add(denormalized[:,0:4], bbox_sequence[0])
        abs_pred_coords = np.concatenate([abs_pred_coords, np.expand_dims(denormalized[:,-1], 1)], axis=1) # re-add speed

        # simple_palette = [32, 64, 96, 128, 160, 192, 224]
        simple_palette = [[32, 32], [64, 64], [96, 96], [128, 128], [160, 160], [192, 192], [224, 224]]
        sp_idx = 0
        
        """
        for idx, coords in enumerate(abs_pred_coords):
            b_org = list(map(int, coords[0:4])).copy()
            #start_point = (b_org[0], b_org[1])
            #end_point = (b_org[2], b_org[3])
            #thickness = 2
            #color = np.array(ade_palette()[idx])
            #color = tuple([int(c) for c in color])
            #img_features[:,:,1] = cv2.rectangle(img_features, start_point, end_point, color, thickness)[:,:,1]
            img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 1] = np.array(ade_palette()[idx])[1]

        for idx, coords in enumerate(bbox_sequence):
            b_org = list(map(int, coords[0:4])).copy()
            #start_point = (b_org[0], b_org[1])
            #end_point = (b_org[2], b_org[3])
            #thickness = 2
            #color = np.array(ade_palette()[idx+60])
            #color = tuple([int(c) for c in color])
            #img_features[:,:,0] = cv2.rectangle(img_features, start_point, end_point, color, thickness)[:,:,0]
            img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0] = np.array(ade_palette()[idx+60])[0]
        """
        
        if feature_type == "scene_context_with_running_ped_v2_doubled":

            for idx, coords in enumerate(bbox_sequence):
                if idx == 0 or ((idx+1) % 5 == 0):
                    b_org = list(map(int, coords[0:4])).copy()
                    img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = np.array(ade_palette()[idx])[0:2]

                #start_point = (b_org[0], b_org[1])
                #end_point = (b_org[2], b_org[3])
                #thickness = 2
                #color = np.array(ade_palette()[idx])
                #color = tuple([int(c) for c in color])
                #img_features = cv2.rectangle(img_features, start_point, end_point, color, thickness)

            # Put index 0 in foreground
            # coords = bbox_sequence[0]
            # b_org = list(map(int, coords[0:4])).copy()
            # img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = np.array(ade_palette()[0])[0:2]

            
        else:
            for idx, coords in enumerate(abs_pred_coords):

                b_org = list(map(int, coords[0:4])).copy()
                
                if (idx+1) % 5 == 0:
                    img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = np.array(ade_palette()[idx+15])[0:2]
                
                #if keep_channel_with_ped:
                    #img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = np.array(ade_palette()[idx+1])[0:2]
                    #tart_point = (b_org[0], b_org[1])
                    #end_point = (b_org[2], b_org[3])
                    #thickness = 2
                    #color = np.array(ade_palette()[idx])
                    #color = tuple([int(c) for c in color])
                    #img_features = cv2.rectangle(img_features, start_point, end_point, color, thickness)

                # img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = [0, 255, 0]

            for idx, coords in enumerate(bbox_sequence):

                b_org = list(map(int, coords[0:4])).copy()

                if idx == len(bbox_sequence)-1:
                    img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = np.array(ade_palette()[idx])[0:2]

                #if keep_channel_with_ped:
                #    if idx % 2 == 1:
                #        if idx < 8:
                #            img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0] = np.array(ade_palette()[idx])[0]
                #        elif idx >= 8:
                #            img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 1] = np.array(ade_palette()[idx])[1]
                #else:
                #    img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], :] = np.array(ade_palette()[idx])
                #img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = np.array(ade_palette()[idx])[0:2]
            
            """
            # Add last element in bbox_sequence to foreground
            for idx, coords in enumerate(bbox_sequence):

                b_org = list(map(int, coords[0:4])).copy()
                if keep_channel_with_ped:
                    if idx == len(bbox_sequence)-1:
                        img_features[b_org[1]:b_org[3], b_org[0]:b_org[2], 0:2] = np.array([simple_palette[sp_idx]])
                        #start_point = (b_org[0], b_org[1])
                        #end_point = (b_org[2], b_org[3])
                        #thickness = 2
                        #color = np.array(ade_palette()[idx])
                        #color = tuple([int(c) for c in color])
                        #img_features = cv2.rectangle(img_features, start_point, end_point, color, thickness)
            """
        return img_features