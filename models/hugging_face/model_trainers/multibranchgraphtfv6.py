from typing import Optional, Tuple, Union
import cv2
import time
import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from transformers import ConvNextV2Model, VanModel, TrainingArguments, Trainer, TimesformerConfig
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers import AutoImageProcessor, ConvNextV2ForImageClassification, ViTForImageClassification, ViTModel
from transformers import GraphormerModel, GraphormerConfig
from transformers import AdamW, get_constant_schedule, get_linear_schedule_with_warmup
from torchvision.models import resnet18 #vgg13

import matplotlib.pyplot as plt 

from models.hugging_face.model_trainers.trajectorytransformer import VanillaTransformerForForecast, get_config_for_timeseries_lib as get_config_for_trajectory_pred
from models.hugging_face.model_trainers.trajectorytransformerb import TrajectoryTransformerModel, get_config_for_timeseries_lib as get_config_for_trajectory_classification
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model, normalize_trajectory_data, denormalize_trajectory_data, HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig, TorchTimeseriesDataset
from models.hugging_face.utilities import compute_loss, get_class_labels_info, get_device
from models.hugging_face.utils.create_optimizer import get_optimizer, override_create_optimizer
from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
from libs.time_series_library.models_tsl.Tokengt import Model as TokengtTransformer
from libs.time_series_library.models_tsl.TrajContextTF import Model as TrajContextTF
from libs.time_series_library.models_tsl.TFwithCrossAttn import Model as TFwithCrossAttn
from models.custom_layers_pytorch import CrossAttention, SelfAttention

NET_INNER_DIM = 40
DROPOUT = 0.1

class MultiBranchGraphTFV6(HuggingFaceTimeSeriesModel):
    """ Base Transformer with cross attention between modalities
    """

    def train(self,
              data_train, 
              data_val,
              batch_size, 
              epochs,
              model_path,  
              generator=False,
              train_opts=None,
              dataset_statistics=None,
              hyperparams=None,
              class_w=None,
              *args, **kwargs):
        print("Starting model loading for model Vanilla Transformer ===========================")

        # Get parameters used by time series library model
        alternate_branch_training = False
        add_graph_context_to_lstm = True
        timeseries_element = data_train['data'][0][0][0][-1] # index 0 consists of segmentation map
                                                            # index 2 consists of timeseries context
                                                            # index -1 consists of timeseries sequence
        timeseries_context_element = data_train['data'][0][0][0][2] \
            if len(data_train["data_params"]["data_sizes"]) <= 3 else data_train['data'][0][0][0][3]
        encoder_input_size = timeseries_element.shape[-1] # + 30 # + resnet encodings
        seq_len = timeseries_element.shape[-2]
        context_len = 15 # timeseries_context_element.shape[-1]
        
        # Get model
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        config_for_timeseries_lib = get_config_for_timeseries_lib(encoder_input_size, seq_len, hyperparams)
        config_for_context_timeseries = get_config_for_context_timeseries(encoder_input_size, context_len, hyperparams,
                                                                          add_graph_context_to_lstm=add_graph_context_to_lstm)
        config_for_pose_timeseries = get_config_for_pose_timeseries(encoder_input_size, 8, hyperparams)
        config_for_huggingface = TimeSeriesTransformerConfig()

        model = VanillaTransformerForClassification(config_for_huggingface, 
                                                    config_for_timeseries_lib,
                                                    config_for_context_timeseries,
                                                    config_for_pose_timeseries,
                                                    add_graph_context_to_lstm=add_graph_context_to_lstm,
                                                    class_w=class_w,
                                                    dataset_statistics=dataset_statistics)
        summary(model)

        # Get datasets
        video_model_config = TimesformerConfig()
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, video_model_config,
            get_image_transform=True, img_model_config=None,
            dataset_statistics=dataset_statistics, ignore_sem_map=True)

        warmup_ratio = 0.1
        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs = 1 if alternate_branch_training else epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1 # added
        )

        optimizer, lr_scheduler = get_optimizer(self, model, args, 
            train_dataset, val_dataset, data_train, train_opts, warmup_ratio)

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=None,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn,
            optimizers=(optimizer, lr_scheduler)
        )

        # Train model
        print("Starting training of model Vanilla Transformer ===========================")
        
        if alternate_branch_training:
            for i in range(epochs):
                train_results = trainer.train()

                optimizer, lr_scheduler = get_optimizer(self, model, args, 
                    train_dataset, val_dataset, data_train, train_opts, warmup_ratio, 
                    alternate_training=True, epoch_index=i+1)
                
                trainer.optimizer = optimizer
                trainer.lr_scheduler = lr_scheduler
        else:
            train_results = trainer.train()
        
        return {
            "trainer": trainer,
            "val_transform": val_transforms_dicts
        }

    def test(self,
             test_data,
             training_result,
             model_path,
             generator=False,
             complete_data=None):
        
        print("Starting inference using trained model Vanilla Transformer ===========================")

        return test_time_series_based_model(
            test_data,
            training_result,
            model_path,
            generator,
            ignore_sem_map=True,
            complete_data=complete_data
        )


class VanillaTransformerForClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib=None,
                 config_for_context_timeseries=None,
                 config_for_pose_timeseries=None,
                 config_for_graphormer=None,
                 add_graph_context_to_lstm=False,
                 class_w=None,
                 dataset_statistics=None,
                 dataset_name=None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self._dataset = dataset_name

        # self.class_w = torch.tensor([0.7484, 0.2516]).to(self._device)
        self.dataset_statistics = dataset_statistics
        self.class_w = torch.tensor(class_w).to(self._device)
        self.add_graph_context_to_net = add_graph_context_to_lstm
        self.is_lstm_bidirectional = False
        self.num_labels = config_for_huggingface.num_labels
        timeseries_config = config_for_timeseries_lib[0]
        context_config = config_for_context_timeseries[0]

        # MODEL PARAMETERS ==========================================

        # LSTM is trained with sequential pedestrian data
        enc_in = 552 # 40, 808 # timeseries_config.num_class
        hidden_size = timeseries_config.num_class # enc_in
        layers = 2

        # Classifier head
        self.van_output_size = 512
        self.combined_classifier_hidden_size = timeseries_config.num_class + \
                                               context_config.num_class
        self.max_classifier_hidden_size = NET_INNER_DIM # max(timeseries_config.num_class, 
                                              # context_config.num_class)
        self.fc1_neurons = 1 * 1 * self.max_classifier_hidden_size # 3
        #self.fc1_neurons = self.fc1_neurons * 2 if self.is_lstm_bidirectional else self.fc1_neurons
        self.fc2_neurons = 40

        # MODEL LAYERS ===============================================

        self.transformer = VanillaTransformerCrossAttnModel1(
            config_for_huggingface, config_for_timeseries_lib)
        
        self.pred_transformer = VanillaTransformerCrossAttnModel1(
            config_for_huggingface, config_for_timeseries_lib)

        #self.pose_transformer = VanillaTransformerCrossAttnModel1(
        #    config_for_huggingface, config_for_pose_timeseries)

        #self.context_transformer = VanillaTransformerCrossAttnModel2(
        #    config_for_huggingface, config_for_context_timeseries)
        
        #self.prev_context_transformer = VanillaTransformerCrossAttnModel2(
        #    config_for_huggingface, config_for_context_timeseries)
        
        # Get pretrained VAN Model -------------------------------------------
        label2id, id2label = get_class_labels_info()
        if self._dataset in ["pie", "combined"]:
            checkpoint = "data/models/jaad/VAN/15Jul2023-13h22m25s_EL7"
            # "data/models/combined/VAN/07Jun2024-14h39m16s_C",
        elif self._dataset == "jaad_all":
            checkpoint = "data/models/jaad/VAN/15Jul2023-13h22m25s_EL7"
        elif self._dataset == "jaad_beh":
            checkpoint = "data/models/jaad/VAN/10Aug2024-14h14m04s_TJ5"
        pretrained_model = VanModel.from_pretrained(
            checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)
        
        # Make all layers untrainable
        for child in pretrained_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.van = pretrained_model
        #self.vgg_encoder = pretrained_model

        # Get pretrained trajectory predictor -------------------------------------------
        config_for_trajectory_predictor = get_config_for_trajectory_pred(
            encoder_input_size=5, seq_len=15, hyperparams={})
        if self._dataset in ["pie", "combined"]:
            checkpoint = "data/models/pie/TrajectoryTransformer/05Jul2024-16h28m50s_TE3"
        elif self._dataset == "jaad_all":
            checkpoint = "data/models/jaad/TrajectoryTransformer/05Aug2024-15h46m13s_TE22"
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

        # Get pretrained trajectory classifier -------------------------------------------
        config_for_trajectory_predictor = get_config_for_trajectory_classification(
            encoder_input_size=6, seq_len=75, hyperparams={})
        if self._dataset in ["pie", "combined"]:
            checkpoint = "data/models/pie/TrajectoryTransformerb/09Aug2024-21h12m39s_TJ4"
        elif self._dataset == "jaad_all":
            checkpoint = "data/models/jaad/TrajectoryTransformerb/10Aug2024-10h17m02s_TJ3"
        elif self._dataset == "jaad_beh":
            checkpoint = "data/models/jaad/TrajectoryTransformerb/10Aug2024-13h03m11s_TJ4"
        pretrained_model = TrajectoryTransformerModel.from_pretrained(
            checkpoint,
            # "data/models/jaad/TrajectoryTransformerb/05Aug2024-16h57m32s_TJ2",
            config_for_timeseries_lib=config_for_trajectory_predictor,
            ignore_mismatched_sizes=True)
        
        # Make all layers untrainable
        for child in pretrained_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.traj_class_TF = pretrained_model

        #self.lstm = nn.LSTM(enc_in, hidden_size, layers, batch_first=True) # bidirectional=True)
        self.dropout = nn.Dropout(p=DROPOUT)

        self.self_attention = SelfAttention(self.max_classifier_hidden_size)
        self.self_attention_ctx = SelfAttention(self.combined_classifier_hidden_size)
        self.fc1 = nn.Linear(self.fc1_neurons, self.fc2_neurons)
        self.fc2 = nn.Linear(self.fc2_neurons, self.num_labels)

        self.projection_lstm = nn.Linear(2 * timeseries_config.num_class, timeseries_config.num_class)
        self.projection_context = nn.Linear(1 * 512, timeseries_config.num_class) # 2 * 552
        self.projection_van = nn.Linear(self.van_output_size, timeseries_config.num_class) # 512

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trajectory_values: Optional[torch.Tensor] = None,
        timeseries_context: Optional[torch.Tensor] = None,
        previous_timeseries_context: Optional[torch.Tensor] = None,
        image_context: Optional[torch.Tensor] = None,
        previous_image_context: Optional[torch.Tensor] = None,
        segmentation_context: Optional[torch.Tensor] = None,
        segmentation_context_2: Optional[torch.Tensor] = None,
        normalized_trajectory_values: Optional[torch.Tensor] = None,
        video_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        debug=False
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        
        """
        trajectory_values = torch.cat(
            [trajectory_values[:,:,0:4], 
             trajectory_values[:,:,-1].unsqueeze(2)], dim=2)
        
        normalized_trajectory_values = torch.cat(
            [normalized_trajectory_values[:,:,0:4], 
             normalized_trajectory_values[:,:,-1].unsqueeze(2)], dim=2)
        """

        add_van_context = True
        add_previous_context = False
        use_separate_tf_for_pose = False
        use_pose_graph_in_ctx = False
        combine_ctx_with_attention = False
        combine_branches_with_attention = False
        return_dict = self._on_entry(output_hidden_states, return_dict)
        self._debug_video_values(debug, video_values)

        # ====================================================================================
        # Apply VAN model to image context ===================================================

        if add_van_context:
            # Get timeseries model output
            output1 = self.van(
                image_context,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output1_tensor = output1.pooler_output if return_dict else output1 # output[1]
            van_output = output1_tensor # self.dropout1(output1_tensor)

            if add_previous_context:
                # Get previous context output from VAN
                output2 = self.van(
                    previous_image_context,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                output2_tensor = output2.pooler_output if return_dict else output2
                van_output_prev = output2_tensor # self.dropout2(output2_tensor)

        # ====================================================================================
        # Apply TF to timeseries context (graph features obtained from =======================
        # semantic segmentation)

        if use_separate_tf_for_pose or use_pose_graph_in_ctx:
            # trajectory_values_no_pose = trajectory_values[:,:,0:5]
            trajectory_pose = trajectory_values[:,:,5:]
            trajectory_tf_input = trajectory_values[:,:,0:5] # trajectory_values[:,:,:21] # trajectory_values_no_pose
        else:
            trajectory_tf_input = trajectory_values

        if use_pose_graph_in_ctx:
            # trajectory_pose = trajectory_values[:,:,5:]
            curr_pose_graph = trajectory_pose[:,-1,:]
            curr_pose_graph = self._format_pose_data_for_tokengt_tf(curr_pose_graph)
        
            # Build timeseries context from scene graph and pedestrian graph
            timeseries_context = timeseries_context[:,0,:,:]
            timeseries_context = torch.cat([timeseries_context, curr_pose_graph], dim=1)
        else:
            timeseries_context = timeseries_context[:,0,:,:]

        """
        outputs_ctx_tf = self.context_transformer(
            timeseries_context[:,:,:] # [:,0:5,:]
        ) # [B, 40]
        """

        """
        if use_separate_tf_for_pose:
            non_graph_ctx_output = self.pose_transformer(
                timeseries_context[:,5:,0].unsqueeze(2)
            )
        """
        """
        sem_map_output = self.vgg_encoder(segmentation_context_2)
        sem_map_output = sem_map_output.pooler_output if self.return_dict else sem_map_output
        sem_map_output = self.vgg_encoder_fc(sem_map_output)
        """

        if add_previous_context:
            
            if use_pose_graph_in_ctx:
                prev_pose_graph = trajectory_pose[:,0,:] # todo revisit
                prev_pose_graph = self._format_pose_data_for_tokengt_tf(prev_pose_graph)

            # Build previous timeseries context from scene graph and pedestrian graph
            previous_timeseries_context = previous_timeseries_context[:,0,:,:]
            previous_timeseries_context = torch.cat([previous_timeseries_context, prev_pose_graph], dim=1)

            outputs_prev_ctx_tf = self.prev_context_transformer(
                previous_timeseries_context
            )

        # Combine all context-based modalities into one context branch =======================

        if self.add_graph_context_to_net:
            if add_van_context:
                if combine_ctx_with_attention:
                    # Combine different context modalities with attention
                    van_output_projected = self._project(van_output, self.projection_van)
                    tuple_to_concat = [outputs_ctx_tf, van_output_projected]
                    original_x = torch.cat(tuple_to_concat, dim=1) # shape=(batch, combined_fc_len)
                    current_ctx_input = self._concatenate_with_attention(self.self_attention_ctx, original_x, 
                        tuple_to_concat, self.combined_classifier_hidden_size)  
                else:
                    # current_ctx_input = torch.cat([outputs_ctx_tf, van_output], dim=1) 
                    current_ctx_input = torch.cat([van_output], dim=1)
            else:
                current_ctx_input = torch.cat([outputs_ctx_tf], dim=1)
            current_ctx_input = current_ctx_input.unsqueeze(1)
            if add_previous_context:
                previous_ctx_input = torch.cat([outputs_prev_ctx_tf, van_output_prev], dim=1) if add_van_context else torch.cat([outputs_prev_ctx_tf], dim=1)
                previous_ctx_input = previous_ctx_input.unsqueeze(1)
                ctx_input = torch.cat([previous_ctx_input, current_ctx_input], dim=1) # [B, 2, D]
            else:
                ctx_input = current_ctx_input # [B, 1, D]
    

        # Apply projections
        ctx_output_projected = self._project(ctx_input, self.projection_context) # [B, 40]

        # ====================================================================================
        # Apply TF to trajectory values ======================================================
        
        """
        outputs = self.transformer(
            # trajectory_tf_input[:,::2,:], # trajectory_tf_input
            trajectory_tf_input[:,2::3,:],
            # outputs_ctx_tf=ctx_output_projected
        ) # [B, 40]
        """

        outputs_pred = self.traj_class_TF(
            trajectory_values=trajectory_values,
            normalized_trajectory_values=normalized_trajectory_values
        )

        # Concatenate trajectory and context branches ========================================

        # tuple_to_concat = [outputs, outputs_pred, ctx_output_projected]
        
        #tuple_to_concat = [outputs_pred, ctx_output_projected]
        #original_x = torch.cat(tuple_to_concat, dim=1) # shape=(batch, combined_fc_len) 
        original_x = ctx_output_projected

        # Add branch-based self-attention ====================================================
        if combine_branches_with_attention:
            x = self._concatenate_with_attention(self.self_attention, original_x, 
                    tuple_to_concat, self.max_classifier_hidden_size)
        else:
            x = original_x

        # Apply fully-connected layers =======================================================
        outputs = self.dropout(nn.ReLU()(self.fc1(x)))
        # outputs = x
        logits = self.fc2(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict,
                            class_w=self.class_w)

    def _concatenate_with_attention(self, self_attention, original_x, 
                                    tuple_to_concat, max_concat_size):
        
        # Concatenate outputs with padding
        tuple_to_concat = self._pad_tensors(tuple_to_concat, max_concat_size)
        x = torch.cat(tuple_to_concat, dim=1) # shape=(batch, nb_models, max_concat_size)

        attention_ctx, attn = self_attention(x)
        attention_ctx = attention_ctx.squeeze(2)
        x = torch.cat((original_x, attention_ctx), dim=1)

        return x
    
    def _project(self, output, projection):
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = projection(output)  # (batch_size, num_classes)
        return output

    def _pad_tensors(self, tensors: list, max_size: int, pad_dim=1):
        for idx, output_tensor in enumerate(tensors):
            tensors[idx] = F.pad(output_tensor, pad=(0, max_size - output_tensor.shape[pad_dim], 0, 0)) # shape=(batch, fc_len)
            if len(tensors[idx].shape) <= 2:
                tensors[idx] = tensors[idx].unsqueeze(1) # shape=(batch, X, fc_len)
        return tensors
    
    def _get_sequence_video_encodings(self, video_values):
        # Get sequence encodings from VGG pretrained context model
        vgg_encodings = None
        x = None
        for i in range(video_values.shape[1]):
            if i % 8 == 0: # trick to save on GPU memory, but less effective
                context = video_values[:,i,:,:,:]
                """
                output = self.encoding_van(
                    context,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict
                )
                """
                output = self.van(context)
                x = output.pooler_output if self.return_dict else output
                x = self.vgg_encoder_fc(x)
                x = x.unsqueeze(1)
            
            if i == 0:
                vgg_encodings = x
            else:
                vgg_encodings = torch.cat((vgg_encodings, x), 1)
        return vgg_encodings

    def _format_pose_data_for_tokengt_tf(self, curr_pose_graph, format="coco_2"):
        """ Format curr_pose_graph so that it can be processed by tokengt transformer
        """
        if format == "coco":
            # curr_pose_graph is formatted as follows:
            # x_coords (0:8), y_coords (8:16), distances (16:24), angles (24:32)

            # Format curr_pose_graph so that it can be processed by tokengt transformer
            nb_kp = 8 # number of keypoints
            keypoint_coords = torch.cat([curr_pose_graph[:,0:nb_kp].unsqueeze(2),
                                            curr_pose_graph[:,nb_kp:2*nb_kp].unsqueeze(2)], dim=2)
            distances_angles = torch.cat([curr_pose_graph[:,2*nb_kp:3*nb_kp].unsqueeze(2),
                                            curr_pose_graph[:,3*nb_kp:4*nb_kp].unsqueeze(2)], dim=2)
        elif format == "coco_2":
            # curr_pose_graph is formatted as follows:
            # x_coords (0:13), y_coords (13:26), distances (26:40), angles (40:54)

            # Format curr_pose_graph so that it can be processed by tokengt transformer
            nb_kp = 13 # number of keypoints
            keypoint_coords = torch.cat([curr_pose_graph[:,0:nb_kp].unsqueeze(2),
                                            curr_pose_graph[:,nb_kp:2*nb_kp].unsqueeze(2)], dim=2)
            distances_angles = torch.cat([curr_pose_graph[:,2*nb_kp:3*nb_kp+1].unsqueeze(2),
                                            curr_pose_graph[:,3*nb_kp+1:4*nb_kp+2].unsqueeze(2)], dim=2)
        else:
            raise Exception
        curr_pose_graph = torch.cat([keypoint_coords, distances_angles], dim=1)
        return curr_pose_graph

    def _format_data_and_feed_to_graphormer(self, timeseries_context):
        """ Work under progress... """
        
        timeseries_context = torch.squeeze(timeseries_context, 2)
        
        input_nodes = timeseries_context[:,None,None,9] # ped_sem_category
        zeros = torch.from_numpy(np.full((16, 1, 1), 0)).to(device='cuda')
        input_nodes = torch.cat([input_nodes,zeros], dim=2)

        road_cm_coord_node = timeseries_context[:,None,0:2] # road_cm_coord
        road_min_coord_node = timeseries_context[:,None,4:6]
        input_nodes = torch.cat([input_nodes,road_min_coord_node,road_cm_coord_node], dim=1)

        in_degree = torch.from_numpy(np.full((16, 3), 1)).to(device='cuda')
        out_degree = torch.from_numpy(np.full((16, 3), 1)).to(device='cuda')
        spatial_pos = torch.from_numpy(np.full((16, 3, 3), 1)).to(device='cuda')
        attn_edge_type = torch.from_numpy(np.full((16, 3, 3), 1)).to(device='cuda')
        
        output = self.graph_transformer(
            input_nodes=input_nodes, # torch.LongTensor,
            input_edges=input_nodes, # torch.LongTensor,
            attn_bias=None, # torch.Tensor,
            in_degree=in_degree, # torch.LongTensor,
            out_degree=out_degree, # torch.LongTensor,
            spatial_pos=spatial_pos, # torch.LongTensor,
            attn_edge_type=attn_edge_type # torch.LongTensor,
        )
        return output
    
    def _debug_video_values(self, debug, video_values):
        if debug:
            cpu_array = video_values[1,1,...].permute(1, 2, 0).cpu()
            #plt.imshow(cpu_array)
            np_array = cpu_array.detach().numpy()
            cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}_1.png", np_array)

    def _on_entry(self, output_hidden_states, return_dict):
        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.return_dict = return_dict
        return return_dict

class VanillaTransformerCrossAttnModel1(TimeSeriesTransformerPreTrainedModel):
    
    base_model_prefix = "transformer" # needs to be a class property
    
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib
        ):
        super().__init__(config_for_huggingface)
        self.traj_context_tf = VanillaTransformerTSLModel(config_for_timeseries_lib[1])
        # self.traj_context_tf = TrajContextTF(config_for_timeseries_lib[1])
        # self.traj_context_tf = TFwithCrossAttn(config_for_timeseries_lib[1])            
        
        #self.cross_attention_1 = CrossAttention(5) # CrossAttention(77)
        #self.cross_attention_2 = CrossAttention(5)
        #self.box_emb_fc = nn.Linear(4, 5)
        #self.box_abs_emb_fc = nn.Linear(4, 5)
        #self.box_speed_fc = nn.Linear(2, 5)
        #self.speed_emb_fc = nn.Linear(1, 5)
        #self.pose_emb_fc = nn.Linear(72, 20)
        #self.hog_emb_fc = nn.Linear(6084, 50) # nn.Linear(3969, 100) 
        
        #self.attn_emb_fc_1 = nn.Linear(15, 5)
        #self.last_attn_emb_fc = nn.Linear(225, 10)
        #self.attn1_fc = nn.Linear(15, 5)
        self.use_cross_attention = False
        self.residual_at_the_end = False
        self.embed_hog_features = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values,
        outputs_ctx_tf=None,
        *args,
        **kwargs
    ):
        if self.use_cross_attention:
            # Compute inter-modal cross-attention
            box_values = nn.ReLU()(self.box_emb_fc(trajectory_values[:,:,0:4]))
            # box_abs_values = nn.ReLU()(self.box_abs_emb_fc(trajectory_values[:,:,4:8]))
            box_speed_values = nn.ReLU()(self.box_speed_fc(trajectory_values[:,:,8:10]))
            speed_values = nn.ReLU()(self.speed_emb_fc(trajectory_values[:,:,10].unsqueeze(2)))

            cross_attn_ctx_1, attn_1 = self.cross_attention_1(box_values, speed_values)
            cross_attn_ctx_2, attn_2 = self.cross_attention_2(box_speed_values, speed_values)
            cross_attn_ctx_3, attn_3 = self.cross_attention_3(box_values, box_speed_values)
            
            trajectory_values = torch.cat([
                trajectory_values, cross_attn_ctx_1, cross_attn_ctx_2, cross_attn_ctx_3], dim=2)

        # Get vanilla transformer model output
        outputs = self.traj_context_tf(
            x_enc=trajectory_values,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None,
            #outputs_ctx_tf=outputs_ctx_tf
        )

        if self.residual_at_the_end:
            attn_1_emb = nn.ReLU()(self.last_attn_emb_fc(attn_1.flatten(1,2)))
            outputs = torch.cat([outputs, attn_1_emb], dim=1)

        return outputs
    
class VanillaTransformerCrossAttnModel2(TimeSeriesTransformerPreTrainedModel):
    
    base_model_prefix = "transformer" # needs to be a class property
    
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib
        ):
        super().__init__(config_for_huggingface)
        self.tsl_transformer_0 = TokengtTransformer(config_for_timeseries_lib[1]) 
        self.cross_attn_dim = 1

        self.cross_attention_1 = CrossAttention(self.cross_attn_dim) # CrossAttention(20)
        self.cross_attention_2 = CrossAttention(self.cross_attn_dim)
        self.cross_attention_3 = CrossAttention(self.cross_attn_dim)
        self.cross_attention_4 = CrossAttention(self.cross_attn_dim)

        self.road_emb_fc = nn.Linear(9, 5) # (9, self.cross_attn_dim)
        self.sidewalk_emb_fc = nn.Linear(4, 5)
        self.pedestrians_emb_fc = nn.Linear(8, 5)
        self.vehicles_emb_fc = nn.Linear(8, 5)
        
        self.use_cross_attention = False
        self.cross_attention_with_context_as_last_dim = False
        self.cross_attention_with_context_as_second_dim = True
        self.residual_at_the_end = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values,
        *args,
        **kwargs
    ):
        if self.use_cross_attention:
            # Compute inter-modal cross-attention
            road_values = nn.ReLU()(self.road_emb_fc(
                torch.squeeze(trajectory_values[:,0:9,:], -1)))
            sidewalk_values = nn.ReLU()(self.sidewalk_emb_fc(
                torch.squeeze(trajectory_values[:,9:13,:], -1)))
            pedestrians_values = nn.ReLU()(self.pedestrians_emb_fc(
                torch.squeeze(trajectory_values[:,13:21,:], -1)))
            vehicles_values = nn.ReLU()(self.vehicles_emb_fc(
                torch.squeeze(trajectory_values[:,21:29,:], -1)))
            
            if self.cross_attention_with_context_as_second_dim:
                unsqueeze_dim = 2 # shape=(batch, dim, 1)
            if self.cross_attention_with_context_as_last_dim:
                unsqueeze_dim = 1 # shape=(batch, 1, dim)
            road_values = road_values.unsqueeze(unsqueeze_dim) 
            sidewalk_values = sidewalk_values.unsqueeze(unsqueeze_dim)
            pedestrians_values = pedestrians_values.unsqueeze(unsqueeze_dim)
            vehicles_values = vehicles_values.unsqueeze(unsqueeze_dim)
            
            cross_attn_ctx_1, attn_1 = self.cross_attention_1(road_values, pedestrians_values)
            cross_attn_ctx_2, attn_2 = self.cross_attention_2(road_values, vehicles_values)
            cross_attn_ctx_3, attn_3 = self.cross_attention_3(sidewalk_values, pedestrians_values)
            cross_attn_ctx_4, attn_4 = self.cross_attention_4(pedestrians_values, vehicles_values)
            
            if self.cross_attention_with_context_as_last_dim:
                cross_attn_ctx_1 = cross_attn_ctx_1.swapaxes(1,2)
                cross_attn_ctx_2 = cross_attn_ctx_2.swapaxes(1,2)
                cross_attn_ctx_3 = cross_attn_ctx_3.swapaxes(1,2)
                cross_attn_ctx_4 = cross_attn_ctx_4.swapaxes(1,2)

            trajectory_values = torch.cat([
                trajectory_values, cross_attn_ctx_1, cross_attn_ctx_2,
                cross_attn_ctx_3, cross_attn_ctx_4], dim=1)

        # Get vanilla transformer model output
        outputs = self.tsl_transformer_0(
            x_enc=trajectory_values,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )

        if self.residual_at_the_end:
            attn_1_emb = nn.ReLU()(self.last_attn_emb_fc(attn_1.flatten(1,2)))
            attn_2_emb = nn.ReLU()(self.last_attn_emb_fc(attn_2.flatten(1,2)))
            attn_3_emb = nn.ReLU()(self.last_attn_emb_fc(attn_3.flatten(1,2)))
            attn_4_emb = nn.ReLU()(self.last_attn_emb_fc(attn_4.flatten(1,2)))
            outputs = torch.cat([outputs, attn_1_emb, attn_2_emb, attn_3_emb, attn_4_emb, attn_5_emb, attn_6_emb], dim=1)

        return outputs

def get_config_for_timeseries_lib(encoder_input_size, seq_len, 
                                  hyperparams):
    seq_len = 5
    hyperparams = hyperparams.get("context_tf", {})

    # time series lib properties
    time_series_dict = {
        "task_name": "classification",
        "pred_len": 0, # for Timesblock
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": 5, # 59, # 22 #77, #6161 # 84 # encoder input size - default value,
        "d_model": 128, # dimension of model - default value 
        "embed": "learned", # time features encoding; note: not used in classification task by vanilla transformer model
        "freq": "h", # freq for time features encoding; note: not used in classification task by vanilla transformer model
        "dropout": DROPOUT, # default,
        "factor": 1, # attn factor; note: not used by vanilla transformer model
        "n_heads": hyperparams.get("n_heads", 4), # num of heads
        "d_ff": hyperparams.get("d_ff", 512), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 2), # num of encoder layers (or 3)
        "seq_len": seq_len, # input sequence length
        "num_class": NET_INNER_DIM, # number of neurons in last Linear layer at the end of model
        # ---------------------------------------------------------------------------------
        "label_len": 1, # Timesnet - start token length
        "num_kernels": 6, # Timesnet - for Inception
        "top_k": 5, # Timesnet - for TimesBlock
        "moving_avg": 3, # FEDformer - window size of moving average, default=25
        "dec_in": 7, # FEDformer - decoder input size
        "d_layers": 2, # FEDformer - num of decoder layers
        "c_out": 7, # FEDformer - output size
        "distil": True, # Informer - whether to use distilling in encoder, using this argument means not using distilling
        #"c_out": 77, # override - MICN - output size
        "p_hidden_dims": [128, 128], # Nonstationary transformer - hidden layer dimensions of projector (List)
        "p_hidden_layers": 2, # Nonstationary transformer - number of hidden layers in projector
        # "num_kernels": 3, # override - Pyraformer
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    time_series_dict_0 = copy.deepcopy(time_series_dict)
    time_series_dict_0["enc_in"] = encoder_input_size
    time_series_dict_0["task_name"] = "encoding"
    config_for_timeseries_lib_0 = TimeSeriesLibraryConfig(time_series_dict_0)
    return config_for_timeseries_lib_0, config_for_timeseries_lib


def get_config_for_context_timeseries(encoder_input_size, seq_len, hyperparams, 
                                      add_graph_context_to_lstm=False):

    hyperparams = hyperparams.get("timeseries_tf", {})

    # time series lib properties
    time_series_dict = {
        "task_name": "classification",
        "graph_type": "scene_graph",
        "pred_len": 0, # for Timesblock
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": 2, #77, #6161 # 84 # encoder input size - default value,
        "d_model": 128, # dimension of model - default value 
        "embed": "learned", # time features encoding; note: not used in classification task by vanilla transformer model
        "freq": "h", # freq for time features encoding; note: not used in classification task by vanilla transformer model
        "dropout": DROPOUT, # default,
        "factor": 1, # attn factor; note: not used by vanilla transformer model
        "n_heads": hyperparams.get("n_heads", 4), # num of heads
        "d_ff": hyperparams.get("d_ff", 512), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 2), # num of encoder layers (or 3)
        "seq_len": seq_len, # + 20, # input sequence length
        "num_class": NET_INNER_DIM, # number of neurons in last Linear layer at the end of model
        # ---------------------------------------------------------------------------------
        "label_len": 1, # Timesnet - start token length
        "num_kernels": 6, # Timesnet - for Inception
        "top_k": 5, # Timesnet - for TimesBlock
        "moving_avg": 3, # FEDformer - window size of moving average, default=25
        "dec_in": 7, # FEDformer - decoder input size
        "d_layers": 2, # FEDformer - num of decoder layers
        "c_out": 7, # FEDformer - output size
        "distil": True, # Informer - whether to use distilling in encoder, using this argument means not using distilling
        #"c_out": 77, # override - MICN - output size
        "p_hidden_dims": [128, 128], # Nonstationary transformer - hidden layer dimensions of projector (List)
        "p_hidden_layers": 2, # Nonstationary transformer - number of hidden layers in projector
        # "num_kernels": 3, # override - Pyraformer
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    time_series_dict_0 = copy.deepcopy(time_series_dict)
    time_series_dict_0["enc_in"] = encoder_input_size
    time_series_dict_0["task_name"] = "encoding"
    config_for_timeseries_lib_0 = TimeSeriesLibraryConfig(time_series_dict_0)
    return config_for_timeseries_lib_0, config_for_timeseries_lib

def get_config_for_pose_timeseries(encoder_input_size, seq_len,
                                   hyperparams):

    hyperparams = hyperparams.get("pose_timeseries", {})

    # time series lib properties
    time_series_dict = {
        "task_name": "classification",
        "graph_type": "pedestrian_graph",
        "pred_len": 0, # for Timesblock
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": 1, #77, #6161 # 84 # encoder input size - default value,
        "d_model": 128, # dimension of model - default value 
        "embed": "learned", # time features encoding; note: not used in classification task by vanilla transformer model
        "freq": "h", # freq for time features encoding; note: not used in classification task by vanilla transformer model
        "dropout": DROPOUT, # default,
        "factor": 1, # attn factor; note: not used by vanilla transformer model
        "n_heads": hyperparams.get("n_heads", 4), # num of heads
        "d_ff": hyperparams.get("d_ff", 512), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 2), # num of encoder layers (or 3)
        "seq_len": 8, # + 20, # input sequence length
        "num_class": NET_INNER_DIM, # number of neurons in last Linear layer at the end of model
        # ---------------------------------------------------------------------------------
        "label_len": 1, # Timesnet - start token length
        "num_kernels": 6, # Timesnet - for Inception
        "top_k": 5, # Timesnet - for TimesBlock
        "moving_avg": 3, # FEDformer - window size of moving average, default=25
        "dec_in": 7, # FEDformer - decoder input size
        "d_layers": 2, # FEDformer - num of decoder layers
        "c_out": 7, # FEDformer - output size
        "distil": True, # Informer - whether to use distilling in encoder, using this argument means not using distilling
        #"c_out": 77, # override - MICN - output size
        "p_hidden_dims": [128, 128], # Nonstationary transformer - hidden layer dimensions of projector (List)
        "p_hidden_layers": 2, # Nonstationary transformer - number of hidden layers in projector
        # "num_kernels": 3, # override - Pyraformer
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    time_series_dict_0 = copy.deepcopy(time_series_dict)
    time_series_dict_0["enc_in"] = encoder_input_size
    time_series_dict_0["task_name"] = "encoding"
    config_for_timeseries_lib_0 = TimeSeriesLibraryConfig(time_series_dict_0)
    return config_for_timeseries_lib_0, config_for_timeseries_lib


def get_number_of_training_steps(data_train, train_opts):
    total_devices = 1 # self.hparams.n_gpus * self.hparams.n_nodes
    train_batches = math.ceil((data_train["count"]["neg_count"]+data_train["count"]["pos_count"]) / train_opts["batch_size"])
    train_batches = train_batches // total_devices
    train_steps = train_opts["epochs"] * train_batches # // self.hparams.accumulate_grad_batches
    return train_steps
