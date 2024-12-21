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
from transformers import TrainerCallback
from torchvision.models import resnet18 #vgg13

import matplotlib.pyplot as plt 

from models.hugging_face.model_trainers.trajectorytransformerb import EncoderTransformer as TrajectoryTransformerModel, get_config_for_timeseries_lib as get_config_for_trajectory_classification
# from hugging_face.model_trainers.vanmultiscale import MultiTransformerForClassification
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model, HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig
from models.hugging_face.utilities import compute_loss, get_class_labels_info, get_device
from models.hugging_face.utils.create_optimizer import get_optimizer, override_create_optimizer
from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
from libs.time_series_library.models_tsl.Tokengt import Model as TokengtTransformer
from libs.time_series_library.models_tsl.TrajContextTF import Model as TrajContextTF
from libs.time_series_library.models_tsl.TFwithCrossAttn import Model as TFwithCrossAttn
from models.custom_layers_pytorch import CrossAttention, SelfAttention

NET_INNER_DIM = 512
NET_OUTER_DIM = 40
DROPOUT = 0.1
CURRENT_EPOCH = 0

class EpochLoggerCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        global CURRENT_EPOCH
        CURRENT_EPOCH = state.epoch

class TrajFusionNet(HuggingFaceTimeSeriesModel):
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
        config_for_huggingface = TimeSeriesTransformerConfig()

        model = VanillaTransformerForClassification(config_for_huggingface, 
                                                    config_for_timeseries_lib,
                                                    config_for_context_timeseries,
                                                    None,
                                                    add_graph_context_to_lstm=add_graph_context_to_lstm,
                                                    class_w=class_w,
                                                    dataset_statistics=dataset_statistics,
                                                    dataset_name=kwargs["model_opts"]["dataset_full"])
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
            learning_rate=train_opts["lr"],
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs = epochs, # 1 if alternate_branch_training else epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1 # added
        )

        optimizer, lr_scheduler = get_optimizer(self, model, args, 
            train_dataset, val_dataset, data_train, train_opts, warmup_ratio)
            # alternate_training=False)

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
        initial_weights = copy.deepcopy(model.state_dict())

        trainer.train()

        best_trainer = trainer
        best_scenario_in_second_round = False

        if alternate_branch_training:
            # Reset weights and keep training model with learning rate of VAN
            # branch set to 0 + weights of embed layer set to 0, during first 5 epochs
            model.load_state_dict(initial_weights)
            with torch.no_grad(): 
                model.van_output_embed.weight.zero_()
                model.van_output_embed.bias.zero_()

            for i in range(epochs):
                #if i == 0:
                #    model.load_state_dict(initial_weights)
                #    # trainer.model.load_state_dict(initial_weights)
                
                optimizer, lr_scheduler = get_optimizer(self, model, args, 
                    train_dataset, val_dataset, data_train, train_opts, warmup_ratio, 
                    alternate_training=True, epoch_index=i+1)
                
                trainer = self._get_trainer(model, args, train_dataset, 
                                            val_dataset, optimizer, lr_scheduler)
                trainer.args.num_train_epochs = 1
                
                #trainer.optimizer = optimizer
                #trainer.lr_scheduler = lr_scheduler

                trainer.train()
                metric = best_trainer.state.best_metric if best_trainer.state.best_metric is not None else 0.0
                if trainer.state.best_metric > metric:
                    best_trainer = trainer # copy.copy()
                    best_scenario_in_second_round = True

            if not best_scenario_in_second_round:
                model.load_state_dict(initial_weights)
                optimizer, lr_scheduler = get_optimizer(self, model, args, 
                    train_dataset, val_dataset, data_train, train_opts, warmup_ratio, 
                    alternate_training=False)
                
                trainer = self._get_trainer(model, args, train_dataset, 
                                            val_dataset, optimizer, lr_scheduler)
                trainer.args.num_train_epochs = epochs

                trainer.train()
                best_trainer = trainer

        
        print(f"Best AUC metric: {best_trainer.state.best_metric}")
        
        return {
            "trainer": best_trainer,
            "val_transform": val_transforms_dicts
        }

    def _get_trainer(self, model, args, train_dataset, 
                     val_dataset, optimizer, lr_scheduler):
        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=None,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn,
            optimizers=(optimizer, lr_scheduler),
            callbacks=[EpochLoggerCallback()]
        )
        return trainer

    def test(self,
             test_data,
             training_result,
             model_path,
             generator=False,
             complete_data=None,
             *args,
             **kwargs):
        
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

        # Classifier head
        self.van_output_size = 512
        self.combined_classifier_hidden_size = timeseries_config.num_class + \
                                               context_config.num_class
        self.max_classifier_hidden_size = NET_OUTER_DIM # max(timeseries_config.num_class, 
                                              # context_config.num_class)
        self.max_classifier_hidden_size_van = NET_INNER_DIM
        self.fc1_neurons = 80 # 1 * 3 * self.max_classifier_hidden_size  + 40 # 2 * 40
        #self.fc1_neurons = self.fc1_neurons * 2 if self.is_lstm_bidirectional else self.fc1_neurons
        self.fc2_neurons = 40

        # MODEL LAYERS ===============================================

        #self.transformer = VanillaTransformerCrossAttnModel1(
        #    config_for_huggingface, config_for_timeseries_lib)
        
        #self.pred_transformer = VanillaTransformerCrossAttnModel1(
        #    config_for_huggingface, config_for_timeseries_lib)

        #self.pose_transformer = VanillaTransformerCrossAttnModel1(
        #    config_for_huggingface, config_for_pose_timeseries)

        #self.context_transformer = VanillaTransformerCrossAttnModel2(
        #    config_for_huggingface, config_for_context_timeseries)
        
        #self.prev_context_transformer = VanillaTransformerCrossAttnModel2(
        #    config_for_huggingface, config_for_context_timeseries)
        
        # Get pretrained VAN Model -------------------------------------------
        label2id, id2label = get_class_labels_info()
        if self._dataset in ["pie", "combined"]:
            checkpoint1 = "data/models/pie/VAN/14Oct2024-00h13m09s_VA10"
            checkpoint2 = "data/models/pie/VAN/14Oct2024-10h37m58s_VA11"
            #checkpoint1 = "data/models/jaad/VAN/12Oct2024-20h59m24s_VA6"
            #checkpoint2 = "data/models/jaad/VAN/12Oct2024-23h06m29s_VA7"
        elif self._dataset == "jaad_all":
            checkpoint1 = "data/models/jaad/VAN/12Oct2024-20h59m24s_VA6"
            checkpoint2 = "data/models/jaad/VAN/12Oct2024-23h06m29s_VA7"
            #checkpoint1 = "data/models/jaad/VAN/13Oct2024-20h16m00s_VA8"
            #checkpoint2 = "data/models/jaad/VAN/13Oct2024-20h56m50s_VA9"
        elif self._dataset == "jaad_beh":
            checkpoint1 = "data/models/jaad/VAN/13Oct2024-20h16m00s_VA8"
            checkpoint2 = "data/models/jaad/VAN/13Oct2024-20h56m50s_VA9"
            #checkpoint1 = "data/models/jaad/VAN/12Oct2024-20h59m24s_VA6"
            #checkpoint2 = "data/models/jaad/VAN/12Oct2024-23h06m29s_VA7"
        
        self.use_separate_vans = True
        if self.use_separate_vans:

            pretrained_model = VanModel.from_pretrained(
                checkpoint1,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True)
            
            # Make all layers untrainable
            for child in pretrained_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            self.van1 = pretrained_model

            pretrained_model = VanModel.from_pretrained(
                checkpoint2,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True)
            
            # Make all layers untrainable
            for child in pretrained_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            self.van2 = pretrained_model


        # Get pretrained trajectory classifier -------------------------------------------
        config_for_trajectory_predictor = get_config_for_trajectory_classification(
            encoder_input_size=5, seq_len=75, hyperparams={})
        if self._dataset in ["pie", "combined"]:
            checkpoint = "data/models/pie/TrajectoryTransformerb/20Nov2024-17h42m36s_NW1"
            #checkpoint = "data/models/pie/TrajectoryTransformerb/06Sep2024-09h18m20s_TJ5"
        elif self._dataset == "jaad_all":
            checkpoint = "data/models/jaad/TrajectoryTransformerb/20Nov2024-12h02m51s_TJ8"
            #checkpoint = "data/models/jaad/TrajectoryTransformerb/13Oct2024-15h45m56s_TJ7"
        elif self._dataset == "jaad_beh":
            checkpoint = "data/models/jaad/TrajectoryTransformerb/20Nov2024-10h36m17s_BE2"
            #checkpoint = "data/models/jaad/TrajectoryTransformerb/20Nov2024-10h01m55s_BE1"
            #checkpoint = "data/models/jaad/TrajectoryTransformerb/13Oct2024-15h45m56s_TJ7"
        pretrained_model = TrajectoryTransformerModel.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_trajectory_predictor,
            ignore_mismatched_sizes=True,
            dataset_name=self._dataset)
        
        # Make all layers untrainable
        for child in pretrained_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.traj_class_TF = pretrained_model

        #self.lstm = nn.LSTM(enc_in, hidden_size, layers, batch_first=True) # bidirectional=True)
        self.dropout = nn.Dropout(p=DROPOUT)
        # self.dropout_van = nn.Dropout(p=0.5)

        self.self_attention = SelfAttention(self.max_classifier_hidden_size)
        self.self_attention_van = SelfAttention(self.max_classifier_hidden_size_van)
        # self.self_attention_ctx = SelfAttention(self.combined_classifier_hidden_size)
        self.van_output_embed = nn.Linear(self.max_classifier_hidden_size_van*2, 40)
        #self.van_output_embed.weight.data.fill_(0.0)
        #self.van_output_embed.weight.data.fill_(0.0)
        
        self.fc1 = nn.Linear(self.fc1_neurons, self.fc2_neurons)
        self.fc2 = nn.Linear(self.fc2_neurons, self.num_labels)

        self.fc_model1_embed = nn.Linear(40, 2)
        self.fc_model2_embed = nn.Linear(40, 2)
        # self.gate = EnsembleSelector() # BinaryModelSelectorGumbel() # LearnableGate() # (40)
        # return self.gate(output1, output2)

        #self.projection_lstm = nn.Linear(2 * timeseries_config.num_class, timeseries_config.num_class)
        #self.projection_context = nn.Linear(1 * 512, timeseries_config.num_class) # 2 * 552
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

        add_van_context_combined = False
        add_previous_context = True
        use_separate_tf_for_pose = False
        use_pose_graph_in_ctx = False
        combine_ctx_with_attention = False
        combine_branches_with_attention = False
        combine_branches_with_attention_van = False
        return_dict = self._on_entry(output_hidden_states, return_dict)
        self._debug_video_values(debug, video_values)
        num_labels = 2 # self.num_labels

        #nn.init.constant_(self.van_output_embed.weight, 0.0)
        #nn.init.constant_(self.van_output_embed.bias, 0.0)

        # ====================================================================================
        # Apply VAN model to image context ===================================================

        if self.use_separate_vans:
            # Get timeseries model output
            output1 = self.van1(
                image_context,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output1_tensor = output1.pooler_output if return_dict else output1 # output[1]
            van_output = output1_tensor # self.dropout1(output1_tensor)

            if add_previous_context:
                # Get previous context output from VAN
                output2 = self.van2(
                    previous_image_context,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                output2_tensor = output2.pooler_output if return_dict else output2
                van_output_prev = output2_tensor # self.dropout2(output2_tensor)

            if combine_branches_with_attention_van:
                tuple_to_concat = [van_output_prev, van_output]
                original_x = torch.cat(tuple_to_concat, dim=1) # shape=(batch, combined_fc_len)
                
                van_output_cat = self._concatenate_with_attention(self.self_attention_van, 
                        original_x, tuple_to_concat, 
                        self.max_classifier_hidden_size_van)
                van_output_cat = self.van_output_embed(van_output_cat)
            else:
                van_output_cat = torch.cat([van_output_prev,
                                            van_output], dim=1)
                van_output_cat = self.van_output_embed(van_output_cat)
                #van_output_cat = self.dropout(nn.ReLU()(van_output_cat))
                #van_output_cat = self.dropout_van(van_output_cat)
            
        else:
            output1 = self.van_multiscale(
                image_context,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            output1_tensor = output1.pooler_output if return_dict else output1 # output[1]
            van_output_cat = output1_tensor # self.dropout1(output1_tensor)


        # ====================================================================================
        # Apply TF to trajectory values ======================================================

        outputs_pred = self.traj_class_TF(
            trajectory_values=trajectory_values,
            normalized_trajectory_values=normalized_trajectory_values
        )

        # Concatenate trajectory and context branches ========================================

        # van_output_projected = self._project(van_output, self.projection_van)

        # Add branch-based self-attention ====================================================
        if combine_branches_with_attention:
            tuple_to_concat = [outputs_pred, van_output_cat]
            original_x = torch.cat(tuple_to_concat, dim=1) # shape=(batch, combined_fc_len)
            
            x = self._concatenate_with_attention(self.self_attention, 
                    original_x, tuple_to_concat, 
                    self.max_classifier_hidden_size)
        else:
            test = 10
            tuple_to_concat = [outputs_pred, van_output_cat]
            x = torch.cat(tuple_to_concat, dim=1)
            #x = self.gate(outputs_pred, van_output_cat)
            #x = self.gate(self.fc_model1_embed(outputs_pred), 
            #              self.fc_model2_embed(van_output_cat))

        
        # Apply fully-connected layers =======================================================
    

        outputs = self.dropout(nn.ReLU()(self.fc1(x)))
        logits = self.fc2(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            num_labels,
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

class LearnableGate(nn.Module):
    def __init__(self): # , output_dim):
        super(LearnableGate, self).__init__()
        # self.gate = nn.Parameter(torch.ones(output_dim) * 0.5)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, output1, output2):

        # Hard binary selection with straight-through estimator
        #select_model1 = torch.round(prob)
        #select_model1 = select_model1.detach() + prob - prob.detach()
        # probs = torch.sigmoid(logits)  # Use probabilities without Gumbel noise
        prob = torch.sigmoid(self.gate)
        
        # Soft selection between output1 and output2
        output = prob * output1 + (1 - prob) * output2
        return output
    
        #weights = torch.sigmoid(self.gate)
        #return weights * output1 + (1 - weights) * output2
        # return select_model1 * output1 + (1 - select_model1) * output2


class BinaryModelSelectorGumbel(nn.Module):
    """ Works on jaad_all, but not on pie
    """
    def __init__(self, tau=0.1): # , output_dim):
        super(BinaryModelSelectorGumbel, self).__init__()
        # self.gate = nn.Parameter(torch.ones(output_dim) * 0.5)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.tau = tau

    def forward(self, output1, output2):

        # Compute logits for binary decision
        logits = torch.stack([self.gate, 1 - self.gate], dim=0)

        if CURRENT_EPOCH < 3:

            # Update tau with annealing over time
            initial_tau = 1
            self.tau = max(0.01, initial_tau * (0.8 ** CURRENT_EPOCH))  # Exponential decay

            # Apply Gumbel-Softmax
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
            probs = torch.softmax((logits + gumbel_noise) / self.tau, dim=0)
            # Choose model based on the max probability
            
            output = probs[0] * output1 + probs[1] * output2
            return output
            #select_model1 = (probs[0] > probs[1]).float()
            #return select_model1 * output1 + (1 - select_model1) * output2
        else:
            # Hard selection after 3 epochs
            probs = torch.softmax(logits, dim=0)  # Use probabilities without Gumbel noise
            select_model1 = (probs[0] > probs[1]).float()  # Hard decision
            output = select_model1 * output1 + (1 - select_model1) * output2
            return output


class EnsembleSelector(nn.Module):
    """ Works on jaad_all, but not on pie
    """
    def __init__(self):
        super(EnsembleSelector, self).__init__()

    def forward(self, output1, output2):
        output = 0.5 * output1 + (1 - 0.5) * output2
        return output


class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_experts)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        # Apply softmax to get a probability distribution (though we use argmax for selection)
        return logits  # Logits directly, no softmax, as we'll select with argmax


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MixtureOfExperts, self).__init__()

        # Define the gating network
        self.gating_network = GatingNetwork(input_dim, num_experts=2)

    def forward(self, output1, output2):

        # Get the gating logits (before softmax)
        gating_logits = self.gating_network(x)
        
        # Select the index of the expert with the highest logit
        selected_expert = torch.argmax(gating_logits, dim=-1)  # Get the index of the selected expert
        
        # Create a one-hot encoding of the selected expert
        one_hot_selection = F.one_hot(selected_expert, num_classes=2).float()  # 2 classes (two experts)
        
        # Weighted sum of the expert outputs based on the selection (but only one expert is active)
        output = one_hot_selection[:, 0].view(-1, 1) * output1 + one_hot_selection[:, 1].view(-1, 1) * output2
        return output


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


def get_number_of_training_steps(data_train, train_opts):
    total_devices = 1 # self.hparams.n_gpus * self.hparams.n_nodes
    train_batches = math.ceil((data_train["count"]["neg_count"]+data_train["count"]["pos_count"]) / train_opts["batch_size"])
    train_batches = train_batches // total_devices
    train_steps = train_opts["epochs"] * train_batches # // self.hparams.accumulate_grad_batches
    return train_steps
