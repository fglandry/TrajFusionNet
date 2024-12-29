import copy
from typing import Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from transformers import VanModel, TrainingArguments, Trainer, TimesformerConfig
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from models.hugging_face.model_trainers.trajectorytransformer import VanillaTransformerForForecast, get_config_for_timeseries_lib as get_config_for_trajectory_pred
from models.hugging_face.model_trainers.trajectorytransformerb import load_pretrained_encoder_transformer
from models.hugging_face.model_trainers.van import load_pretrained_van, VAN
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model, normalize_trajectory_data, denormalize_trajectory_data, HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig, TorchTimeseriesDataset
from models.hugging_face.utilities import compute_loss, get_class_labels_info, get_device
from models.hugging_face.utils.create_optimizer import get_optimizer
from models.custom_layers_pytorch import SelfAttention
from utils.data_load import DataGenerator
from utils.action_predict_utils.run_in_subprocess import run_and_capture_model_path

NET_INNER_DIM = 512
NET_OUTER_DIM = 40
DROPOUT = 0.1


class TrajFusionNet(HuggingFaceTimeSeriesModel):

    def train(self,
              data_train: dict,  
              data_val: DataGenerator,
              batch_size: int,
              epochs: int,
              model_path: str,   
              generator=False,
              train_opts: dict = None,  
              dataset_statistics: dict = None,
              hyperparams: dict = None,
              class_w: list = None,
              test_only: bool = False,
              train_end_to_end: bool = False,
              submodels_paths: dict = None,
              *args, **kwargs):
        """ Train model
        Args:
            data_train [dict]: training data (data_train['data'][0] contains the generator)
            data_val [DataGenerator]: validation data
            model_path [str]: path where the model will be saved
            train_opts [str]: training options (includes learning rate)
            dataset_statistics [dict]: contains dataset statistics such as avg / std dev per feature
            class_w [list]: class weights
            test_only [bool]: is set to True, model will not be trained, only tested
        """

        print("Starting model loading for model TrajFusionNet ===========================")

        """
        submodels_paths = {
            "traj_tf_path": "data/models/jaad/TrajectoryTransformer/05Oct2024-11h30m11s_TE24",
            "enc_tf_path": "data/models/jaad/TrajectoryTransformerb/20Nov2024-12h02m51s_TJ8",
            "van_path": "data/models/jaad/VAN/25Dec2024-11h25m13s_VA6B",
            "van_prev_path": "data/models/jaad/VAN/25Dec2024-13h28m35s_VA7B"
        }
        """

        # Get hyperparameters (if specified) and model configs
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        config_for_huggingface = TimeSeriesTransformerConfig()
        self.class_w = class_w

        # If specified, start by training submodels first 
        if train_end_to_end:
            submodels_paths = train_submodels(
                dataset=kwargs["model_opts"]["dataset_full"],
                submodels_paths=submodels_paths)

        model = TrajFusionNetForClassification(config_for_huggingface, 
                                               class_w=class_w,
                                               dataset_statistics=dataset_statistics,
                                               dataset_name=kwargs["model_opts"]["dataset_full"],
                                               submodels_paths=submodels_paths)
        summary(model)

        # Get datasets
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, None,
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
            num_train_epochs = epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1
        )
        
        if test_only:
            optimizer, lr_scheduler = get_optimizer(self, model, args, 
                    train_dataset, val_dataset, data_train, train_opts)
            trainer = self._get_trainer(model, args, train_dataset, 
                                        val_dataset, optimizer, lr_scheduler)
        else:
            # Train model
            print("Starting training of model TrajFusionNet ===========================")
            trainer = self.train_with_initial_vam_branch_disabling(
                model, epochs, args, train_dataset,
                val_dataset, data_train, train_opts,
                submodels_paths=submodels_paths
            )
     
        return {
            "trainer": trainer,
            "val_transform": val_transforms_dicts
        }
    
    def train_with_initial_vam_branch_disabling(self,
            model, epochs, args, train_dataset, 
            val_dataset, data_train, train_opts,
            disable_vam_branch_initially=True,
            submodels_paths=None):
        
        initial_weights = copy.deepcopy(model.state_dict())

        # Run first part of training procedure
        best_metric = 0
        best_trainer = None
        best_scenario_in_second_training = False

        optimizer, lr_scheduler = get_optimizer(self, model, args, 
            train_dataset, val_dataset, data_train, train_opts,
            disable_vam_branch=False)
        trainer = self._get_trainer(model, args, train_dataset, 
                                     val_dataset, optimizer, lr_scheduler)
        
        trainer.train()
        best_trainer = trainer
        best_metric = trainer.state.best_metric if trainer.state.best_metric else 0
        print(f"Best AUC metric in first part of training: {best_trainer.state.best_metric}")
        
        
        # Run second part of training procedure
        if disable_vam_branch_initially:
            # Reset weights and keep training model with learning rate of VAM
            # branch set to 0 + weights of embed layer set to 0, during first 10 epochs
            model.load_state_dict(initial_weights)
            with torch.no_grad(): 
                model.van_output_embed.weight.zero_()
                model.van_output_embed.bias.zero_()
            # args.output_dir = model_path[:-1] + "-part2"

            for i in range(epochs):
                
                # Get custom optimizer to disable learning in the projection layer of
                # the VAM branch for the first 5 epochs
                optimizer, lr_scheduler = get_optimizer(self, model, args, 
                    train_dataset, val_dataset, data_train, train_opts,
                    disable_vam_branch=True, epoch_index=i+1)
                
                trainer = self._get_trainer(model, args, train_dataset, 
                                             val_dataset, optimizer, lr_scheduler)
                trainer.args.num_train_epochs = 1
                
                trainer.train()

                # metric = best_trainer.state.best_metric if best_trainer.state.best_metric is not None else 0.0
                if i == 10:
                    print("Reached transition epoch")
                if trainer.state.best_metric > best_metric and i != 10: # do not take best checkpoint right when the VAM branch is enabled again
                    best_trainer = trainer
                    best_metric = trainer.state.best_metric
                    best_scenario_in_second_training = True

                print(f"Best AUC metric in second part of training: {trainer.state.best_metric}")
        
            if not best_scenario_in_second_training:
                # The huggingface trainer object forgets the first training after
                # the second training is launched. If better results are obtained
                # in the first training, re-compute that training so that results get
                # stored in the trainer.
                # TODO: find a way to avoid having to do this

                model.load_state_dict(initial_weights)

                optimizer, lr_scheduler = get_optimizer(self, model, args, 
                    train_dataset, val_dataset, data_train, train_opts,
                    disable_vam_branch=False)
                trainer = self._get_trainer(model, args, train_dataset, 
                                            val_dataset, optimizer, lr_scheduler)
                trainer.args.num_train_epochs = epochs
                trainer.train()
                best_trainer = trainer
                best_metric = trainer.state.best_metric

        print(f"Best AUC metric: {best_trainer.state.best_metric}")
        return best_trainer

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
            optimizers=(optimizer, lr_scheduler)
        )
        return trainer

    def test(self,
             test_data: tuple,
             training_result: dict,
             model_info: dict,
             *args,
             dataset_name: str = "",
             generator: bool = False,
             test_only: bool = False,
             complete_data=None,
             **kwargs):
        
        print("Starting inference using trained model ===========================")

        if test_only:
            pretrained_model = load_pretrained_trajfusionnet(dataset_name)
            training_result["trainer"].model = pretrained_model

        return test_time_series_based_model(
            test_data,
            training_result,
            model_info,
            generator
        )


class TrajFusionNetForClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 class_w: list = None,
                 dataset_statistics: dict = None,
                 dataset_name: str = "",
                 submodels_paths: dict = None
        ):
        super().__init__(config_for_huggingface)
        self._device = get_device()
        self._dataset = dataset_name
        self.dataset_statistics = dataset_statistics
        self.class_w = torch.tensor(class_w).to(self._device) if class_w else None
        self.num_labels = config_for_huggingface.num_labels

        self.combine_branches_with_attention = False
        self.combine_vans_with_attention = False

        # MODEL PARAMETERS ==========================================

        # Classifier head parameters
        self.van_output_size = 256
        self.max_classifier_hidden_size = NET_OUTER_DIM
        self.max_classifier_hidden_size_van = NET_INNER_DIM
        self.fc1_neurons = 2 * self.max_classifier_hidden_size
        self.fc2_neurons = NET_OUTER_DIM
        
        # Get pretrained VAN Models -------------------------------------------
        self.van1 = load_pretrained_van( # predicted ped overlays
            dataset_name,
            is_predicted_overlays=True,
            add_classification_head=False,
            submodels_paths=submodels_paths)

        self.van2 = load_pretrained_van( # observed ped overlays
            dataset_name,
            is_predicted_overlays=False,
            add_classification_head=False,
            submodels_paths=submodels_paths)

        # Get pretrained encoder transformer -----------------------------------------
        self.traj_class_TF = load_pretrained_encoder_transformer(dataset_name,
                                                                 add_classification_head=False,
                                                                 submodels_paths=submodels_paths)

        # Classifier layers
        if self.combine_branches_with_attention:
            self.self_attention = SelfAttention(self.max_classifier_hidden_size)
        if self.combine_vans_with_attention:
            self.self_attention_van = SelfAttention(self.max_classifier_hidden_size_van)
        self.van_output_embed = nn.Linear(self.max_classifier_hidden_size_van*2, NET_OUTER_DIM)
        self.dropout = nn.Dropout(p=DROPOUT)
        self.fc1 = nn.Linear(self.fc1_neurons, self.fc2_neurons)
        self.fc2 = nn.Linear(self.fc2_neurons, self.num_labels)

        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        trajectory_values: torch.Tensor = None,
        image_context: torch.Tensor = None,
        previous_image_context: torch.Tensor = None,
        normalized_trajectory_values: torch.Tensor = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        *args, **kwargs
    ):
        """ Args:
        trajectory_values [torch.Tensor]: non-normalized observed trajectory values
            of shape [batch, seq_len, enc]
        normalized_trajectory_values [torch.Tensor]: normalized observed trajectory values
            of shape [batch, seq_len, enc]
        labels [torch.Tensor]: future target trajectory values of shape [batch, pred_len, enc]
            (between time t=0 and time t=60)
        """
        
        return_dict = self._on_entry(output_hidden_states, return_dict)

        # Apply VAN model to image context with predicted pedestrian overlays =========================
        output1 = self.van1(
            image_context,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        van_output = output1.pooler_output if return_dict else output1 # shape=[batch, 512]

        # Apply VAN model to image context at time t-15 with observed pedestrian overlays =============
        output2 = self.van2(
            previous_image_context,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        van_output_prev = output2.pooler_output if return_dict else output2

        if self.combine_vans_with_attention:
            tuple_to_concat = [van_output_prev, van_output]
            original_x = torch.cat(tuple_to_concat, dim=1) # shape=[batch, combined_fc_len]
            
            van_output_cat = self._concatenate_with_attention(self.self_attention_van, 
                    original_x, tuple_to_concat, 
                    self.max_classifier_hidden_size_van)
            van_output_cat = self.van_output_enc(van_output_cat)
        else:
            van_output_cat = torch.cat([van_output_prev, 
                                        van_output], dim=1)
            van_output_cat = self.van_output_embed(van_output_cat) # shape=[batch, 40]
            
        # Apply Encoder TF to trajectory values =============================================

        outputs_pred = self.traj_class_TF(
            trajectory_values=trajectory_values,
            normalized_trajectory_values=normalized_trajectory_values
        )

        # Fuse branches =====================================================================

        if self.combine_branches_with_attention:
            tuple_to_concat = [outputs_pred, van_output_cat]
            original_x = torch.cat(tuple_to_concat, dim=1) # shape=[batch, combined_fc_len]
            
            x = self._concatenate_with_attention(self.self_attention, 
                    original_x, tuple_to_concat, self.max_classifier_hidden_size)
        else:
            tuple_to_concat = [outputs_pred, van_output_cat]
            x = torch.cat(tuple_to_concat, dim=1) # shape=[batch, 80]

        # Apply fully-connected layers
        outputs = self.dropout(nn.ReLU()(self.fc1(x)))
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
        
        # concatenate outputs with padding
        tuple_to_concat = self._pad_tensors(tuple_to_concat, max_concat_size)
        x = torch.cat(tuple_to_concat, dim=1) # shape=[batch, nb_models, max_concat_size]

        attention_ctx, attn = self_attention(x)
        attention_ctx = attention_ctx.squeeze(2)
        x = torch.cat((original_x, attention_ctx), dim=1)

        return x

    def _pad_tensors(self, tensors: list, max_size: int, pad_dim=1):
        for idx, output_tensor in enumerate(tensors):
            tensors[idx] = F.pad(output_tensor, pad=(0, max_size - output_tensor.shape[pad_dim], 0, 0)) # shape=[batch, fc_len]
            if len(tensors[idx].shape) <= 2:
                tensors[idx] = tensors[idx].unsqueeze(1) # shape=[batch, X, fc_len]
        return tensors
    
    def _on_entry(self, output_hidden_states, return_dict):
        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.return_dict = return_dict
        return return_dict

def train_submodels(dataset: str,
                    submodels_paths: dict):

    # SAM module ===============================================================
    
    # Train encoder transformer
    enc_tf_path = run_and_capture_model_path(
        ["python3", "train_test.py", "-c", "config_files/TrajectoryTransformerb.yaml", 
         "-d", dataset])

    # VAM module ===============================================================
    
    # Train VAN with image context at time t and predicted trajectory overlays
    van_path = run_and_capture_model_path(
        ["python3", "train_test.py", "-c", "config_files/VAN.yaml", 
         "-d", dataset])
    
    # Train VAN with image context at time t-15 and observed trajectory overlays
    van_prev_path = run_and_capture_model_path(
        ["python3", "train_test.py", "-c", "config_files/VAN_previous.yaml", 
         "-d", dataset])
    
    submodels_paths.update({
        "enc_tf_path": enc_tf_path,
        "van_path": van_path,
        "van_prev_path": van_prev_path
    })

    return submodels_paths

def load_pretrained_trajfusionnet(dataset_name: str):
    if dataset_name in ["pie", "combined"]:
        checkpoint = "data/models/pie/TrajFusionNet/26Dec2024-11h42m23s_PIE1"
    elif dataset_name == "jaad_all":
        checkpoint = "data/models/jaad/TrajFusionNet/25Dec2024-15h02m22s_ALL1"
    elif dataset_name == "jaad_beh":
        checkpoint = "data/models/jaad/TrajFusionNet/25Dec2024-21h50m11s_BEH1"
        
    pretrained_model = TrajFusionNetForClassification.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes=True,
        dataset_name=dataset_name,
        class_w=None)
    
    # Make all layers untrainable
    for child in pretrained_model.children():
        for param in child.parameters():
            param.requires_grad = False
    return pretrained_model