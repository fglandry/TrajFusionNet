from typing import Any, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary
from transformers import TrainingArguments, Trainer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel

from models.hugging_face.model_trainers.trajectorytransformerb import load_pretrained_encoder_transformer
from models.hugging_face.model_trainers.van import load_pretrained_van
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TorchTimeseriesDataset
from models.hugging_face.utilities import compute_loss, get_device
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
              generator: bool = False,
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
            batch_size [int]: batch size
            epochs [int]: number of epochs
            model_path [str]: path where the model will be saved
            generator [bool]: specifies if a generator is used to load data
            train_opts [str]: training options (includes learning rate)
            dataset_statistics [dict]: contains dataset statistics such as avg / std dev per feature
            hyperparams [dict]: dict containing hyperparameters to use, if enabled
            class_w [list]: class weights
            test_only [bool]: is set to True, model will not be trained, only tested
            train_end_to_end [bool]: if True, all modules in the network will be trained (see modular
                                     training section in the paper)
            submodels_paths [dict]: dictionary containing paths to submodels saved on disk
        """

        print("Starting model loading for model TrajFusionNet ===========================")

        # Get hyperparameters (if specified) and model configs
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        config_for_huggingface = TimeSeriesTransformerConfig()
        self.class_w = class_w

        # If training end-to-end, start by training submodels
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
            dataset_statistics=dataset_statistics)

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
                val_dataset, data_train, train_opts
            )
     
        return {
            "trainer": trainer,
            "val_transform": val_transforms_dicts
        }
    
    def train_with_initial_vam_branch_disabling(self,
            model: Any, epochs: int, 
            args: TrainingArguments, train_dataset: TorchTimeseriesDataset, 
            val_dataset: TorchTimeseriesDataset, data_train: dict, train_opts: dict):
        
        best_metric = 0
        best_trainer = None
        half_epochs = round(epochs / 2)
        
        # Run first part of training procedure with the VAM branch disabled for 15 epochs
        # to improve learning in the SAM branch.
        # In order to do this, the weights in the VAM projection layer ('van_output_embed')
        # as well as the associated learning rate are set to zero
        with torch.no_grad(): 
            model.van_output_embed.weight.zero_()
            model.van_output_embed.bias.zero_()

        for i in range(half_epochs):
            
            # Get custom optimizer to set learning rate to zero in the VAM projection layer
            optimizer, lr_scheduler = get_optimizer(self, model, args, 
                train_dataset, val_dataset, data_train, train_opts,
                disable_vam_branch=True, nb_epochs_disabled=15, epoch_index=i+1)
            
            trainer = self._get_trainer(model, args, train_dataset, 
                                        val_dataset, optimizer, lr_scheduler)
            trainer.args.num_train_epochs = 1
            
            trainer.train()

            if trainer.state.best_metric > best_metric:
                best_trainer = trainer
                best_metric = trainer.state.best_metric

        # Run second part of training procedure with the VAM branch re-enabled       
        optimizer, lr_scheduler = get_optimizer(self, model, args, 
            train_dataset, val_dataset, data_train, train_opts,
            disable_vam_branch=False) # learning rate of the VAM projection layer is
                                      # reset to the global learning rate

        trainer = self._get_trainer(model, args, train_dataset, 
                                    val_dataset, optimizer, lr_scheduler)
        trainer.args.num_train_epochs = half_epochs

        trainer.train()

        if trainer.state.best_metric > best_metric:
            best_trainer = trainer
            best_metric = trainer.state.best_metric if trainer.state.best_metric else 0

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
         "-d", dataset, "-j", submodels_paths['traj_tf_path']])

    # VAM module ===============================================================
    
    # Train VAN with image context at time t and predicted trajectory overlays
    van_path = run_and_capture_model_path(
        ["python3", "train_test.py", "-c", "config_files/VAN.yaml", 
         "-d", dataset, "-j", submodels_paths['traj_tf_path']])
    
    # Train VAN with image context at time t-15 and observed trajectory overlays
    van_prev_path = run_and_capture_model_path(
        ["python3", "train_test.py", "-c", "config_files/VAN_previous.yaml", 
         "-d", dataset, "-j", submodels_paths['traj_tf_path']])
    
    submodels_paths.update({
        "enc_tf_path": enc_tf_path,
        "van_path": van_path,
        "van_prev_path": van_prev_path
    })

    return submodels_paths

def load_pretrained_trajfusionnet(dataset_name: str):
    if dataset_name in ["pie", "combined"]:
        checkpoint = "data/models/pie/TrajFusionNet/weights_trajfusionnet_pie"
    elif dataset_name == "jaad_all":
        checkpoint = "data/models/jaad_all/TrajFusionNet/weights_trajfusionnet_jaadall"
    elif dataset_name == "jaad_beh":
        checkpoint = "data/models/jaad_beh/TrajFusionNet/weights_trajfusionnet_jaadbeh"
        
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
