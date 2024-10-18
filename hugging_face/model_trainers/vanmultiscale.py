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
from transformers import VanModel, TrainingArguments, Trainer, TimesformerConfig
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

import matplotlib.pyplot as plt 

from hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model, HuggingFaceTimeSeriesClassificationModel, TimeSeriesLibraryConfig
from hugging_face.model_trainers.trajectorytransformer import VanillaTransformerForForecast
from hugging_face.model_trainers.trajectorytransformerb import get_config_for_trajectory_pred
from hugging_face.utilities import compute_loss, get_class_labels_info, get_device
from hugging_face.utils.create_optimizer import get_optimizer
from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
from libs.time_series_library.models_tsl.TrajContextTF import Model as TrajContextTransformerTSLModel
from models.custom_layers_pytorch import SelfAttention

NET_INNER_DIM = 40
DROPOUT = 0.1

class VanMultiscale(HuggingFaceTimeSeriesClassificationModel):
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

        
        # Get model
        encoder_input_size = 5
        seq_len = 15
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        config_for_timeseries_lib = get_config_for_timeseries_lib(
            encoder_input_size, seq_len, hyperparams)
        #config_for_context_timeseries = get_config_for_context_timeseries(encoder_input_size, context_len, hyperparams,
        #                                                                  add_graph_context_to_lstm=add_graph_context_to_lstm)
        config_for_huggingface = TimeSeriesTransformerConfig()
        # config_for_huggingface.num_labels = 2

        model = MultiTransformerForClassification(config_for_huggingface, 
                                              config_for_timeseries_lib,
                                              None, # config_for_context_timeseries,
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
            learning_rate=train_opts["lr"],
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs = epochs,
            warmup_ratio=warmup_ratio,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1 # added
        )

        #optimizer, lr_scheduler = get_optimizer(self, model, args, 
        #    train_dataset, val_dataset, data_train, train_opts, warmup_ratio)

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=None,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn,
            #optimizers=(optimizer, lr_scheduler)
        )

        # Train model
        print("Starting training of model Vanilla Transformer ===========================")
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
            #ignore_sem_map=True,
            #complete_data=complete_data
        )


class VanMultiscaleModel(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib=None,
                 config_for_context_timeseries=None,
                 class_w=None,
                 dataset_statistics=None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self._dataset = "pie"

        # self.class_w = torch.tensor([0.7484, 0.2516]).to(self._device)
        self.dataset_statistics = dataset_statistics
        self.class_w = torch.tensor(class_w).to(self._device) if class_w else None
        self.num_labels = config_for_huggingface.num_labels
        self.timeseries_config = config_for_timeseries_lib #[1]

        # MODEL PARAMETERS ==========================================

        # Classifier head
        self.van_output_size = 512
        self.ctx_output_size = 40

        # MODEL LAYERS ===============================================

        # self.classification_tf = ClassificationTransformerModel(config_for_huggingface, self.timeseries_config)

        # Get pretrained VAN Model -------------------------------------------
        #checkpoint1 = "data/models/jaad/VAN/29Sep2024-11h50m02s_VA1"
        #checkpoint1 = "data/models/jaad/VAN/06Oct2024-18h14m58s_VA4"
        checkpoint1 = "data/models/jaad/VAN/12Oct2024-20h59m24s_VA6"
        #checkpoint2 = "data/models/jaad/VAN/15Jul2023-13h22m25s_EL7"
        #checkpoint2 = "data/models/jaad/VAN/01Oct2024-20h01m53s_VA2"  
        #checkpoint2 = "data/models/jaad/VAN/01Oct2024-22h44m37s_VA3"
        #checkpoint2 = "data/models/jaad/VAN/06Oct2024-22h01m30s_VA5"
        checkpoint2 = "data/models/jaad/VAN/12Oct2024-23h06m29s_VA7"
              
        label2id, id2label = get_class_labels_info()
        pretrained_model = VanModel.from_pretrained(
            checkpoint1,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)
        
        # Make all layers untrainable
        #for child in pretrained_model.children():
        #    for param in child.parameters():
        #        param.requires_grad = False
        self.van1 = pretrained_model

        pretrained_model = VanModel.from_pretrained(
            checkpoint2,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)
        
        # Make all layers untrainable
        #for child in pretrained_model.children():
        #    for param in child.parameters():
        #        param.requires_grad = False
        self.van2 = pretrained_model

        # Get context transformer
        #self.context_transformer = VanillaTransformerCrossAttnModel2(
        #    config_for_huggingface, config_for_context_timeseries)

        self.max_classifier_hidden_size = 512
        self.self_attention = SelfAttention(self.max_classifier_hidden_size)

        nb_fc1_neurons = int(2 * 1024)
        self.fc1 = nn.Linear(nb_fc1_neurons, self.num_labels)

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
        return_logits: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:
        
        #_debug_video_values(image_context)
        #_debug_video_values(previous_image_context)
        
        add_van_context = True
        add_graph_context = False
        merge_in_encoder_nn = False
        add_ctx_to_seq_elements = False
        combine_branches_with_attention = True
        return_dict = self._on_entry(output_hidden_states, return_dict)

        # ====================================================================================
        # Apply VAN model to image context ===================================================

        output1 = self.van1(
            image_context,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output1_tensor = output1.pooler_output if return_dict else output1 # output[1]

        output2 = self.van2(
            previous_image_context,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        output2_tensor = output2.pooler_output if return_dict else output2 # output[1]
        
        if combine_branches_with_attention:
            tuple_to_concat = [output1_tensor, output2_tensor]
            original_x = torch.cat(tuple_to_concat, dim=1) # shape=(batch, combined_fc_len)
            
            outputs = self._concatenate_with_attention(self.self_attention, 
                    original_x, tuple_to_concat, 
                    self.max_classifier_hidden_size)
        else:
            outputs = torch.cat([output1_tensor,
                                output2_tensor], dim=1)
            #outputs = output1_tensor

        logits = self.fc1(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict)
 
    def _project(self, output, projection):
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = projection(output)  # (batch_size, num_classes)
        return output

    def _on_entry(self, output_hidden_states, return_dict):
        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        self.return_dict = return_dict
        return return_dict
    
    def _concatenate_with_attention(self, self_attention, original_x, 
                                    tuple_to_concat, max_concat_size):
        
        # Concatenate outputs with padding
        tuple_to_concat = self._pad_tensors(tuple_to_concat, max_concat_size)
        x = torch.cat(tuple_to_concat, dim=1) # shape=(batch, nb_models, max_concat_size)

        attention_ctx, attn = self_attention(x)
        attention_ctx = attention_ctx.squeeze(2)
        x = torch.cat((original_x, attention_ctx), dim=1)

        return x

    def _pad_tensors(self, tensors: list, max_size: int, pad_dim=1):
        for idx, output_tensor in enumerate(tensors):
            tensors[idx] = F.pad(output_tensor, pad=(0, max_size - output_tensor.shape[pad_dim], 0, 0)) # shape=(batch, fc_len)
            if len(tensors[idx].shape) <= 2:
                tensors[idx] = tensors[idx].unsqueeze(1) # shape=(batch, X, fc_len)
        return tensors
    

class VANMultiscaleForClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib=None,
                 config_for_abs_timeseries=None,  
                 # dataset_statistics=None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self.num_labels = config_for_huggingface.num_labels

        self.van_multiscale = VanMultiscaleModel(
            config_for_huggingface, config_for_timeseries_lib,
            config_for_abs_timeseries=config_for_abs_timeseries)

        classifier_hidden_size = config_for_timeseries_lib.num_class # number of neurons in last Linear layer at the end of model
        self.classifier = nn.Linear(
            classifier_hidden_size, config_for_huggingface.num_labels) \
            if config_for_huggingface.num_labels > 0 else nn.Identity()

        self.timeseries_config = config_for_timeseries_lib

        self.fc1 = nn.Linear(40, self.num_labels)

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
        return_logits: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:

        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get trajectory transformer model output
        outputs = self.van_multiscale(
            trajectory_values,
            normalized_trajectory_values # predicted_trajectory
        )

        logits = self.fc1(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict)


# config for classification_TF
def get_config_for_timeseries_lib(encoder_input_size, seq_len,
                                  hyperparams, pred_len=None):
    encoder_input_size = encoder_input_size + 1
    if hyperparams:
        hyperparams = hyperparams["MultiTransformerForClassificationClassification"]

    # time series lib properties
    time_series_dict = {
        "task_name": "classification",
        "pred_len": pred_len if pred_len else 60,
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": encoder_input_size, # encoder input size - default value,
        "d_model": hyperparams.get("d_model", 128), # dimension of model - default value 
        "embed": "learned", # time features encoding; note: not used in classification task by vanilla transformer model
        "freq": "h", # freq for time features encoding; note: not used in classification task by vanilla transformer model
        "dropout": 0.1, # default,
        "factor": 1, # attn factor; note: not used by vanilla transformer model
        "n_heads": hyperparams.get("n_heads", 12), # num of heads
        "d_ff": hyperparams.get("d_ff", 1024), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 6), # num of encoder layers (or 3)
        "seq_len": seq_len, # input sequence length
        "num_class": 40, # number of neurons in last Linear layer at the end of model
        # ---------------------------------------------------------------------------------
        "label_len": 15, # section shared by encoder and decoder
        "num_kernels": 6, # Timesnet - for Inception
        "top_k": 5, # Timesnet - for TimesBlock
        "moving_avg": 3, # FEDformer - window size of moving average, default=25
        "dec_in": 5, # FEDformer - decoder input size
        "d_layers": 2, # FEDformer - num of decoder layers
        "c_out": 5,
        "distil": True, # Informer - whether to use distilling in encoder, using this argument means not using distilling
        "p_hidden_dims": [128, 128], # Nonstationary transformer - hidden layer dimensions of projector (List)
        "p_hidden_layers": 2, # Nonstationary transformer - number of hidden layers in projector
        # "num_kernels": 3, # override - Pyraformer
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    return config_for_timeseries_lib


def get_number_of_training_steps(data_train, train_opts):
    total_devices = 1 # self.hparams.n_gpus * self.hparams.n_nodes
    train_batches = math.ceil((data_train["count"]["neg_count"]+data_train["count"]["pos_count"]) / train_opts["batch_size"])
    train_batches = train_batches // total_devices
    train_steps = train_opts["epochs"] * train_batches # // self.hparams.accumulate_grad_batches
    return train_steps

def _debug_video_values(video_values):
    cpu_array = video_values[1,...].permute(1, 2, 0).cpu()
    #plt.imshow(cpu_array)
    np_array = cpu_array.detach().numpy()
    cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}_1.png", np_array)