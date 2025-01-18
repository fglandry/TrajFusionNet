from typing import Optional, Tuple, Union

import torch
from torch import nn
from torchsummary import summary
from transformers import TrainingArguments, Trainer, TimesformerConfig
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from models.hugging_face.model_trainers.smalltrajectorytransformer import VanillaTransformerForForecast, get_config_for_timeseries_lib as get_config_for_trajectory_pred
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig
from models.hugging_face.utilities import compute_loss, get_device
from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel

PRED_LEN = 60


class SmallTrajectoryTransformerb(HuggingFaceTimeSeriesModel):

    def train(self,
              data_train, 
              data_val,
              batch_size, 
              epochs,
              model_path,  
              generator=False,
              train_opts=None,
              hyperparams=None,
              dataset_statistics=None,
              class_w=None,
              *args, **kwargs):
        print("Starting model loading for model Trajectory Transformer ===========================")

        # Get parameters used by time series library model
        seq_len = 15 + PRED_LEN
        
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        hyperparam_vals = hyperparams["VanillaTransformerForForecastClassification"] if hyperparams else {}
        lr = hyperparam_vals.get("lr", train_opts["lr"])
        batch_size = hyperparam_vals.get("batch_size", batch_size)
        epochs = hyperparam_vals.get("epochs", epochs)
        
        config_for_timeseries_lib = get_config_for_timeseries_lib(
            5, seq_len, hyperparams)
        config_for_huggingface = TimeSeriesTransformerConfig()

        self.num_labels = config_for_huggingface.num_labels

        model = VanillaTransformerForForecastClassification(
            config_for_huggingface, config_for_timeseries_lib,
            config_for_abs_timeseries=None,
            class_w=class_w,
            dataset_statistics=dataset_statistics,
            dataset_name=kwargs["model_opts"]["dataset_full"]
        )
        summary(model)

        # Get datasets
        video_model_config = TimesformerConfig()
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, video_model_config,
            get_image_transform=False, img_model_config=None,
            dataset_statistics=dataset_statistics)

        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1, # added
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=None,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn
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
             *args,
             generator=False,
             **kwargs):
        
        print("Starting inference using trained model Vanilla Transformer ===========================")

        return test_time_series_based_model(
            test_data,
            training_result,
            model_path,
            generator
        )


class TrajectoryTransformerModel(TimeSeriesTransformerPreTrainedModel):
    
    base_model_prefix = "transformer" # needs to be a class property
    
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib,
                 config_for_abs_timeseries = None,
                 dataset_name=None
        ):
        super().__init__(config_for_huggingface)
        self._dataset = dataset_name
        self._device = get_device()
        self.tsl_transformer = VanillaTransformerTSLModel(config_for_timeseries_lib)
        if config_for_abs_timeseries:
            self.tsl_transformer_abs = VanillaTransformerTSLModel(config_for_abs_timeseries)

        # Get pretrained trajectory predictor -------------------------------------------
        config_for_trajectory_predictor = get_config_for_trajectory_pred(
            encoder_input_size=5, seq_len=15, hyperparams={}, pred_len=60)
        if self._dataset in ["pie", "combined"]:
            checkpoint = "data/models/pie/SmallTrajectoryTransformer/17Nov2024-17h45m15s_SM4"
        elif self._dataset == "jaad_all":
            checkpoint = "data/models/jaad/SmallTrajectoryTransformer/16Nov2024-22h12m21s_SM1"
        elif self._dataset == "jaad_beh":
            checkpoint = "data/models/jaad/SmallTrajectoryTransformer/16Nov2024-22h12m21s_SM1"
        pretrained_model = VanillaTransformerForForecast.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_trajectory_predictor,
            ignore_mismatched_sizes=True)
        
        # Make all layers untrainable
        for child in pretrained_model.children():
            for param in child.parameters():
                param.requires_grad = False
        self.traj_TF = pretrained_model

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values,
        normalized_trajectory_values,
        *args,
        **kwargs
    ):
        normalized_abs_box_in_values = False
        pose_in_values = False

        # Trajectory prediction ====================================================

        if pose_in_values:
            traj_predictor_input = torch.cat((normalized_trajectory_values[:,:,0:4], 
                                              normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
            normalized_trajectory_values = torch.cat((normalized_trajectory_values[:,:,0:4], 
                                                      normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
        elif normalized_abs_box_in_values:
            traj_predictor_input = torch.cat((normalized_trajectory_values[:,:,0:4], 
                                              normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
            abs_classifier_input = normalized_trajectory_values[:,:,4:8]
            normalized_trajectory_values = torch.cat((normalized_trajectory_values[:,:,0:4], 
                                                      normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
        else:
            traj_predictor_input = normalized_trajectory_values
        
        # Predict trajectory between TTE=0 and TTE=60
        predicted_trajectory = self.traj_TF(
             normalized_trajectory_values=traj_predictor_input
        ).logits

        # Crossing prediction ====================================================

        # Add element to indicate predicted vs past trajectory

        extra_dim_vals = torch.zeros((predicted_trajectory.shape[0],          # add 0
                            predicted_trajectory.shape[1], 1)).to(self._device)
        predicted_trajectory = torch.cat((predicted_trajectory, extra_dim_vals), 2)
        extra_dim_vals = torch.ones((normalized_trajectory_values.shape[0],   # add 1
                            normalized_trajectory_values.shape[1], 1)).to(self._device)
        trajectory_values = torch.cat((normalized_trajectory_values, extra_dim_vals), 2)



        # Only keep values found in prediction horizon (last 1 second)
        predicted_trajectory = torch.cat([trajectory_values,
                                          predicted_trajectory], dim=1)

        # Get vanilla transformer model output
        outputs = self.tsl_transformer(
            x_enc=predicted_trajectory,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )

        if normalized_abs_box_in_values:
            # Get abs transformer model output
            outputs_abs = self.tsl_transformer_abs(
                x_enc=abs_classifier_input,
                x_mark_enc=None,
                x_dec=None,
                x_mark_dec=None
            )

            cat_outputs = torch.cat([outputs, 
                                     outputs_abs], dim=1)
            return cat_outputs

        return outputs
        

class VanillaTransformerForForecastClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib=None,
                 config_for_abs_timeseries=None,
                 class_w=None,
                 dataset_statistics=None,  
                 dataset_name=None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self._dataset = dataset_name

        self.dataset_statistics = dataset_statistics
        self.class_w = torch.tensor(class_w).to(self._device)
        self.num_labels = config_for_huggingface.num_labels

        self.transformer = TrajectoryTransformerModel(config_for_huggingface, config_for_timeseries_lib,
                                                      config_for_abs_timeseries=config_for_abs_timeseries,
                                                      dataset_name=self._dataset)

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
        normalized_trajectory_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:

        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get trajectory transformer model output
        outputs = self.transformer(
            trajectory_values,
            normalized_trajectory_values # predicted_trajectory
        )

        logits = self.fc1(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict,
                            class_w=self.class_w)


def get_config_for_timeseries_lib(encoder_input_size, seq_len,
                                  hyperparams, pred_len=None):
    encoder_input_size = encoder_input_size + 1
    if hyperparams:
        hyperparams = hyperparams["VanillaTransformerForForecastClassification"]

    # time series lib properties
    time_series_dict = {
        "task_name": "classification",
        "pred_len": pred_len if pred_len else PRED_LEN,
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": encoder_input_size, # encoder input size - default value,
        "d_model": hyperparams.get("d_model", 128), # dimension of model - default value 
        "embed": "learned", # time features encoding; note: not used in classification task by vanilla transformer model
        "freq": "h", # freq for time features encoding; note: not used in classification task by vanilla transformer model
        "dropout": 0.1, # default,
        "factor": 1, # attn factor; note: not used by vanilla transformer model
        "n_heads": hyperparams.get("n_heads", 2), # num of heads
        "d_ff": hyperparams.get("d_ff", 256), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 2), # num of encoder layers (or 3)
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
