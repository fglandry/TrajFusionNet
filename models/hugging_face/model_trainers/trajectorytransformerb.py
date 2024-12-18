from typing import Optional, Tuple, Union

import torch
from torch import nn
from torchsummary import summary
from transformers import TrainingArguments, Trainer, TimesformerConfig
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from models.hugging_face.model_trainers.trajectorytransformer import VanillaTransformerForForecast, get_config_for_timeseries_lib as get_config_for_trajectory_pred, load_pretrained_trajectory_transformer
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model, denormalize_trajectory_data
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig, TorchTimeseriesDataset
from models.hugging_face.utilities import compute_loss, get_device
from models.hugging_face.utils.create_optimizer import get_optimizer
from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
from models.custom_layers_pytorch import CrossAttention, SelfAttention


PRED_LEN = 60

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


class TrajectoryTransformerb(HuggingFaceTimeSeriesModel):

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
              *args, **kwargs):
        print("Starting model loading for model Trajectory Transformer ===========================")

        # Get parameters used by time series library model
        data_element = data_train['data'][0][0][0][0] # revisit
        encoder_input_size = data_element.shape[-1]
        seq_len = 15 + PRED_LEN # data_element.shape[-2] # 90
        
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        hyperparam_vals = hyperparams["VanillaTransformerForForecastClassification"] if hyperparams else {}
        lr = hyperparam_vals.get("lr", train_opts["lr"])
        batch_size = hyperparam_vals.get("batch_size", batch_size)
        epochs = hyperparam_vals.get("epochs", epochs)
        
        config_for_timeseries_lib = get_config_for_timeseries_lib(
            5, seq_len, hyperparams)
        config_for_abs_timeseries = get_config_for_timeseries_lib(
            3, 15, hyperparams, pred_len=30)
        config_for_huggingface = TimeSeriesTransformerConfig()
        # config_for_huggingface.problem_type = "trajectory"
        self.num_labels = config_for_huggingface.num_labels

        model = VanillaTransformerForForecastClassification(
            config_for_huggingface, config_for_timeseries_lib,
            config_for_abs_timeseries=None
            # dataset_statistics)
        )
        summary(model)

        # Get datasets
        video_model_config = TimesformerConfig()
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, video_model_config,
            get_image_transform=False, img_model_config=None,
            dataset_statistics=dataset_statistics, ignore_sem_map=True)

        warmup_ratio = 0.1
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

        """
        optimizer, lr_scheduler = get_optimizer(self, model, args, 
            train_dataset, val_dataset, data_train, train_opts, warmup_ratio)
        """

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
                 # dataset_statistics=None
                 dataset_name=None
        ):
        super().__init__(config_for_huggingface)
        self._dataset = dataset_name
        self._device = get_device()
        self.tsl_transformer = VanillaTransformerTSLModel(config_for_timeseries_lib)
        if config_for_abs_timeseries:
            self.tsl_transformer_abs = VanillaTransformerTSLModel(config_for_abs_timeseries)
        self.predict_abs_traj = False
        self.use_cross_attention = False

        self.traj_TF = load_pretrained_trajectory_transformer(dataset_name)


        if self.predict_abs_traj:
            # Get pretrained trajectory predictor for normalized_abs_box --------------------------------
            config_for_trajectory_predictor = get_config_for_trajectory_pred(
                encoder_input_size=5, seq_len=15, hyperparams={}, pred_len=60)
            checkpoint = "data/models/pie/TrajectoryTransformer/25Sep2024-18h59m33s_TE36b" # pred_len = 60, higher loss weight at t=30 and t=60 
            #checkpoint = "data/models/pie/TrajectoryTransformer/25Sep2024-14h49m19s_TE35b" # pred_len = 60, higher loss weight at t=60 
            # checkpoint = "data/models/pie/TrajectoryTransformer/24Sep2024-20h44m11s_TE34" # pred_len = target at t=60
            # checkpoint = "data/models/pie/TrajectoryTransformer/17Sep2024-20h49m51s_TE33" # pred_len = last 15 (out of 75)
            # checkpoint = "data/models/pie/TrajectoryTransformer/15Sep2024-10h58m37s_TE32" # pred_len = last 30
            # checkpoint = "data/models/pie/TrajectoryTransformer/14Sep2024-17h38m26s_TE31" # pred_len = first 30
            # checkpoint = "data/models/pie/TrajectoryTransformer/13Sep2024-21h33m35s_TE28" # normalized_abs_box only

            pretrained_model = VanillaTransformerForForecast.from_pretrained(
                checkpoint,
                config_for_timeseries_lib=config_for_trajectory_predictor,
                ignore_mismatched_sizes=True)
            
            # Make all layers untrainable
            for child in pretrained_model.children():
                for param in child.parameters():
                    param.requires_grad = False
            self.traj_abs_TF = pretrained_model

        if self.use_cross_attention:
            cross_attn_dim = 3
            self.cross_attention_1 = CrossAttention(cross_attn_dim)
            self.cross_attention_2 = CrossAttention(cross_attn_dim)
            self.box_emb_fc = nn.Linear(4, cross_attn_dim)
            self.speed_emb_fc = nn.Linear(1, cross_attn_dim)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values,
        normalized_trajectory_values,
        *args,
        **kwargs
    ):
        predict_after_tte = False
        predict_abs_traj = False
        normalized_abs_box_in_values = False
        pose_in_values = False
        use_cross_attention = False

        # Trajectory prediction ====================================================

        if pose_in_values:
            traj_predictor_input = torch.cat((normalized_trajectory_values[:,:,0:4], 
                                              normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
            pose_classifier_input = normalized_trajectory_values[:,:,4:14]
            normalized_trajectory_values = torch.cat((normalized_trajectory_values[:,:,0:4], 
                                                      normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
        elif normalized_abs_box_in_values:
            traj_predictor_input = torch.cat((normalized_trajectory_values[:,:,0:4], 
                                              normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
            #abs_traj_pred_input = torch.cat((normalized_trajectory_values[:,:,4:8], 
            #                                 normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
            abs_traj_pred_input = normalized_trajectory_values # None
            abs_classifier_input = normalized_trajectory_values[:,:,4:8]
            normalized_trajectory_values = torch.cat((normalized_trajectory_values[:,:,0:4], 
                                                      normalized_trajectory_values[:,:,-1].unsqueeze(2)), 2)
        else:
            traj_predictor_input = normalized_trajectory_values
        
        # Predict trajectory between TTE=0 and TTE=60
        predicted_trajectory = self.traj_TF(
             normalized_trajectory_values=traj_predictor_input
        ).logits

        if predict_after_tte:
            after_tte_predicted_trajectory = self.after_tte_traj_TF(
                normalized_trajectory_values=trajectory_values
            ).logits
            predicted_trajectory = torch.cat((predicted_trajectory, 
                                              after_tte_predicted_trajectory[:, 60:, :]), 1)
        if predict_abs_traj:
            abs_predicted_trajectory = self.traj_abs_TF(
                normalized_trajectory_values=normalized_trajectory_values
            ).logits
            #predicted_trajectory = torch.cat((predicted_trajectory[:,:,0:4],
            #                                  abs_predicted_trajectory,
            #                                  predicted_trajectory[:,:,-1].unsqueeze(2)), 2)

        # Crossing prediction ====================================================

        # Add element to indicate predicted vs past trajectory

        extra_dim_vals = torch.zeros((predicted_trajectory.shape[0],          # add 0
                            predicted_trajectory.shape[1], 1)).to(self._device)
        predicted_trajectory = torch.cat((predicted_trajectory, extra_dim_vals), 2)
        extra_dim_vals = torch.ones((normalized_trajectory_values.shape[0],   # add 1
                            normalized_trajectory_values.shape[1], 1)).to(self._device)
        trajectory_values = torch.cat((normalized_trajectory_values, extra_dim_vals), 2)

        """
        extra_dim_vals = torch.zeros((predicted_trajectory.shape[0],          # add 01
                            predicted_trajectory.shape[1], 0)).to(self._device)
        extra_dim_vals[:,:,1] = torch.ones((extra_dim_vals.shape[0],
                            extra_dim_vals.shape[1]))
        predicted_trajectory = torch.cat((predicted_trajectory, extra_dim_vals), 2)

        extra_dim_vals = torch.zeros((normalized_trajectory_values.shape[0],  # add 00
                            normalized_trajectory_values.shape[1], 2)).to(self._device)
        trajectory_values = torch.cat((normalized_trajectory_values, extra_dim_vals), 2)
        """

        """
        if predict_abs_traj:
            extra_dim_vals = torch.zeros((abs_predicted_trajectory.shape[0],  # add 10
                                abs_predicted_trajectory.shape[1], 2)).to(self._device)
            extra_dim_vals[:,:,0] = torch.ones((extra_dim_vals.shape[0],
                            extra_dim_vals.shape[1]))
            abs_predicted_trajectory = torch.cat((abs_predicted_trajectory, extra_dim_vals), 2)

            predicted_trajectory[:,-1,:] = abs_predicted_trajectory[:,-1,:] #.squeeze(1)
            predicted_trajectory[:,29,:] = abs_predicted_trajectory[:,29,:]
            
            #extra_dim_vals = torch.ones((abs_traj_pred_input.shape[0], 
            #                    abs_traj_pred_input.shape[1], 1)).to(self._device)
            #abs_trajectory_values = torch.cat((abs_traj_pred_input, extra_dim_vals), 2)
        """


        # Only keep values found in prediction horizon (last 1 second)
        # predicted_trajectory = predicted_trajectory[:,30:,:]
        predicted_trajectory = torch.cat([trajectory_values,
                                          predicted_trajectory], dim=1)
                                          # abs_predicted_trajectory], dim=1)
        #if predict_abs_traj:
        #    abs_predicted_trajectory = torch.cat([abs_trajectory_values, 
        #                                          abs_predicted_trajectory], dim=1)

        if use_cross_attention:
            # Compute inter-modal cross-attention
            box_values = nn.ReLU()(self.box_emb_fc(predicted_trajectory[:,:,0:4]))
            speed_values = nn.ReLU()(self.speed_emb_fc(predicted_trajectory[:,:,5].unsqueeze(2)))

            cross_attn_ctx_1, attn_1 = self.cross_attention_1(box_values, speed_values)
            cross_attn_ctx_2, attn_2 = self.cross_attention_2(speed_values, box_values)
            
            predicted_trajectory = torch.cat([
                predicted_trajectory, cross_attn_ctx_1, cross_attn_ctx_2], dim=2)


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
                 # dataset_statistics=None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self.num_labels = config_for_huggingface.num_labels

        self.transformer = TrajectoryTransformerModel(config_for_huggingface, config_for_timeseries_lib,
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
                            return_dict)

