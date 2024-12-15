from typing import Optional, Tuple, Union

import torch
from torch import nn
from torchsummary import summary
from transformers import TrainingArguments, Trainer, TimesformerConfig
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model, denormalize_trajectory_data
from hugging_face.timeseries_utils import HuggingFaceTimeSeriesClassificationModel, TimeSeriesLibraryConfig, TorchTimeseriesDataset
from hugging_face.utilities import compute_loss, get_device
from hugging_face.utils.create_optimizer import get_optimizer
from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
#from libs.time_series_library.models_tsl.TimesNet import Model as VanillaTransformerTSLModel
#from libs.time_series_library.models_tsl.Nonstationary_Transformer import Model as VanillaTransformerTSLModel
#from libs.time_series_library.models_tsl.FEDformer import Model as VanillaTransformerTSLModel
#from libs.time_series_library.models_tsl.Informer import Model as VanillaTransformerTSLModel
#from libs.time_series_library.models_tsl.Autoformer import Model as VanillaTransformerTSLModel


PRED_LEN = 60

def get_config_for_timeseries_lib(encoder_input_size, seq_len,
                                  hyperparams, pred_len=None):

    if hyperparams:
        hyperparams = hyperparams["VanillaTransformerForForecast"]
    
    # time series lib properties
    time_series_dict = {
        "task_name": "short_term_forecast",
        "pred_len": pred_len if pred_len else PRED_LEN,
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": encoder_input_size, # encoder input size - default value,
        "d_model": 128, # dimension of model - default value 
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
        "dec_in": encoder_input_size, # decoder input size
        "d_layers": hyperparams.get("d_layers", 2), # num of decoder layers
        # ---------------------------------------------------------------------------------
        "label_len": 15, # section shared by encoder and decoder
        "num_kernels": 6, # Timesnet - for Inception
        "top_k": 5, # Timesnet - for TimesBlock
        "moving_avg": 3, # 3, # FEDformer - window size of moving average, default=25
        "c_out": encoder_input_size,
        "distil": True, # Informer - whether to use distilling in encoder, using this argument means not using distilling
        "p_hidden_dims": [128, 128], # Nonstationary transformer - hidden layer dimensions of projector (List)
        "p_hidden_layers": 2, # Nonstationary transformer - number of hidden layers in projector
        # "num_kernels": 3, # override - Pyraformer
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    return config_for_timeseries_lib


class SmallTrajectoryTransformer(HuggingFaceTimeSeriesClassificationModel):

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
              *args, **kwargs):
        print("Starting model loading for model Trajectory Transformer ===========================")

        # Get parameters used by time series library model
        data_element = data_train['data'][0][0][0][0] # revisit
        encoder_input_size = data_element.shape[-1]
        seq_len = data_element.shape[-2]
        
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        hyperparam_vals = hyperparams["VanillaTransformerForForecast"] if hyperparams else {}
        lr = hyperparam_vals.get("lr", train_opts["lr"])
        batch_size = hyperparam_vals.get("batch_size", batch_size)
        
        config_for_timeseries_lib = get_config_for_timeseries_lib(encoder_input_size, seq_len, hyperparams)
        config_for_huggingface = TimeSeriesTransformerConfig()
        # config_for_huggingface.problem_type = "trajectory"
        config_for_huggingface.num_labels = 4

        model = VanillaTransformerForForecast(config_for_huggingface, config_for_timeseries_lib)
                                              # dataset_statistics)
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
            warmup_ratio=warmup_ratio,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="mse",
            greater_is_better=False,
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
            data_collator=self.collate_fn,
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
                 # dataset_statistics=None
        ):
        super().__init__(config_for_huggingface)
        self.tsl_transformer = VanillaTransformerTSLModel(config_for_timeseries_lib)    

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values,
        x_dec,
        *args,
        **kwargs
    ):
        # Get vanilla transformer model output
        outputs = self.tsl_transformer(
            x_enc=trajectory_values,
            x_mark_enc=None,
            x_dec=x_dec,
            x_mark_dec=None
        )
        return outputs
        

class VanillaTransformerForForecast(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib=None, 
                 # dataset_statistics=None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self.num_labels = config_for_huggingface.num_labels

        self.transformer = TrajectoryTransformerModel(config_for_huggingface, config_for_timeseries_lib)

        classifier_hidden_size = config_for_timeseries_lib.num_class # number of neurons in last Linear layer at the end of model
        self.classifier = nn.Linear(
            classifier_hidden_size, config_for_huggingface.num_labels) \
            if config_for_huggingface.num_labels > 0 else nn.Identity()

        self.timeseries_config = config_for_timeseries_lib
        # self.dataset_statistics = dataset_statistics

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trajectory_values: Optional[torch.Tensor] = None,
        normalized_trajectory_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_logits: Optional[bool] = None
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:

        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        zero_tensor = labels if labels is not None else normalized_trajectory_values
        dec_inp = torch.zeros_like(
            torch.empty(zero_tensor.shape[0], self.timeseries_config.pred_len, zero_tensor.shape[-1])).float().to(self.device)
            # zero_tensor[:, -self.timeseries_config.pred_len:, :]).float()
        dec_inp = torch.cat([
            normalized_trajectory_values[:, -self.timeseries_config.label_len:, :], dec_inp], dim=1).float().to(self.device)

        # Get trajectory transformer model output
        outputs = self.transformer(
            normalized_trajectory_values,
            dec_inp
        )
       
        # outputs = outputs[:, -self.args.pred_len:, :]
        # batch_y = labels[:, -self.timeseries_config.pred_len:, :].to(self.device)
        # logits = self.classifier(outputs)

        # Denormalize data
        """
        outputs_denorm = denormalize_trajectory_data(outputs, 
                                              normalization_type="0_1_scaling",
                                              dataset_statistics=self.dataset_statistics, 
                                              device=self.device)
        labels_denorm = denormalize_trajectory_data(labels, 
                                              normalization_type="0_1_scaling",
                                              dataset_statistics=self.dataset_statistics,
                                              device=self.device)
        """

        if return_logits:
            return outputs
        return compute_loss(None,
                            outputs,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict,
                            problem_type="trajectory")

