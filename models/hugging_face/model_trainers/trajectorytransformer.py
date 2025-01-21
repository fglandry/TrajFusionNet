from typing import Optional

import torch
from torchsummary import summary
from transformers import Trainer, TrainingArguments
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel

from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig
from models.hugging_face.utilities import compute_loss, get_device
from utils.data_load import DataGenerator

PRED_LEN = 60


class TrajectoryTransformer(HuggingFaceTimeSeriesModel):

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
              test_only: bool = False,
              *args, **kwargs):
        """ Train model
        Args:
            data_train [dict]: training data (data_train['data'][0] contains the generator)
            data_val [DataGenerator]: validation data
            model_path [str]: path where the model will be saved
            train_opts [str]: training options (includes learning rate)
            dataset_statistics [dict]: contains dataset statistics such as avg / std dev per feature
            hyperparams [dict]: hyperparameters to change during training
            test_only [bool]: is set to True, model will not be trained, only tested
        """
        
        print("Starting model loading for model Trajectory Transformer ===========================")

        # Get parameters to be used by TSLib library
        data_element = data_train['data'][0][0][0][0]
        encoder_input_size = data_element.shape[-1]
        seq_len = data_element.shape[-2]
        
        # Get hyperparameters if specified for training run
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        hyperparam_vals = hyperparams["VanillaTransformerForForecast"] if hyperparams else {}
        lr = hyperparam_vals.get("lr", train_opts["lr"])
        batch_size = hyperparam_vals.get("batch_size", batch_size)
        
        config_for_timeseries_lib = get_config_for_timeseries_lib(encoder_input_size, seq_len, hyperparams)
        config_for_huggingface = TimeSeriesTransformerConfig()
        config_for_huggingface.num_labels = encoder_input_size

        model = VanillaTransformerForForecast(config_for_huggingface, config_for_timeseries_lib)
        summary(model)

        # Get datasets
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, None,
            get_image_transform=False, img_model_config=None,
            dataset_statistics=dataset_statistics)

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
            max_steps=-1,
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
        if not test_only:
            print("Starting training of model Trajectory Transformer ===========================")
            trainer.train()

        return {
            "trainer": trainer,
            "val_transform": val_transforms_dicts
        }

    def test(self,
             test_data: tuple,
             training_result: dict,
             model_info: dict,
             *args,
             dataset_name: str = "",
             generator: bool = False,
             test_only: bool = False,
             **kwargs):
        """ Test model
        Args:
            test_data [tuple]: tuple containing data (index 0) and targets (index 1)
            training_result [dict]: dictionary containing training results
            model_info [dict]: dict containing model info such as saved path and transforms
            dataset_name [str]: name of dataset
            generator [bool]: if set to true, input data is provided in a generator
        """
        
        print("Starting inference using trained model Trajectory Transformer ===========================")

        if test_only:
            pretrained_model = load_pretrained_trajectory_transformer(dataset_name)
            training_result["trainer"].model = pretrained_model

        return test_time_series_based_model(
            test_data,
            training_result,
            model_info,
            generator
        )


class VanillaTransformerForForecast(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 config_for_timeseries_lib: dict = None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self.num_labels = config_for_huggingface.num_labels

        self.transformer = TrajectoryTransformerModel(config_for_huggingface, config_for_timeseries_lib)

        self.timeseries_config = config_for_timeseries_lib

        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        trajectory_values: Optional[torch.Tensor] = None,
        normalized_trajectory_values: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_logits: Optional[bool] = None
    ):
        """ Args:
        trajectory_values [Optional[torch.Tensor]]: non-normalized observed trajectory values
            of shape [batch, seq_len, enc]
        normalized_trajectory_values [torch.Tensor]: normalized observed trajectory values
            of shape [batch, seq_len, enc]
        labels [torch.Tensor]: future target trajectory values of shape [batch, pred_len, enc]
            (between time t=0 and time t=60)
        """

        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Create tensor of size [batch, seq_len+pred_len, enc] with trajectory values
        # between time t = -seq_len and t = 0, and zeros between time t=0 and time t=pred_len
        zero_tensor = labels if labels is not None else normalized_trajectory_values
        dec_inp = torch.zeros_like(
            torch.empty(zero_tensor.shape[0], 
                        self.timeseries_config.pred_len, 
                        zero_tensor.shape[-1])).float().to(self.device)
        dec_inp = torch.cat([
            normalized_trajectory_values[:, -self.timeseries_config.label_len:, :], 
            dec_inp], dim=1).float().to(self.device) # shape: [batch, 75, 5]

        # Get trajectory transformer model output
        outputs = self.transformer(
            normalized_trajectory_values,
            dec_inp
        )

        if return_logits:
            return outputs
        return compute_loss(None,
                            outputs,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict,
                            problem_type="trajectory")


class TrajectoryTransformerModel(TimeSeriesTransformerPreTrainedModel):
    
    base_model_prefix = "transformer" # needs to be a class property
    
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 config_for_timeseries_lib: dict
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
        # Get encoder-decoder transformer model output
        outputs = self.tsl_transformer(
            x_enc=trajectory_values,
            x_mark_enc=None,
            x_dec=x_dec,
            x_mark_dec=None
        )
        return outputs


def load_pretrained_trajectory_transformer(dataset_name: str,
                                           submodels_paths=None,
                                           traj_model_path_override: str = None):
    config_for_trajectory_predictor = get_config_for_timeseries_lib(
        encoder_input_size=5, seq_len=15, hyperparams={}, pred_len=60)
    if traj_model_path_override:
        checkpoint = traj_model_path_override
    elif submodels_paths:
        checkpoint = submodels_paths["traj_tf_path"]
    else:
        raise Exception()
        if dataset_name in ["pie", "combined"]:
            #checkpoint = "data/models/pie/TrajectoryTransformer/09Jan2025-08h21m06s"
            #checkpoint = "data/models/pie/TrajectoryTransformer/08Jan2025-12h01m11s"
            checkpoint = "data/models/pie/TrajectoryTransformer/13Aug2024-11h16m29s_TE22"
        elif dataset_name == "jaad_all":
            checkpoint = "data/models/jaad/TrajectoryTransformer/05Oct2024-11h30m11s_TE24"
        elif dataset_name == "jaad_beh":
            #checkpoint = "data/models/jaad/TrajectoryTransformer/20Dec2024-14h24m55s_BE26"
            checkpoint = "data/models/jaad/TrajectoryTransformer/20Nov2024-10h50m14s_TE25"
            # checkpoint = "data/models/jaad/TrajectoryTransformer/10Aug2024-11h55m10s_TE23"
            # checkpoint = "data/models/pie/TrajectoryTransformer/05Jul2024-16h28m50s_TE3"

    pretrained_model = VanillaTransformerForForecast.from_pretrained(
        checkpoint,
        config_for_timeseries_lib=config_for_trajectory_predictor,
        ignore_mismatched_sizes=True)
    
    # Make all layers untrainable
    for child in pretrained_model.children():
        for param in child.parameters():
            param.requires_grad = False
    return pretrained_model


def get_config_for_timeseries_lib(
        encoder_input_size: int, 
        seq_len: int,
        hyperparams: dict, 
        pred_len: int = None):

    if hyperparams:
        hyperparams = hyperparams["VanillaTransformerForForecast"]
    
    # time series lib (TSLib) properties
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
        "n_heads": hyperparams.get("n_heads", 4), # num of heads
        "d_ff": hyperparams.get("d_ff", 512), # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": hyperparams.get("e_layers", 8), # num of encoder layers (or 3)
        "seq_len": seq_len, # input sequence length
        "num_class": 40, # number of neurons in last Linear layer at the end of model
        "dec_in": encoder_input_size, # decoder input size
        "d_layers": hyperparams.get("d_layers", 8), # num of decoder layers
        "label_len": 15,
        "c_out": encoder_input_size,
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    return config_for_timeseries_lib
