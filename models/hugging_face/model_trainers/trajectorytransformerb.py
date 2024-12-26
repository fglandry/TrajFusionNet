from typing import Optional

import torch
from torch import nn
from torchsummary import summary
from transformers import TrainingArguments, Trainer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel

from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
from models.hugging_face.model_trainers.trajectorytransformer import load_pretrained_trajectory_transformer
from models.hugging_face.timeseries_utils import get_timeseries_datasets, test_time_series_based_model
from models.hugging_face.timeseries_utils import HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig
from models.hugging_face.utilities import compute_loss, get_device
from utils.data_load import DataGenerator

PRED_LEN = 60


class TrajectoryTransformerb(HuggingFaceTimeSeriesModel):

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
        
        print("Starting model loading for model Trajectory Transformer Classifier ======================")

        # Get parameters to be used by TSLib library
        data_element = data_train['data'][0][0][0][0]
        encoder_input_size = data_element.shape[-1]
        seq_len = data_element.shape[-2] + PRED_LEN # 75
        
        # Get hyperparameters if specified for training run
        hyperparams = hyperparams.get(self.__class__.__name__.lower(), {}) if hyperparams else {}
        hyperparam_vals = hyperparams["EncoderTransformerForClassification"] if hyperparams else {}
        lr = hyperparam_vals.get("lr", train_opts["lr"])
        batch_size = hyperparam_vals.get("batch_size", batch_size)
        epochs = hyperparam_vals.get("epochs", epochs)
        
        config_for_timeseries_lib = get_config_for_timeseries_lib(
            encoder_input_size, seq_len, hyperparams)
        config_for_huggingface = TimeSeriesTransformerConfig()
        self.num_labels = config_for_huggingface.num_labels

        model = EncoderTransformerForClassification(
            config_for_huggingface, config_for_timeseries_lib,
            dataset_name=kwargs["model_opts"]["dataset_full"]
        )
        summary(model)

        # Get datasets
        train_dataset, val_dataset, val_transforms_dicts = get_timeseries_datasets(
            data_train, data_val, model, generator, None,
            get_image_transform=False, img_model_config=None,
            dataset_statistics=dataset_statistics, ignore_sem_map=True)

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
            max_steps=-1,
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
        if not test_only:
            print("Starting training of model Trajectory Transformer Classifier ===========================")
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
        
        print("Starting inference using trained model Trajectory Transformer Classifier ===========================")

        if test_only:
            pretrained_model = load_pretrained_encoder_transformer(dataset_name)
            training_result["trainer"].model = pretrained_model

        return test_time_series_based_model(
            test_data,
            training_result,
            model_info,
            generator
        )


class EncoderTransformerForClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 config_for_timeseries_lib: dict = None,
                 dataset_name: str = None
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self._device = get_device()
        self.num_labels = config_for_huggingface.num_labels
        self.timeseries_config = config_for_timeseries_lib

        self.transformer = EncoderTransformer(config_for_huggingface, config_for_timeseries_lib,
                                              dataset_name=dataset_name)

        classifier_hidden_size = config_for_timeseries_lib.num_class # number of neurons in last linear layer at the end of model
        self.classifier = nn.Linear(
            classifier_hidden_size, config_for_huggingface.num_labels) \
            if config_for_huggingface.num_labels > 0 else nn.Identity()

        self.fc1 = nn.Linear(classifier_hidden_size, self.num_labels) # [40, 2]

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trajectory_values: torch.Tensor = None,
        normalized_trajectory_values: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        """ Args:
        trajectory_values [torch.Tensor]: non-normalized observed trajectory values
            of shape [batch, seq_len, enc]
        normalized_trajectory_values [torch.Tensor]: normalized observed trajectory values
            of shape [batch, seq_len, enc]
        labels [torch.Tensor]: future target trajectory values of shape [batch, pred_len, enc]
            (between time t=0 and time t=60)
        """

        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get encoder transformer model output
        outputs = self.transformer(
            trajectory_values,
            normalized_trajectory_values
        )

        logits = self.fc1(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict)


class EncoderTransformer(TimeSeriesTransformerPreTrainedModel):
    
    base_model_prefix = "transformer" # needs to be a class property
    
    def __init__(self,
                 config_for_huggingface: TimeSeriesTransformerConfig,
                 config_for_timeseries_lib: dict,
                 dataset_name: str = None
        ):
        super().__init__(config_for_huggingface)
        self._dataset = dataset_name
        self._device = get_device()

        self.tsl_transformer = VanillaTransformerTSLModel(config_for_timeseries_lib)

        self.traj_TF = load_pretrained_trajectory_transformer(dataset_name)

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values: torch.Tensor,
        normalized_trajectory_values: torch.Tensor,
        *args,
        **kwargs
    ):
        """ Args:
        trajectory_values [torch.Tensor]: non-normalized observed trajectory values
            of shape [batch, seq_len, enc]
        normalized_trajectory_values [torch.Tensor]: normalized observed trajectory values
            of shape [batch, seq_len, enc]
        """

        # Trajectory prediction ====================================================
        
        # Predict trajectory (pedestrian bounding boxes and vehicle speed) between 
        # t=0 and t=60
        predicted_trajectory = self.traj_TF(
             normalized_trajectory_values=normalized_trajectory_values
        ).logits # [b, 60, 5]

        # Crossing prediction ====================================================

        # Add type identifier to indicate past observed trajectory (add 1) vs 
        # predicted trajectory (add 0)
        extra_dim_vals = torch.zeros((predicted_trajectory.shape[0],          # add 0 token
                                      predicted_trajectory.shape[1], 1)).to(self._device)
        predicted_trajectory = torch.cat((predicted_trajectory, extra_dim_vals), 2)
        extra_dim_vals = torch.ones((normalized_trajectory_values.shape[0],   # add 1 token
                                     normalized_trajectory_values.shape[1], 1)).to(self._device)
        trajectory_values = torch.cat((normalized_trajectory_values, extra_dim_vals), 2)

        predicted_trajectory = torch.cat([trajectory_values,
                                          predicted_trajectory], dim=1) # [b, 75, 6]

        # Get vanilla transformer model output
        outputs = self.tsl_transformer(
            x_enc=predicted_trajectory,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )

        return outputs # [b, 40]


def load_pretrained_encoder_transformer(dataset_name: str,
                                        add_classification_head: bool = True):
    config_for_trajectory_predictor = get_config_for_timeseries_lib(
            encoder_input_size=5, seq_len=75, hyperparams={})
    if dataset_name in ["pie", "combined"]:
        #checkpoint = "data/models/pie/TrajectoryTransformerb/20Nov2024-17h42m36s_NW1"
        checkpoint = "data/models/pie/TrajectoryTransformerb/06Sep2024-09h18m20s_TJ5"
    elif dataset_name == "jaad_all":
        checkpoint = "data/models/jaad/TrajectoryTransformerb/20Nov2024-12h02m51s_TJ8"
        #checkpoint = "data/models/jaad/TrajectoryTransformerb/13Oct2024-15h45m56s_TJ7"
        #checkpoint = "data/models/pie/TrajectoryTransformerb/06Sep2024-09h18m20s_TJ5"
    elif dataset_name == "jaad_beh":
        #checkpoint = "data/models/jaad/TrajectoryTransformerb/20Dec2024-14h54m11s_BE3"
        checkpoint = "data/models/jaad/TrajectoryTransformerb/20Nov2024-10h36m17s_BE2"
        #checkpoint = "data/models/jaad/TrajectoryTransformerb/20Nov2024-10h01m55s_BE1"
        #checkpoint = "data/models/jaad/TrajectoryTransformerb/13Oct2024-15h45m56s_TJ7"

    if add_classification_head:
        pretrained_model = EncoderTransformerForClassification.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_trajectory_predictor,
            ignore_mismatched_sizes=True,
            dataset_name=dataset_name)
    else:
        pretrained_model = EncoderTransformer.from_pretrained(
            checkpoint,
            config_for_timeseries_lib=config_for_trajectory_predictor,
            ignore_mismatched_sizes=True,
            dataset_name=dataset_name)
    
    # Make all layers untrainable
    for child in pretrained_model.children():
        for param in child.parameters():
            param.requires_grad = False
    return pretrained_model


def get_config_for_timeseries_lib(encoder_input_size: int, 
                                  seq_len: int,
                                  hyperparams: dict, 
                                  pred_len: int = None):
    
    encoder_input_size = encoder_input_size + 1 # plus one to account for type identifiers
    if hyperparams:
        hyperparams = hyperparams["EncoderTransformerForClassification"]

    # time series lib (TSLib) properties
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
        "label_len": 15,
        "c_out": 5,
    }
    
    config_for_timeseries_lib = TimeSeriesLibraryConfig(time_series_dict)
    return config_for_timeseries_lib
