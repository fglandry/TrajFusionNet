from typing import Optional, Tuple, Union

import torch
from torch import nn
from torchsummary import summary
from transformers import TrainingArguments, Trainer
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention

from models.hugging_face.timeseries_utils import test_time_series_based_model, HuggingFaceTimeSeriesModel, TimeSeriesLibraryConfig, TorchTimeseriesDataset
from models.hugging_face.utilities import compute_loss
from libs.time_series_library.models_tsl.Transformer import Model as VanillaTransformerTSLModel
#from libs.time_series_library.models_tsl.TimesNet import Model as TimesnetModel
#from libs.time_series_library.models_tsl.FEDformer import Model as FEDformer
#from libs.time_series_library.models_tsl.Autoformer import Model as Autoformer
#from libs.time_series_library.models_tsl.Crossformer import Model as Crossformer
#from libs.time_series_library.models_tsl.DLinear import Model as DLinear
#from libs.time_series_library.models_tsl.ETSformer import Model as ETSformer
#from libs.time_series_library.models_tsl.FiLM import Model as FiLM
#from libs.time_series_library.models_tsl.Informer import Model as Informer
#from libs.time_series_library.models_tsl.LightTS import Model as LightTS
#from libs.time_series_library.models_tsl.MICN import Model as MICN
#from libs.time_series_library.models_tsl.Nonstationary_Transformer import Model as Nonstationary_Transformer
#from libs.time_series_library.models_tsl.PatchTST import Model as PatchTST
#from libs.time_series_library.models_tsl.Pyraformer import Model as Pyraformer
#from libs.time_series_library.models_tsl.Reformer import Model as Reformer


def get_config_for_timeseries_lib(encoder_input_size, seq_len):

    # time series lib properties
    time_series_dict = {
        "task_name": "classification",
        "pred_len": 0, # for Timesblock
        "output_attention": False, # whether to output attention in encoder; note: not used by vanilla transformer model
        "enc_in": encoder_input_size, # encoder input size - default value,
        "d_model": 128, # dimension of model - default value 
        "embed": "learned", # time features encoding; note: not used in classification task by vanilla transformer model
        "freq": "h", # freq for time features encoding; note: not used in classification task by vanilla transformer model
        "dropout": 0.1, # default,
        "factor": 1, # attn factor; note: not used by vanilla transformer model
        "n_heads": 12, # num of heads
        "d_ff": 256, # dimension of fcn (or 2048)
        "activation": "gelu",
        "e_layers": 2, # num of encoder layers (or 3)
        "seq_len": seq_len, # input sequence length
        "num_class": 40, # number of neurons in last Linear layer at the end of model
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
    return config_for_timeseries_lib


class VanillaTransformer(HuggingFaceTimeSeriesModel):

    def train(self,
              data_train, 
              data_val,
              batch_size, 
              epochs,
              model_path,  
              generator=False,
              *args, **kwargs):
        print("Starting model loading for model Vanilla Transformer ===========================")

        # Get parameters used by time series library model
        data_element = data_train['data'][0][0][0][0] # revisit
        encoder_input_size = data_element.shape[-1]
        seq_len = data_element.shape[-2]
        
        config_for_timeseries_lib = get_config_for_timeseries_lib(encoder_input_size, seq_len)
        config_for_huggingface = TimeSeriesTransformerConfig()

        model = VanillaTransformerForClassification(config_for_huggingface, config_for_timeseries_lib)
        summary(model)

        train_dataset = TorchTimeseriesDataset(data_train['data'][0], None, 'train', generator=generator)
        val_dataset = TorchTimeseriesDataset(data_val, None, 'val', generator=generator)

        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
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
            data_collator=self.collate_fn,
        )

        # Train model
        print("Starting training of model Vanilla Transformer ===========================")
        train_results = trainer.train()

        return {
            "trainer": trainer,
            "val_transform": None
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


class VanillaTransformerModel(TimeSeriesTransformerPreTrainedModel):
    
    base_model_prefix = "transformer" # needs to be a class property
    
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib
        ):
        super().__init__(config_for_huggingface)
        self.tsl_transformer = VanillaTransformerTSLModel(config_for_timeseries_lib)    

        # Initialize weights and apply final processing
        self.post_init()
    
    def forward(
        self,
        trajectory_values,
        *args,
        **kwargs
    ):
        # Get vanilla transformer model output
        outputs = self.tsl_transformer(
            x_enc=trajectory_values,
            x_mark_enc=None,
            x_dec=None,
            x_mark_dec=None
        )
        return outputs # Todo: return BaseModelOutput instead?
        

class VanillaTransformerForClassification(TimeSeriesTransformerPreTrainedModel):
    def __init__(self,
                 config_for_huggingface,
                 config_for_timeseries_lib=None, 
        ):
        super().__init__(config_for_huggingface, config_for_timeseries_lib)
        self.num_labels = config_for_huggingface.num_labels

        self.transformer = VanillaTransformerModel(config_for_huggingface, config_for_timeseries_lib)

        classifier_hidden_size = config_for_timeseries_lib.num_class # number of neurons in last Linear layer at the end of model
        self.classifier = nn.Linear(
            classifier_hidden_size, config_for_huggingface.num_labels) \
            if config_for_huggingface.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        trajectory_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:

        assert output_hidden_states is None
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get vanilla transformer model output
        outputs = self.transformer(
            trajectory_values
        )
       
        # Concatenate outputs
        #outputs = torch.cat((output1_tensor, output2_tensor), dim=1)

        logits = self.classifier(outputs)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self.config,
                            self.num_labels,
                            return_dict)
