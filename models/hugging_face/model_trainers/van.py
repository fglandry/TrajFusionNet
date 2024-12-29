from typing import Optional

import torch
from torch import nn
import torch.nn.functional
import torch.utils.checkpoint
from torchsummary import summary
from transformers import AutoImageProcessor
from transformers import TrainingArguments, Trainer
from transformers import VanConfig, VanModel, VanPreTrainedModel
from transformers.models.van.modeling_van import VanEncoder
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention

from models.hugging_face.image_utils import test_image_based_model, HuggingFaceImageClassificationModel, TorchImageDataset
from models.hugging_face.utilities import compute_loss, get_class_labels_info, get_device
from utils.data_load import DataGenerator


class VAN(HuggingFaceImageClassificationModel):

    def train(self,
              data_train: dict, 
              data_val: DataGenerator,
              batch_size: int,
              epochs: int,
              model_path: str, 
              *args,
              train_opts: dict = None,  
              generator: bool = False,
              dataset_statistics: dict = None,
              test_only: bool = False,
              **kwargs
        ):
        """ Train model
        Args:
            data_train [dict]: training data (data_train['data'][0] contains the generator)
            data_val [DataGenerator]: validation data
            model_path [str]: path where the model will be saved
            train_opts [str]: training options (includes learning rate)
            dataset_statistics [dict]: contains dataset statistics such as avg / std dev per feature
            test_only [bool]: is set to True, model will not be trained, only tested
        """
    
        print("Starting model loading for model VAN: Visual Attention Network ===========================")

        self._device = get_device()
        image_processor, config = get_van_image_processor_and_config(
            data_train, dataset_statistics
        )

        model_ckpt = "Visual-Attention-Network/van-base"
        model = VanEncodingsForImageClassification.from_pretrained(
            model_ckpt,
            config=config,
            ignore_mismatched_sizes=True)
        summary(model)

        if not generator:
            train_dataset = TorchImageDataset(
                data_train['data'][0][0], data_train['data'][1], 'train',
                image_processor=image_processor, num_channels=config.num_channels
            )
            val_dataset = TorchImageDataset(
                data_val[0][0], data_val[1], 'val',
                image_processor=image_processor, num_channels=config.num_channels
            )
        else:
            train_dataset = TorchImageDataset(
                data_train['data'][0], None, 'train', 
                generator=generator, image_processor=image_processor, num_channels=config.num_channels
            )
            val_dataset = TorchImageDataset(
                data_val, None, 'val', 
                generator=generator, image_processor=image_processor, num_channels=config.num_channels
            )

        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=train_opts["lr"],
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=image_processor,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn
        )

        # Train model
        if not test_only:
            print("Starting training of model VAN: Visual Attention Network ===========================")
            trainer.train()
        
        return {
            "trainer": trainer,
            "val_transform": val_dataset.transform
        }

    def test(self,
             test_data: tuple,
             training_result: dict,
             model_info: dict,
             *args,
             dataset_name: str = "",
             generator=False,
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
        
        print("Starting inference using trained model VAN: Visual Attention Network ===========================")

        if test_only:
            is_predicted_overlays = "_previous" not in kwargs["complete_data"]["data_params"]["data_types"][0]
            pretrained_model = load_pretrained_van(dataset_name,
                                                   is_predicted_overlays=is_predicted_overlays)
            training_result["trainer"].model = pretrained_model

        return test_image_based_model(
            test_data,
            training_result,
            model_info,
            generator
        )


class VanEncodingsForImageClassification(VanPreTrainedModel):
    """ Adapted from the transformers library """

    def __init__(self, 
                 config: VanConfig):
        super().__init__(config)
        self.van = VanEncodingsModel(config)
        self._config = config
        self.classifier = (
            nn.Linear(30, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )
        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.van(
            pixel_values, 
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        
        logits = self.classifier(pooled_output)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self._config,
                            self._config.num_labels,
                            return_dict,
                            problem_type=self._config.problem_type)


class VanEncodingsModel(VanPreTrainedModel):
    """ Adapted from the transformers library """

    def __init__(self, config: VanConfig):
        super().__init__(config)
        self.config = config
        self.encoder = VanEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps) # final layernorm layer
        self.dropout = nn.Dropout(p=0.2)
        self.fcN = nn.Linear(config.hidden_sizes[-1], 30) 
        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            pixel_values,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        # global average pooling, n c w h -> n c
        pooled_output = last_hidden_state.mean(dim=[-2, -1])
        pooled_output = self.dropout(nn.ReLU()(self.fcN(pooled_output)))

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndNoAttention(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
        )


def load_pretrained_van(dataset_name: str,
                        is_predicted_overlays: bool = True,
                        add_classification_head: bool = True,
                        submodels_paths: dict = None):
    if submodels_paths:
        checkpoint1 = submodels_paths["van_path"]
        checkpoint2 = submodels_paths["van_prev_path"]
    else:
        label2id, id2label = get_class_labels_info()
        if dataset_name in ["pie", "combined"]:
            checkpoint1 = "data/models/pie/VAN/14Oct2024-00h13m09s_VA10"
            checkpoint2 = "data/models/pie/VAN/14Oct2024-10h37m58s_VA11"
            #checkpoint1 = "data/models/jaad/VAN/12Oct2024-20h59m24s_VA6"
            #checkpoint2 = "data/models/jaad/VAN/12Oct2024-23h06m29s_VA7"
        elif dataset_name == "jaad_all":
            checkpoint1 = "data/models/jaad/VAN/25Dec2024-11h25m13s_VA6B"
            checkpoint2 = "data/models/jaad/VAN/25Dec2024-13h28m35s_VA7B"
            #checkpoint1 = "data/models/jaad/VAN/12Oct2024-20h59m24s_VA6"
            #checkpoint2 = "data/models/jaad/VAN/12Oct2024-23h06m29s_VA7"
        elif dataset_name == "jaad_beh":
            checkpoint1 = "data/models/jaad/VAN/25Dec2024-16h43m58s_VA8B"
            checkpoint2 = "data/models/jaad/VAN/25Dec2024-21h06m13s_VA9B"
            #checkpoint1 = "data/models/jaad/VAN/13Oct2024-20h16m00s_VA8"
            #checkpoint2 = "data/models/jaad/VAN/13Oct2024-20h56m50s_VA9"
    
    checkpoint = checkpoint1 if is_predicted_overlays else checkpoint2

    label2id, id2label = get_class_labels_info()

    if add_classification_head:
        pretrained_model = VanEncodingsForImageClassification.from_pretrained(
            checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)
    else:
        pretrained_model = VanModel.from_pretrained(
            checkpoint,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)

    # Make all layers untrainable
    for child in pretrained_model.children():
        for param in child.parameters():
            param.requires_grad = False
    return pretrained_model


def get_van_image_processor_and_config(
        data_train: dict, 
        dataset_statistics: dict = None
    ):
    class_labels = ["no_cross", "cross"]
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    model_ckpt = "Visual-Attention-Network/van-base"
    
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

    obs_input_type = data_train["data_params"]["data_types"][0]
    image_processor.image_mean = dataset_statistics["dataset_means"][obs_input_type]
    image_processor.image_std = dataset_statistics["dataset_std_devs"][obs_input_type]

    # Get VAN model config
    config = VanEncodingsForImageClassification.from_pretrained(
        model_ckpt,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True).config # TODO: there must be a better way to do this without loading the model
    config.num_channels = 3
    config.problem_type = "single_label_classification"

    return image_processor, config
