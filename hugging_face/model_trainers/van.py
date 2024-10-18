import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional
import torch.utils.checkpoint
from torchsummary import summary
from typing import Optional, Union, Tuple

from transformers import AutoImageProcessor
from transformers import TrainingArguments, Trainer
from transformers.models.van.modeling_van import VanEncoder
from transformers import VanForImageClassification, VanPreTrainedModel, VanConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndNoAttention, ImageClassifierOutputWithNoAttention

from hugging_face.image_utils import test_image_based_model, HuggingFaceImageClassificationModel, TorchImageDataset
from hugging_face.utils.create_optimizer import get_optimizer
from hugging_face.utilities import compute_loss, get_device
from hugging_face.utils.custom_trainer import CustomTrainer

class VAN(HuggingFaceImageClassificationModel):

    def train(self,
              data_train, 
              data_val,
              batch_size, 
              epochs,
              model_path,
              *args,  
              generator=False,
              train_opts=None,
              dataset_statistics=None,
              class_w=None,
              **kwrags
        ):
        
        
        print("Starting model loading for model VAN: Visual Attention Network ===========================")

        self._device = get_device()
        self.class_w = torch.tensor(class_w).to(self._device)
        image_processor, config = get_van_image_processor_and_config(
            data_train, dataset_statistics
        )

        model_ckpt = "Visual-Attention-Network/van-base"
        #model_ckpt = "Visual-Attention-Network/van-large"
        #model_ckpt = "Visual-Attention-Network/van-small"
        #model_ckpt = "Visual-Attention-Network/van-tiny"

        model = VanEncodingsForImageClassification.from_pretrained(
            model_ckpt,
            config=config,
            ignore_mismatched_sizes = True)
        summary(model)
        """
        model = VanForImageClassification.from_pretrained(
            "data/models/jaad/VAN/27Aug2023-12h11m53s_ENC2",
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True)
        # Make all layers untrainable
        for idx, child in enumerate(model.children()):
            if idx == 1:
                continue
            for param in child.parameters():
                param.requires_grad = False
        """

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

        warmup_ratio = 0.1
        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            # gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            push_to_hub=False,
            max_steps=-1, # added
            # device_map='auto' // not supported
        )

        #optimizer, lr_scheduler = get_optimizer(self, model, args, 
        #    train_dataset, val_dataset, data_train, train_opts, warmup_ratio)

        """
        trainer = CustomTrainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=image_processor,
            compute_metrics=self.compute_metrics,
            data_collator=self.collate_fn,
            optimizers=(optimizer, lr_scheduler),
            class_w=self.class_w,
        )
        """
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
        print("Starting training of model VAN: Visual Attention Network ===========================")
        train_results = trainer.train()

        """
        if free_memory:
            free_train_and_val_memory(data_train, data_val)
            free_train_and_val_memory(train_dataset, val_dataset)
        """
        
        return {
            "trainer": trainer,
            "val_transform": val_dataset.transform
        }
        # return trainer

    def test(self,
             test_data,
             training_result,
             model_path,
             *args,
             generator=False,
             **kwargs):
        
        print("Starting inference using trained model (VAN: Visual Attention Network) ===========================")

        return test_image_based_model(
            test_data,
            training_result,
            model_path,
            generator
        )

class VanEncodingsModel(VanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.encoder = VanEncoder(config)
        # final layernorm layer
        self.layernorm = nn.LayerNorm(config.hidden_sizes[-1], eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=0.2)
        self.fcN = nn.Linear(config.hidden_sizes[-1], 30) 
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor],
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPoolingAndNoAttention]:
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


class VanEncodingsForImageClassification(VanPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.van = VanEncodingsModel(config)
        self._config = config
        self.classifier = (
            nn.Linear(30, config.num_labels) if config.num_labels > 0 else nn.Identity()
        )
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ImageClassifierOutputWithNoAttention]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.van(pixel_values, output_hidden_states=output_hidden_states, return_dict=return_dict)

        pooled_output = outputs.pooler_output if return_dict else outputs[1]
        
        logits = self.classifier(pooled_output)

        return compute_loss(outputs,
                            logits,
                            labels,
                            self._config,
                            self._config.num_labels,
                            return_dict,
                            problem_type=self._config.problem_type)


def get_van_image_processor_and_config(data_train, dataset_statistics, 
                                       obs_input_type=None):
    class_labels = ["no_cross", "cross"]
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    #model_ckpt = "microsoft/swin-tiny-patch4-window7-224"
    model_ckpt = "Visual-Attention-Network/van-base"
    #model_ckpt = "Visual-Attention-Network/van-large"
    #model_ckpt = "Visual-Attention-Network/van-small"
    #model_ckpt = "Visual-Attention-Network/van-tiny"
    
    image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

    if not obs_input_type:
        obs_input_type = data_train["data_params"]["data_types"][0]
    if "flow_optical_v4" in obs_input_type:
        # Image data has 4 channels (3 RGB + 1 OF)
        image_processor.image_mean.append(image_processor.image_mean[-1])
        image_processor.image_std.append(image_processor.image_std[-1])
    if ("flow_optical_v5" in obs_input_type):
        # Image data has 5 channels (3 RGB + 2 OF)
        image_processor.image_mean.append(image_processor.image_mean[-1])
        image_processor.image_std.append(image_processor.image_std[-1])
    else:
        image_processor.image_mean = dataset_statistics["dataset_means"][obs_input_type]
        image_processor.image_std = dataset_statistics["dataset_std_devs"][obs_input_type]

    # Model without any pre-training
    """
    config = VanConfig()
    model = VanEncodingsForImageClassification(
        config=config
    )
        """

    # Model with pre-training
    
    config = VanEncodingsForImageClassification.from_pretrained(
        model_ckpt,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes = True).config # find something better to get proper config
    #config = VanConfig()
    config.num_channels = 3
    config.problem_type = "single_label_classification"

    return image_processor, config