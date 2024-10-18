import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer

from hugging_face.image_utils import get_vit_image_transforms, test_image_based_model, HuggingFaceImageClassificationModel, TorchImageDataset
from hugging_face.utils.create_optimizer import get_optimizer
from hugging_face.utils.custom_trainer import CustomTrainer
from hugging_face.utilities import compute_loss as _compute_loss, get_device

label2id = {}

class VisionTransformer(HuggingFaceImageClassificationModel):

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
              **kwargs): 
        
        
        print("Starting model loading for model Vision Transformer ===========================")

        self._device = get_device()
        self.class_w = torch.tensor(class_w).to(self._device)
        image_processor, config = get_vit_image_processor_and_config(
            data_train, dataset_statistics
        )
        
        model_ckpt = "google/vit-base-patch16-224-in21k"
        model = ViTForImageClassification.from_pretrained(
            model_ckpt,
            config=config,
            ignore_mismatched_sizes=True)
        """
        model = ViTForImageClassification(
            config=config
        )
        """
        
        train_transform, val_transform = \
            get_vit_image_transforms(image_processor)

        if not generator:
            train_dataset = TorchImageDataset(data_train['data'][0][0], data_train['data'][1], 'train', transform=train_transform)
            val_dataset = TorchImageDataset(data_val[0][0], data_val[1], 'val', transform=val_transform)
        else:
            train_dataset = TorchImageDataset(data_train['data'][0], None, 'train', generator=generator, transform=train_transform)
            val_dataset = TorchImageDataset(data_val, None, 'val', generator=generator, transform=val_transform)

        warmup_ratio = 0.1
        args = TrainingArguments(
            output_dir=model_path,
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size, # 10 in example
            per_device_eval_batch_size=batch_size,  # 4 in example
            num_train_epochs=epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="auc",
            #logging_dir='logs', # ?
            #warmup_ratio=0.1,
            logging_steps=10,
            push_to_hub=False,
            max_steps=-1,
            # device_map='auto' // not supported
        )

        optimizer, lr_scheduler = get_optimizer(self, model, args, 
            train_dataset, val_dataset, data_train, train_opts, warmup_ratio)

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

        # Train model
        print("Starting training of model Video Transformer ===========================")
        train_results = trainer.train()

        """
        if free_memory:
            free_train_and_val_memory(data_train, data_val)
            free_train_and_val_memory(train_dataset, val_dataset)
        """
            
        return {
            "trainer": trainer,
            "val_transform": val_transform
        }
        # return trainer

    def test(self,
             test_data,
             training_result,
             model_path,
             *args,
             generator=False,
             **kwargs):
        
        print("Starting inference using trained model (Vision Transformer) ===========================")

        return test_image_based_model(
            test_data,
            training_result,
            model_path,
            generator
        )

def get_vit_image_processor_and_config(data_train, dataset_statistics,
                                       obs_input_type=None):
    class_labels = ["no_cross", "cross"]
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    model_ckpt = "google/vit-base-patch16-224-in21k"
    image_processor = ViTImageProcessor.from_pretrained(model_ckpt)
    
    if obs_input_type is None:
        obs_input_type = data_train["data_params"]["data_types"][0]
    if "flow_optical_v4" in obs_input_type:
        # Image data has 4 channels (3 RGB + 1 OF)
        image_processor.image_mean.append(image_processor.image_mean[-1])
        image_processor.image_std.append(image_processor.image_std[-1])
    elif ("flow_optical_v5" in obs_input_type):
        image_processor = ViTImageProcessor.from_pretrained(model_ckpt)
        # Image data has 5 channels (3 RGB + 2 OF)
        image_processor.image_mean.append(image_processor.image_mean[-1])
        image_processor.image_std.append(image_processor.image_std[-1])
    else:
        image_processor.image_mean = dataset_statistics["dataset_means"][obs_input_type]
        image_processor.image_std = dataset_statistics["dataset_std_devs"][obs_input_type]

    config = ViTForImageClassification.from_pretrained(
            model_ckpt,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True).config # find something better to get proper config
    
    config.num_channels = 3

    return image_processor, config