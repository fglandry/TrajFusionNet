import torch

from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import TrainingArguments, Trainer

from models.hugging_face.image_utils import get_image_transforms, test_image_based_model, HuggingFaceImageClassificationModel, TorchImageDataset
from models.hugging_face.utils.create_optimizer import get_optimizer
from models.hugging_face.utilities import get_device
from models.hugging_face.utils.custom_trainer import CustomTrainer


label2id = {}

class HF_Resnet50(HuggingFaceImageClassificationModel):

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
              **kwrags):
        
        
        print("Starting model loading for model Resnet50 ===========================")

        class_labels = ["no_cross", "cross"]
        label2id = {label: i for i, label in enumerate(class_labels)}
        id2label = {i: label for label, i in label2id.items()}
        self._device = get_device()
        self.class_w = torch.tensor(class_w).to(self._device)

        #model_ckpt = "microsoft/swin-tiny-patch4-window7-224"
        model_ckpt = "microsoft/resnet-50"
        image_processor = AutoImageProcessor.from_pretrained(model_ckpt)
        model = ResNetForImageClassification.from_pretrained(
            model_ckpt,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes = True)

        train_transform, val_transform = \
            get_image_transforms(image_processor, dataset_statistics=dataset_statistics,
                                 modality="scene_context_with_segmentation_v3")

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
        print("Starting training of model Resnet50 ===========================")
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
        
        print("Starting inference using trained model (VAN: Visual Attention Network) ===========================")

        return test_image_based_model(
            test_data,
            training_result,
            model_path,
            generator
        )
