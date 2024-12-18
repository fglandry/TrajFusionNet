import copy
import cv2
import numpy as np
import os
import pickle
from PIL import Image
from scipy.special import softmax
import time
import torch
from torch.utils.data import Dataset
from datasets import load_metric

from transformers import AutoImageProcessor
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from models.hugging_face.utilities import compute_huggingface_metrics


class HuggingFaceImageClassificationModel():
    
    def compute_metrics(self, eval_pred):
        return compute_huggingface_metrics(eval_pred)

    def collate_fn(self, examples):
        """
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = [example["label"] for example in examples]
        labels = torch.stack(labels) # is it needed? added
        labels = torch.squeeze(labels) # added
        return {"pixel_values": pixel_values, "labels": labels}
        """
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        if "pixel_values_2" in examples[0]:
            pixel_values_2 = torch.stack([example["pixel_values_2"] for example in examples])
            return {"pixel_values": pixel_values, "pixel_values_2": pixel_values_2, "labels": labels}
        else:   
            return {"pixel_values": pixel_values, "labels": labels}


def get_image_datasets(data_train, data_val, generator, 
                       img_model_config,
                       dataset_statistics, 
                       image_processor=None,
                       get_dual_dataset=False):

    assert generator

    # Get image processor if not provided
    if not image_processor:
        img_proc_ckpt = "Visual-Attention-Network/van-base"
        image_processor = AutoImageProcessor.from_pretrained(img_proc_ckpt)

    train_img_transform, val_img_transform = \
        get_image_transforms(image_processor,
                             num_channels=img_model_config.num_channels,
                             config=img_model_config)
    if get_dual_dataset:
        image_processor_2 = copy.deepcopy(image_processor)
        obs_input_type = data_train["data_params"]["data_types"][1]
        image_processor_2.image_mean = dataset_statistics["dataset_means"][obs_input_type]
        image_processor_2.image_std = dataset_statistics["dataset_std_devs"][obs_input_type]
        train_img_transform_2, val_img_transform_2 = \
            get_image_transforms(image_processor_2,
                                num_channels=4,
                                config=img_model_config)

    train_dataset = TorchImageDataset(
        data_train['data'][0], None, 'train', 
        generator=generator,
        transform=train_img_transform,
        dual_transform=train_img_transform_2)
    val_dataset = TorchImageDataset(
        data_val, None, 'val', 
        generator=generator, 
        transform=val_img_transform,
        dual_transform=val_img_transform_2)
    
    val_transforms_dicts = {
        "val_img_transform": val_img_transform,
        "val_img_transform_2": val_img_transform_2
    }

    return train_dataset, val_dataset, val_transforms_dicts


class TorchImageDataset(Dataset):
    def __init__(self, data, targets, data_type, generator=False, 
                 transform=None, dual_transform=None, image_processor=None, 
                 num_channels=None, debug=False):
        if not generator:
            self.data = torch.from_numpy(data)
            self.targets = torch.LongTensor(targets)
        else:
            self.data = data # data is a Generator object, casting is done in __getitem__
            if data_type != 'test':
                self.targets = None
            elif data_type == 'test':
                self.targets = torch.LongTensor(targets)
            
        # Get transforms
        if dual_transform:
            self.transform = transform
            self.dual_transform = dual_transform
        else:
            if not transform:
                train_transform, val_transform = \
                    get_image_transforms(image_processor, num_channels)
                self.transform = train_transform if data_type=="train" else val_transform
            else:
                self.transform = transform
            self.dual_transform = None

        self.data_type = data_type
        self.generator = generator
        self.debug = debug

    def _get_image_item(self, index, dual_img=False):
        return get_image_item(index, self.data, self.data_type, self.generator,
                              dual_transform=self.dual_transform, 
                              dual_img=dual_img)
    
    def __getitem__(self, index):
        item = self._get_image_item(index)
        if self.transform:
            item = self.transform(item)
            x = {"pixel_values": item}

        if self.dual_transform:
            item2 = self._get_image_item(index, dual_img=True)
            item2 = self.dual_transform(item2)
            x.update({"pixel_values_2": item2})
        
        if not self.generator or (self.generator and self.data_type=='test'):
            label = self.targets[index]
        else:
            label = torch.LongTensor(self.data[index][1])
            label = label.reshape((1))
        x["label"] = label

        return x
    
    def __len__(self):
        return len(self.data)

def get_image_item(index, data, data_type, generator, 
                   dual_transform=None, dual_img=False, debug=False):
    if not generator:
        item = data[index].permute(3, 0, 1, 2)
    else:
        # data is of type DataGenerator
        if data_type=='test' and not dual_transform: # todo: verify
            item = data[index]
        else:
            item = data[index][0]
        if dual_transform:
            item = item[0] if not dual_img else item[1]

        item = np.asarray(item)
        #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/sem_output_{str(time.time()).replace('.', '_')}.png", np.squeeze(item))
        item = np.squeeze(item)
    
    return convert_img_to_format_used_by_transform(item, debug)

def convert_img_to_format_used_by_transform(item, debug):
    if item.shape[-1] == 4:
        x = Image.fromarray(item.astype('uint8'), 'RGBA') # return PIL image
    elif item.shape[-1] == 5:
        x = item.astype('uint8') # return image as numpy array
    elif item.shape[-1] == 224:
        x = np.repeat(item[:,:,np.newaxis], 3, axis=2)
        x = Image.fromarray(x.astype('uint8'), 'RGB') # return PIL image
    else:
        x = Image.fromarray(item.astype('uint8'), 'RGB') # return PIL image
    if debug:
        #x.show()
        x.save("test.png")
    return x

def get_image_transforms(image_processor, num_channels=3, 
                         dataset_statistics=None, modality=None):

    if dataset_statistics:
        normalize = Normalize(
            mean=dataset_statistics["dataset_means"][modality], 
            std=dataset_statistics["dataset_std_devs"][modality])
    else:
        normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    if "height" in image_processor.size:
        size = (image_processor.size["height"], image_processor.size["width"])
        crop_size = size
        max_size = None
    elif "shortest_edge" in image_processor.size:
        size = image_processor.size["shortest_edge"]
        crop_size = (size, size)
        max_size = image_processor.size.get("longest_edge")

    if num_channels == 5: # image can't be converted to PIL image
        train_transforms = Compose(
            [
                ToTensor(),
                #RandomResizedCrop(crop_size),
                #RandomHorizontalFlip(),
                normalize,
            ]
        )
        val_transforms = Compose(
            [
                ToTensor(),
                #Resize(size),
                #CenterCrop(crop_size),
                normalize,
            ]
        )
    else:
        train_transforms = Compose(
            [
                #RandomResizedCrop(crop_size),
                #RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )
        val_transforms = Compose(
            [
                #Resize(size),
                #CenterCrop(crop_size),
                ToTensor(),
                normalize,
            ]
        )

    return train_transforms, val_transforms


def test_image_based_model(
        test_data,
        training_result,
        model_path, 
        generator,
        dual_transform=None,
        save_results=True
    ):
    
    if isinstance(training_result["val_transform"], dict) and \
        "val_img_transform_2" in training_result["val_transform"]: # a dual_transform is found
        val_transform = training_result["val_transform"]["val_img_transform"]
        dual_transform = training_result["val_transform"]["val_img_transform_2"]
    else:
        val_transform = training_result["val_transform"]
        dual_transform = None

    trainer = training_result["trainer"]
    
    if not generator:
        test_dataset = TorchImageDataset(test_data[0][0], test_data[1], 'test', generator=generator, 
                                         transform=val_transform,
                                         dual_transform=dual_transform)
    else:
        #test_data[0] contains the DataGenerator object
        test_dataset = TorchImageDataset(test_data[0], test_data[1], 'test', generator=generator, 
                                         transform=val_transform,
                                         dual_transform=dual_transform)

    trainer_predictions = trainer.predict(test_dataset)
    if save_results:
        logits = trainer_predictions[0]
        probs = softmax(logits, axis=1)
        probs = probs[:,1]
        #preds = np.where(probs > 0.5, 1, 0)
        preds = np.expand_dims(probs, axis=1)
        #targets = np.squeeze(test_data[1])
        targets = test_data[1]

        with open(os.path.join(model_path["saved_files_path"], 'predictions.pkl'), 'wb') as picklefile:
            pickle.dump({'predictions': preds,
                        'test_data': targets}, picklefile)

        trainer.save_model()
    
    test_results = trainer_predictions[2]
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_state()

    acc = test_results["test_accuracy"]
    auc = test_results["test_auc"]
    f1 = test_results["test_f1"]
    precision = test_results["test_precision"]
    recall = test_results["test_recall"]

    return acc, auc, f1, precision, recall


def get_vit_image_transforms(processor):

    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]

    normalize = Normalize(mean=image_mean, std=image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(size),
                CenterCrop(size),
                ToTensor(),
                normalize,
            ]
        )

    """
    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['img']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['img']]
        return examples
    """

    return _train_transforms, _val_transforms
