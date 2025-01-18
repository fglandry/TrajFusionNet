import os
import pickle
from typing import Any

from PIL import Image
import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import Dataset

from transformers.trainer_utils import EvalPrediction
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from models.hugging_face.utilities import compute_huggingface_metrics


class HuggingFaceImageClassificationModel():
    
    def compute_metrics(self, eval_pred: EvalPrediction):
        return compute_huggingface_metrics(eval_pred)

    def collate_fn(self, examples: list):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["label"] for example in examples])
        if "pixel_values_2" in examples[0]:
            pixel_values_2 = torch.stack([example["pixel_values_2"] for example in examples])
            return {"pixel_values": pixel_values, "pixel_values_2": pixel_values_2, "labels": labels}
        else:   
            return {"pixel_values": pixel_values, "labels": labels}
        

class TorchImageDataset(Dataset):
    def __init__(self, data: Any, targets: Any, data_type: str, 
                 generator: bool = False, transform: Any = None, 
                 dual_transform: Any = None, image_processor: Any = None, 
                 num_channels: int = None, debug: bool = False):
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

    def _get_image_item(self, index: int, dual_img: bool = False):
        return get_image_item(index, self.data, self.data_type, self.generator,
                              dual_transform=self.dual_transform, 
                              dual_img=dual_img)
    
    def __getitem__(self, index: int):
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


def get_image_item(index: int, data: Any, data_type: str, 
                   generator: bool, 
                   dual_transform: Any = None, 
                   dual_img: bool = False, 
                   debug: bool = False):
    if not generator:
        item = data[index].permute(3, 0, 1, 2)
    else:
        # data is of type DataGenerator
        if data_type=='test' and not dual_transform:
            item = data[index]
        else:
            item = data[index][0]
        if dual_transform:
            item = item[0] if not dual_img else item[1]

        item = np.asarray(item)
        item = np.squeeze(item)
    
    return convert_img_to_format_used_by_transform(item, debug)

def convert_img_to_format_used_by_transform(item: np.ndarray, debug: bool):
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
        x.show()
    return x

def get_image_transforms(image_processor: Any, 
                         num_channels: int = 3, 
                         dataset_statistics: dict = None, 
                         modality=None):

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
        preds = np.expand_dims(probs, axis=1)
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
