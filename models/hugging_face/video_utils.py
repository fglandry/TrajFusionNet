import numpy as np
import os
import pickle

from datasets import load_metric
from scipy.special import softmax
import torch
from torch.utils.data import Dataset


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

from models.hugging_face.utilities import compute_huggingface_metrics


class HuggingFaceVideoClassificationModel():
    
    def compute_metrics(self, eval_pred):
        return compute_huggingface_metrics(eval_pred)

    def collate_fn(self, examples):
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        #labels = torch.tensor([example["label"] for example in examples])
        labels = [example["label"] for example in examples]
        labels = torch.stack(labels) # is it needed? added
        labels = torch.squeeze(labels) # added
        return {"pixel_values": pixel_values, "labels": labels}

"""
class TorchVideoDataset(Dataset):
    def __init__(self, data, targets, data_type, generator=False, transform=None):
        if not generator:
            self.data = torch.from_numpy(data)
            self.targets = torch.LongTensor(targets)
        else:
            self.data = data # data is a Generator object, casting is done in __getitem__
            if data_type != 'test':
                self.targets = None
            elif data_type == 'test':
                self.targets = torch.LongTensor(targets)

        self.data_type = data_type
        self.transform = transform
        self.generator = generator

    def _get_video_item(self, index):
        return get_video_item(index, self.data, self.data_type, self.generator)
        
    def __getitem__(self, index):
        
        item = self._get_video_item(index)
        x = {"video": item}
        
        if self.transform:
            x = self.transform(x)
        
        if not self.generator or (self.generator and self.data_type=='test'):
            label = self.targets[index]
        else:
            label = torch.LongTensor(self.data[index][1])
            label = label.reshape((1))
        x["label"] = label

        return x
    
    def __len__(self):
        return len(self.data)

def get_video_item(index, data, data_type, generator, debug=False):
    if not generator:
        item = data[index].permute(3, 0, 1, 2)
    else:
        # self.data is of type DataGenerator
        if data_type=='test':
            item = torch.from_numpy(np.asarray(data[index]))
        else:
            item = torch.from_numpy(np.asarray(data[index][0]))
        item = torch.squeeze(item)
        item = item.permute(3, 0, 1, 2)
    item = item.float() # item.shape=[3,16,224,224]
    return item
"""

"""
def test_video_based_model(
        test_data,
        training_result,
        model_path, 
        generator
    ):
    
    val_transform = training_result["val_transform"]
    trainer = training_result["trainer"]
    
    if not generator:
        test_dataset = TorchVideoDataset(test_data[0][0], test_data[1], 'test', generator=generator, transform=val_transform)
    else:
        #test_data[0] contains the DataGenerator object
        test_dataset = TorchVideoDataset(test_data[0], test_data[1], 'test', generator=generator, transform=val_transform)


    logits = trainer.predict(test_dataset)
    probs = softmax(logits[0], axis=1)
    probs = probs[:,1]
    #preds = np.where(probs > 0.5, 1, 0)
    preds = np.expand_dims(probs, axis=1)
    #targets = np.squeeze(test_data[1])
    targets = test_data[1]

    with open(os.path.join(model_path["saved_files_path"], 'predictions.pkl'), 'wb') as picklefile:
        pickle.dump({'predictions': preds,
                     'test_data': targets}, picklefile)

    trainer.save_model()
    test_results = trainer.evaluate(test_dataset)
    trainer.log_metrics("test", test_results)
    trainer.save_metrics("test", test_results)
    trainer.save_state()

    acc = test_results["eval_accuracy"]
    auc = test_results["eval_auc"]
    f1 = test_results["eval_f1"]
    precision = test_results["eval_precision"]
    recall = test_results["eval_recall"]

    return acc, auc, f1, precision, recall
"""

def get_video_transforms(model, image_processor, config):
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = config.num_frames # model.config.num_frames
    #sample_rate = 4
    #fps = 30
    #clip_duration = num_frames_to_sample * sample_rate / fps

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        #UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        #RandomShortSideScale(min_size=256, max_size=320),
                        #RandomCrop(resize_to),
                        #RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        #UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        #Resize(resize_to),
                    ]
                ),
            ),
        ]
    )

    return train_transform, val_transform
