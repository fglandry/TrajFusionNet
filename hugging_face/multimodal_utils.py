import numpy as np
import os
import pickle
from scipy.special import softmax
import torch
from torch.utils.data import Dataset

from hugging_face.utilities import compute_huggingface_metrics
from hugging_face.image_utils import convert_img_to_format_used_by_transform, get_image_transforms
from hugging_face.video_utils import get_video_transforms
from transformers import AutoImageProcessor


def get_multimodal_datasets(data_train, data_val, model, generator, 
                            img_model_config, video_model_config, 
                            image_processor=None):

    assert generator

    # Timesformer ------------------------------------------
    # Get image processor
    model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
    video_processor = AutoImageProcessor.from_pretrained(model_ckpt)

    train_video_transform, val_video_transform = \
        get_video_transforms(model, video_processor, video_model_config)

    # VAN ---------------------------------------------------
    # Get image processor
    if not image_processor:
        img_proc_ckpt = "Visual-Attention-Network/van-base"
        image_processor = AutoImageProcessor.from_pretrained(img_proc_ckpt)

    train_img_transform, val_img_transform = \
        get_image_transforms(image_processor,
                             num_channels=img_model_config.num_channels,
                             config=img_model_config)

    train_dataset = TorchMultiModelDataset(
        data_train['data'][0], None, 'train', 
        generator=generator,
        video_transform=train_video_transform,
        img_transform=train_img_transform)
    val_dataset = TorchMultiModelDataset(
        data_val, None, 'val', 
        generator=generator, 
        video_transform=val_video_transform,
        img_transform=val_img_transform)
    
    val_transforms_dicts = {
        "val_img_transform": val_img_transform,
        "val_video_transform": val_video_transform
    }

    return train_dataset, val_dataset, val_transforms_dicts


class TorchMultiModelDataset(Dataset):
    def __init__(self, data, targets, data_type, generator=False, 
                 video_transform=None, img_transform=None):
        assert generator

        self.data = data # data is a Generator object, casting is done in __getitem__
        if data_type != 'test':
            self.targets = None
        elif data_type == 'test':
            self.targets = torch.LongTensor(targets)

        self.data_type = data_type
        self.video_transform = video_transform
        self.img_transform = img_transform
        self.generator = generator
        self.debug = False
        if len(self.data.input_type_list) >= 3:
            self.video_context = "box" not in self.data.input_type_list[3]
            self.timeseries_context = self.data.input_type_list[-1].startswith("scene_graph") # todo verify
            self.is_scene_context_with_segmentation = "scene_context_with_segmentation" in self.data.input_type_list[2] 
        else:   
            self.video_context = "box" not in self.data.input_type_list[2] # self.data.input_type_list[2] != "box_speed_pose"
            self.timeseries_context = False
            self.is_scene_context_with_segmentation = False

    def _get_video_item(self, index):
        """
        self.data is of type DataGenerator
        """
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0] 

        # todo: assuming the second item in list corresponds to the context image
        item = torch.from_numpy(np.asarray(item[1]))

        item = torch.squeeze(item)
        item = item.permute(3, 0, 1, 2)
        item = item.float() # item.shape=[3,16,224,224]
        item = {"video": item}
        return item
    
    def _get_image_item(self, index):
        """
        self.data is of type DataGenerator
        """
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0]

        # todo: assuming the first item in list corresponds to the context image
        item = np.asarray(item[0]) 
        item = np.squeeze(item) 
    
        x = convert_img_to_format_used_by_transform(item, debug=False)
        return x

    def _get_timeseries_item(self, index):
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0]

        # todo: assuming the third or fourth item in list corresponds to the trajectory data
        if self.is_scene_context_with_segmentation:
            item = np.asarray(item[3])
        elif self.video_context:
            item = np.asarray(item[3])
        else:
            item = np.asarray(item[2])

        item = np.squeeze(item) 
    
        item = torch.from_numpy(item)
        item = item.float() # item.shape=[seq_len, n_features]
        item = {"timeseries": item}
        
        return item
    
    def _get_timeseries_context_item(self, index):
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0]

        # todo: assuming the third or fourth item in list corresponds to the trajectory data
        if self.is_scene_context_with_segmentation:
            item = np.asarray(item[3])
        elif self.video_context:
            item = np.asarray(item[3])
        else:
            item = np.asarray(item[2])

        item = np.squeeze(item) 
    
        item = torch.from_numpy(item)
        item = item.float() # item.shape=[seq_len, n_features]
        item = {"timeseries": item}
        
        return item
    
    def _get_video_context_item(self, index):
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0] 

        # todo: assuming the second item in list corresponds to the video context
        item = torch.from_numpy(np.asarray(item[1]))

        item = torch.squeeze(item)
        item = item.permute(3, 0, 1, 2)
        item = item.float() # item.shape=[3,16,224,224]
        item = {"video_context": item}
        return item

    def __getitem__(self, index):
        # Get image item
        image_item = self._get_image_item(index)
        if self.img_transform:
            image_item = self.img_transform(image_item)
        image_item = {
            "pixel_values": image_item
        }

        # Get video item
        video_item = self._get_video_item(index)
        if self.video_transform:
            video_item = self.video_transform(video_item)

        # Get video context item
        if self.video_context:
            video_context_item = self._get_video_context_item(index) 
            # Do not apply any transform

        # Get time series item
        timeseries_item = self._get_timeseries_item(index)

        # Get time series context item
        if self.timeseries_context:
            timeseries_context_item = self._get_timeseries_context_item(index)
        
        # Get label
        if not self.generator or (self.generator and self.data_type=='test'):
            label = self.targets[index]
        else:
            label = torch.LongTensor(self.data[index][1])
            label = label.reshape((1))

        # Get complete dictionary
        item = {}
        item.update(image_item)
        item.update(video_item)
        item.update(timeseries_item)
        if self.video_context:
            item.update(video_context_item)
        if self.timeseries_context:
            item.update(timeseries_context_item)
        item["label"] = label

        return item
    
    def __len__(self):
        return len(self.data)
    
    
class HuggingFaceMultimodalClassificationModel():

    def compute_metrics(self, eval_pred):
        return compute_huggingface_metrics(eval_pred)

    def collate_fn(self, examples):
        
        # Collate video-based data
        # permute to (num_frames, num_channels, height, width)
        video_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        
        # Collate image-based data
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        
        # Collate timeseries data
        timeseries_values = torch.stack([example["timeseries"] for example in examples])

        # Collate labels
        labels = torch.tensor([example["label"] for example in examples])
        #labels = [example["label"] for example in examples]
        #labels = torch.stack(labels) # is it needed? added
        #labels = torch.squeeze(labels) # added

        return_dict = {
            "pixel_values": pixel_values,
            "video_values": video_values,
            "timeseries_values": timeseries_values,
            "labels": labels
        }

        if "video_context" in examples[0]:
            # Collate video context data
            video_context_values = torch.stack(
                [example["video_context"].permute(1, 0, 2, 3) for example in examples]
            )
            return_dict.update({"video_context_values": video_context_values})
        return return_dict

def test_multimodal_model(
        test_data,
        training_result,
        model_path, 
        generator,
        save_results=True
    ):
    
    test_img_transform = training_result["val_transform"]["val_img_transform"]
    test_video_transform = training_result["val_transform"]["val_video_transform"]
    trainer = training_result["trainer"]
    
    test_dataset = TorchMultiModelDataset(
        test_data[0] if generator else test_data[0][0],
        test_data[1], 
        'test', 
        generator=generator,
        video_transform=test_video_transform,
        img_transform=test_img_transform)

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
