import os
import pickle
import time
import cv2

import numpy as np
from scipy.special import softmax
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from models.hugging_face.image_utils import convert_img_to_format_used_by_transform, get_image_transforms

from models.hugging_face.utilities import compute_huggingface_metrics, compute_huggingface_forecast_metrics
from models.hugging_face.video_utils import get_video_transforms
from utils.utils import open_pickle_file

def get_timeseries_datasets(data_train, data_val, model, generator, 
                            video_model_config, 
                            get_image_transform=False,
                            get_seg_maps_transforms = False,
                            img_model_config=None,
                            ignore_sem_map=False,
                            dataset_statistics=None):

    assert generator

    train_video_transform, val_video_transform = None, None
    if video_model_config:
        # Timesformer ------------------------------------------
        # Get image processor
        model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
        image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

        train_video_transform, val_video_transform = \
            get_video_transforms(model, image_processor, video_model_config)

    if get_image_transform:
        img_proc_ckpt = "Visual-Attention-Network/van-base"
        image_processor = AutoImageProcessor.from_pretrained(img_proc_ckpt)

        #model_ckpt = "facebook/convnextv2-tiny-1k-224"
        #image_processor = AutoImageProcessor.from_pretrained(model_ckpt)

        train_img_transform, val_img_transform = \
            get_image_transforms(image_processor, img_model_config)

    if get_seg_maps_transforms:    
        # todo: revisit
        train_segm_map_transform, val_segm_map_transform = \
            get_image_transforms(image_processor, img_model_config, 
                                 dataset_statistics=dataset_statistics,
                                 modality="scene_context_with_segmentation_v0")
        train_segm2_map_transform, val_segm2_map_transform = \
            get_image_transforms(image_processor, img_model_config, 
                                 dataset_statistics=dataset_statistics,
                                 modality="scene_context_with_segmentation_v3")

    train_dataset = TorchTimeseriesDataset(
        data_train['data'][0], None, 'train', 
        generator=generator,
        transform=train_video_transform,
        img_transform=train_img_transform if get_image_transform else None,
        segm_transform=train_segm_map_transform if get_seg_maps_transforms else None,
        segm2_transform=train_segm2_map_transform if get_seg_maps_transforms else None,
        ignore_sem_map=ignore_sem_map,
        dataset_statistics=dataset_statistics)
    val_dataset = TorchTimeseriesDataset(
        data_val, None, 'val', 
        generator=generator, 
        transform=val_video_transform,
        img_transform=val_img_transform if get_image_transform else None,
        segm_transform=val_segm_map_transform if get_seg_maps_transforms else None,
        segm2_transform=val_segm2_map_transform if get_seg_maps_transforms else None,
        ignore_sem_map=ignore_sem_map,
        dataset_statistics=dataset_statistics)
    
    val_transforms_dicts = {
        "val_img_transform": val_img_transform if get_image_transform else None,
        "val_video_transform": val_video_transform if video_model_config else None,
        "val_segm_map_transform": val_segm_map_transform if get_seg_maps_transforms else None,
        "val_segm_map2_transform": val_segm2_map_transform if get_seg_maps_transforms else None,
        "dataset_statistics": dataset_statistics
    }

    return train_dataset, val_dataset, val_transforms_dicts

class HuggingFaceTimeSeriesModel():
    
    def compute_metrics(self, eval_pred):
        if self.forecast:
            return compute_huggingface_forecast_metrics(eval_pred)
        else:
            return compute_huggingface_metrics(eval_pred)

    def collate_fn(self, examples):
        video_data = "video" in examples[0]
        timeseries_context = "timeseries_context" in examples[0]
        previous_timeseries_context = "previous_timeseries_context" in examples[0]
        image_context = "image_context" in examples[0]
        previous_image_context = "previous_image_context" in examples[0]
        segm_context = "segmentation_context" in examples[0]
        segm_context_2 = "segmentation_context_2" in examples[0]
        normalized_trajectory = "normalized_trajectory_values" in examples[0]
        is_tte_label = "tte_label" in examples[0]
        is_tte_pos_label = "tte_pos_label" in examples[0]

        trajectory_values = torch.stack([example["pixel_values"] for example in examples])
        if examples[0]["label"].shape[-1] > 1:
            self.forecast = True
            labels = torch.FloatTensor(np.array([example["label"] for example in examples]))
        else:
            self.forecast = False
            labels = torch.LongTensor(np.array([example["label"] for example in examples]))

        return_dict = {
            "trajectory_values": trajectory_values, 
            "labels": labels,
        }

        if video_data:
            # Collate video-based data
            # permute to (num_frames, num_channels, height, width)
            video_values = torch.stack(
                [example["video"].permute(1, 0, 2, 3) for example in examples]
            )
            return_dict.update({"video_values": video_values})
        if timeseries_context:
            timeseries_context_values = torch.stack([example["timeseries_context"] for example in examples])
            return_dict.update({"timeseries_context": timeseries_context_values})
        if previous_timeseries_context:
            previous_timeseries_context_values = torch.stack([example["previous_timeseries_context"] for example in examples])
            return_dict.update({"previous_timeseries_context": previous_timeseries_context_values})
        if image_context:
            image_context_values = torch.stack([example["image_context"] for example in examples])
            return_dict.update({"image_context": image_context_values})
        if previous_image_context:
            previous_image_context_values = torch.stack([example["previous_image_context"] for example in examples])
            return_dict.update({"previous_image_context": previous_image_context_values})
        if segm_context:
            segm_context_values = torch.stack([example["segmentation_context"] for example in examples])
            return_dict.update({"segmentation_context": segm_context_values})
        if segm_context_2:
            segm_context_2_values = torch.stack([example["segmentation_context_2"] for example in examples])
            return_dict.update({"segmentation_context_2": segm_context_2_values})
        if normalized_trajectory:
            normalized_trajectory_values = torch.stack([example["normalized_trajectory_values"] for example in examples])
            return_dict.update({"normalized_trajectory_values": normalized_trajectory_values})
        if is_tte_label:
            tte_labels = torch.tensor(np.array([example["tte_label"] for example in examples]))
            return_dict.update({"tte_labels": tte_labels})
        if is_tte_pos_label:
            tte_pos_labels = torch.tensor(np.array([example["tte_pos_label"] for example in examples]))
            return_dict.update({"tte_pos_labels": tte_pos_labels})
        return return_dict
        

class TorchTimeseriesDataset(Dataset):
    def __init__(self, data, targets, data_type, 
                 generator=False, transform=None, img_transform=None, ignore_sem_map=False, 
                 complete_data=None, segm_transform=None, segm2_transform=None, 
                 dataset_statistics=None, debug=False):
        if not generator:
            self.data = torch.from_numpy(data)
            self.targets = torch.LongTensor(targets)
        else:
            self.data = data # data is a Generator object, casting is done in __getitem__
            if data_type != 'test':
                self.targets = None
            elif data_type == 'test':
                # self.targets = torch.LongTensor(targets)
                self.targets = targets

        # Verify if we have video data
        self.video_data = False
        self.context_image = False
        self.video_embeddings = False
        self.segm_map, self.segm_map2 = "", ""
        self.model = ""

        self.timeseries_context = False
        self.timeseries_double_context = False
        self.context_image = False
        self.previous_context_image = False
        if len(self.data.input_type_list) == 2: # TODO: recode this in a better way
            self.context_image = True
            self.model = "TrajectoryTransformerV3"
        elif len(self.data.input_type_list) == 3:
            #self.timeseries_context = "scene_graph" in self.data.input_type_list[1]
            #self.timeseries_double_context = "scene_graph_doubled" in self.data.input_type_list[1]
            self.context_image = "scene_context" in self.data.input_type_list[0]
            if ("scene_context" in self.data.input_type_list[1] and "_doubled" in self.data.input_type_list[1]) \
                or "local_context" in self.data.input_type_list[1]:
                self.previous_context_image = True
            self.model = "VanMultiscale"
        elif len(self.data.input_type_list) == 4:
            self.context_image = True
            self.model = "TrajectoryTransformerV3b"
            self.timeseries_context = "scene_graph" in self.data.input_type_list[1]
        elif len(self.data.input_type_list) >= 5:
            self.timeseries_context = "scene_graph" in self.data.input_type_list[2]
            self.timeseries_double_context = "scene_graph_doubled" in self.data.input_type_list[3]
            self.context_image = "scene_context" in self.data.input_type_list[4]
            self.previous_context_image = "scene_context_doubled" in self.data.input_type_list[5]
            self.segm_map = "scene_context_with_segmentation_v0" in self.data.input_type_list[0] 
            self.segm_map2 = "scene_context_with_segmentation_v3" in self.data.input_type_list[7]

        if len(self.data[0][0]) > 1 and not ignore_sem_map and not \
            self.model.startswith("TrajectoryTransformerV3") and not \
                self.model == "VanMultiscale":
            self.video_data = True
        elif self.video_embeddings: # len(self.data[0][0]) >= 8: # ToDo: check if that makes sense
            self.video_data = True

        self.data_type = data_type
        self.img_transform = img_transform
        self.segm_transform = segm_transform
        self.segm2_transform = segm2_transform
        self.item_norm = True
        self.video_transform = transform if not self.video_embeddings else None
        self.generator = generator
        self.ignore_sem_map = ignore_sem_map
        self.segformer_model = None
        self.complete_data = complete_data
        self.tte_label = False
        self.tte_pos_label = False
        self.custom_labels = False
        self.dataset_statistics = dataset_statistics
        self.debug = debug

    def _get_trajectory_features_item(self, index):
        return get_trajectory_features_item(index, self.data, self.data_type, 
                                            self.generator, self.video_data, 
                                            self.ignore_sem_map,
                                            timeseries_context=self.timeseries_context,
                                            timeseries_double_context=self.timeseries_double_context,
                                            dataset_statistics=self.dataset_statistics)
    
    def __getitem__(self, index):
        # Get timeseries item
        item, item_norm = self._get_trajectory_features_item(index)
        item = item.type(torch.FloatTensor) # required by Time Series Library models
        item = {"pixel_values": item} # 'pixel_values' does not really make sense here, but is expected by the huggingface model class
        
        # Get normalized trajectory values
        if self.item_norm:
            item_norm = item_norm.type(torch.FloatTensor)
            item_norm = {"normalized_trajectory_values": item_norm}
        
        # Get video item
        if self.video_data:
            video_item = self._get_video_item(index)
            if self.video_transform:
                video_item = self.video_transform(video_item)
        
        # Get timeseries context item
        if self.timeseries_context:
            timeseries_context_item = self._get_timeseries_context_item(index)

        # Get image context item
        if self.context_image:
            context_image_item = self._get_context_image_item(index)
        if self.previous_context_image:
            previous_context_image_item = self._get_context_image_item(index, get_previous_context=True)
        if self.segm_map:
            segm_map_item = self._get_segm_map_item(index)
        if self.segm_map2:
            segm_map2_item = self._get_segm_map2_item(index)

        labels_dict = self._process_labels(index,
                                           dataset_statistics=self.dataset_statistics)

        if self.video_data:
            item.update(video_item)
        if self.timeseries_context:
            item.update(timeseries_context_item)
        if self.context_image:
            item.update(context_image_item)
        if self.previous_context_image:
            item.update(previous_context_image_item)
        if self.segm_map:
            item.update(segm_map_item)
        if self.segm_map2:
            item.update(segm_map2_item)
        if self.item_norm:
            item.update(item_norm)
        
        item["label"] = labels_dict["label"]
        
        if self.tte_label:
            item["tte_label"] = labels_dict["tte_label"]
        if self.tte_pos_label:
            item["tte_pos_label"] = labels_dict["tte_pos_label"]
        if self.custom_labels: # TODO: remove
            item["label"] = labels_dict["label"][-1,:]
            item["label"] = np.expand_dims(item["label"], 0)

        return item
    
    def _process_labels(self, index,
                        dataset_statistics=None,
                        normalization_type="z_score"):
        if not self.generator or (self.generator and self.data_type=='test'):
            label = self.targets[index]
        else:
            #label = torch.LongTensor(self.data[index][1])
            #label = label.reshape((1))
            label = self.data[index][1]
            label = np.squeeze(label, axis=0)
        
        # If we have forecast labels ...
        if label.shape[-1] > 1:
            label = normalize_trajectory_data(label, normalization_type, 
                                              dataset_statistics=dataset_statistics)

        # Extract tte labels from labels variable
        if label.shape[0] > 1:
            if label.shape[0] == 2:
                self.tte_label = True
                tte_label = label[1, ...]
                label = label[0, ...]
            elif label.shape[0] == 5:
                self.tte_pos_label = True
                tte_pos_label = label[1:, ...] # todo remove
                label = label[0, ...]
        
        return {
            "label": label,
            "tte_label": tte_label if self.tte_label else None,
            "tte_pos_label": tte_pos_label if self.tte_pos_label else None
        }


    def _get_timeseries_context_item(self, index, 
                                     get_double_context_from_previous_index=False):
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0]

        # todo: assuming the second item in list corresponds to the timeseries context data
        obs_input_type_index = 1
        if self.timeseries_double_context:
            obs_input_type_index = 2
        context_item = np.asarray(item[obs_input_type_index])

        context_item = np.squeeze(context_item) 
        context_item = torch.from_numpy(context_item)
        context_item = context_item.float() # item.shape=[seq_len, n_features]

        if self.timeseries_double_context:
            if get_double_context_from_previous_index:
                previous_index = self.get_previous_index(index)
                prev_item = self.data[previous_index]
                prev_item = prev_item[0] if self.data_type!='test' else prev_item[0]
                prev_item = np.asarray(prev_item[1])
                prev_item = np.squeeze(prev_item) 
                prev_item = torch.from_numpy(prev_item)
                prev_item = prev_item.float() # item.shape=[seq_len, n_features]
                item = {
                    "timeseries_context": context_item,
                    "previous_timeseries_context": prev_item
                }
            else:
                double_context_item = self.get_previous_context_item(item)
                item = {
                    "timeseries_context": context_item,
                    "previous_timeseries_context": double_context_item
                }
        else:
            item = {"timeseries_context": context_item}
        
        return item
    
    def get_previous_context_item(self, item, debug=False):
        double_context_item = np.asarray(item[3])
        double_context_item = np.squeeze(double_context_item) 
        double_context_item = torch.from_numpy(double_context_item)
        double_context_item = double_context_item.float() # item.shape=[seq_len, n_features]
            
        return double_context_item
    
    def get_previous_index(self, index, debug=False):
        previous_index = self._get_previous_index(index)

        if debug:
            img1 = open_pickle_file(self.data.data[0][previous_index][0])
            img2 = open_pickle_file(self.data.data[0][index][0])
            img3 = open_pickle_file(self.data.data[2][previous_index][0]) # context img
            img4 = open_pickle_file(self.data.data[2][index][0]) # context img

            if not self.segformer_model:
                raise Exception("Not implemented in this repo")
                self.segformer_model = SegformerForSemanticSegmentationWrapper()
            self.segformer_model.display_segmentation_map(img1, img3, get_img_combined_with_segmentation_map=True)
            self.segformer_model.display_segmentation_map(img2, img4, get_img_combined_with_segmentation_map=True)
            #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/img_output_{str(time.time()).replace('.', '_')}1.png", img3)  
            #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/img_output_{str(time.time()).replace('.', '_')}2.png", img4)  

        return previous_index
    
    def _get_previous_index(self, index):
        overlap = 0.8 # ToDo: get values from opts, opts['overlap'] # if data_type == 'train' else 0.0
        obs_length = 16
        previous_index_same_video = -1
        olap_res = obs_length if overlap == 0 else int((1 - overlap) * obs_length)
        olap_res = 1 if olap_res < 1 else olap_res
        rewind = 2 * obs_length // olap_res 
        previous_index = index - rewind if index - rewind >= 0 else 0

        for p in range(previous_index, index):
            current_video_id = self.data.data[0][index][0].split("/")[-2]
            current_ped_id = self.complete_data["ped_id"][index][-1][0]
            previous_video_id = self.data.data[0][p][0].split("/")[-2]
            previous_ped_id = self.complete_data["ped_id"][p][-1][0]
            if current_video_id != previous_video_id or current_ped_id != previous_ped_id:
                continue
            else:
                #if index - p < olap_res:
                #    previous_index_same_video = index # use same image
                #else:
                previous_index_same_video = p
                break
        
        if previous_index_same_video == -1:
            previous_index_same_video = index

        return previous_index_same_video
    
    def _get_video_item(self, index):
        """
        self.data is of type DataGenerator
        """
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0] 

        # todo: assuming the n item in list corresponds to the context image
        item = torch.from_numpy(np.asarray(item[6]))

        item = torch.squeeze(item)
        item = item.permute(3, 0, 1, 2)
        item = item.float() # item.shape=[3,16,224,224]
        item = {"video": item}
        return item
    
    def _get_context_image_item(self, index, get_previous_context=False, debug=False):
        """
        self.data is of type DataGenerator
        """
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0]

        # todo: assuming the 'n' item in list corresponds to the context image
        if get_previous_context:
            if self.model == "VanMultiscale":
                item = np.asarray(item[1])
            else:
                item = np.asarray(item[5])
        else:
            if self.model == "TrajectoryTransformerV3" or self.model == "VanMultiscale":
                item = np.asarray(item[0])
            elif self.model == "TrajectoryTransformerV3b":
                item = np.asarray(item[2])
            else:
                item = np.asarray(item[4])

        item = np.squeeze(item)

        """
        if debug and index == 80:
            sem_item = np.squeeze(sem_item)
            cv2.imwrite(f"/home/francois/MASTER/sem_imgs/img_output_{str(time.time()).replace('.', '_')}.png", item)
            #cv2.imwrite(f"/home/francois/MASTER/sem_imgs/img_output_{str(time.time()).replace('.', '_')}2.png", sem_item)
        """

        x = convert_img_to_format_used_by_transform(item, debug=False)
        
        if self.img_transform:
            x = self.img_transform(x)
        
        if get_previous_context:
            image_item = {
                "previous_image_context": x
            }
        else:
            image_item = {
                "image_context": x
            }

        return image_item
    
    def _get_segm_map_item(self, index):
        """
        self.data is of type DataGenerator
        """
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0]

        # todo: assuming the first item in list corresponds to the segmentation map
        item = np.asarray(item[0])
        item = np.squeeze(item)

        x = convert_img_to_format_used_by_transform(item, debug=False)
        
        if self.segm_transform:
            x = self.segm_transform(x)
        
        # Copy single channel to get 3 channels
        #if torch.is_tensor(x):
        #    x = x.repeat(3, 1, 1)
        #else:
        #    test = 10

        image_item = {
            "segmentation_context": x
        }

        return image_item
    
    def _get_segm_map2_item(self, index):
        """
        self.data is of type DataGenerator
        """
        item = self.data[index]
        item = item[0] if self.data_type!='test' else item[0]

        # todo: assuming the 8th item in list corresponds to the segmentation map
        item = np.asarray(item[7])
        item = np.squeeze(item)

        x = convert_img_to_format_used_by_transform(item, debug=False)
        
        if self.segm2_transform:
            x = self.segm2_transform(x)

        image_item = {
            "segmentation_context_2": x
        }

        return image_item
    
    def __len__(self):
        return len(self.data)


class TimeSeriesLibraryConfig(dict):
    def __init__(self, *args, **kwargs):
        super(TimeSeriesLibraryConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_trajectory_features_item(index, data, data_type, generator, 
                                 video_data, ignore_sem_map, 
                                 timeseries_context=False,
                                 timeseries_double_context=False,
                                 dataset_statistics=None,
                                 normalization_type="z_score", # "0_1_scaling"
                                 debug=False):
    # assert dataset_statistics
    item = data[index]
    item = item[0] if data_type!='test' else item[0]

    if ignore_sem_map:
        item = [item[-1]]
    if data_type=="test" and type(item) is list:
        if len(item) == 2:
            item = item[1]
        elif len(item) >= 3:
            item = item[-1]
    
    # item = np.asarray(item[1]) if video_data else np.asarray(item)
    item = np.asarray(item)
    item = np.squeeze(item) 

    # transform trajectory data (normalization)
    if normalization_type:
        item_norm = normalize_trajectory_data(item, normalization_type, 
                                              dataset_statistics=dataset_statistics)
        item_norm = torch.from_numpy(item_norm)
    else:
        item_norm = None

    item = torch.from_numpy(item)
    return item, item_norm

def normalize_trajectory_data(
        item: np.ndarray, 
        normalization_type: str, 
        dataset_statistics=None):
    """ Transform trajectory data by normalizing values
    Args:
        item [np.ndarray]: trajectory sequence data
        normalization_type [str]: type of normalization (z_score, 0_1_scaling, etc.)
        dataset_statistics [dict]: dataset statistics
    Returns:
        item_norm [np.ndarray]: normalized trajectory sequence data
    """
    
    if dataset_statistics:
        if normalization_type=="z_score":
            mean = np.array(dataset_statistics["dataset_means"]["trajectory"])
            std_dev = np.array(dataset_statistics["dataset_std_devs"]["trajectory"])
            item_norm = (item - mean) / std_dev
        else:
            raise Exception
    else:
        item_norm = item

    return item_norm


def denormalize_trajectory_data(
        item, 
        normalization_type, 
        dataset_statistics
    ):
    """ Denormalize predicted trajectory data
    Args:
        item [np.ndarray]: normalized predicted trajectory data
        normalization_type [str]: type of normalization (z_score, 0_1_scaling, etc.)
        dataset_statistics [dict]: dataset statistics
    Returns:
        item_denormed [np.ndarray]: denormalized trajectory sequence data
    """

    if normalization_type=="z_score":
        mean = np.array(dataset_statistics["dataset_means"]["trajectory"])
        std_dev = np.array(dataset_statistics["dataset_std_devs"]["trajectory"])
        item_denormed = (item * std_dev) + mean
    else:
        raise Exception

    return item_denormed
    
def test_time_series_based_model(
        test_data,
        training_result,
        model_path, 
        generator,
        save_results=True,
        ignore_sem_map=False,
        complete_data=None
    ):
    
    if training_result["val_transform"]:
        val_video_transform = training_result["val_transform"]["val_video_transform"]
        val_img_transform = training_result["val_transform"]["val_img_transform"]
        val_segm_map_transform = training_result["val_transform"]["val_segm_map_transform"]
        val_segm_map2_transform = training_result["val_transform"]["val_segm_map2_transform"]
        dataset_statistics = training_result["val_transform"]["dataset_statistics"]
    else:
        val_video_transform, val_img_transform = None, None
        val_segm_map_transform, val_segm_map2_transform = None, None
    trainer = training_result["trainer"]
    
    if not generator:
        _test_data = test_data[0][0]
    else:
        _test_data = test_data[0] #test_data[0] contains the DataGenerator object
    
    test_dataset = TorchTimeseriesDataset(_test_data, test_data[1], 'test', 
                                          generator=generator, 
                                          transform=val_video_transform,
                                          img_transform=val_img_transform,
                                          segm_transform=val_segm_map_transform,
                                          segm2_transform=val_segm_map2_transform,
                                          ignore_sem_map=ignore_sem_map,
                                          complete_data=complete_data,
                                          dataset_statistics=dataset_statistics)

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

    acc = test_results.get("test_accuracy", 0)
    auc = test_results.get("test_auc", 0)
    f1 = test_results.get("test_f1", 0)
    precision = test_results.get("test_precision", 0)
    recall = test_results.get("test_recall", 0)

    return acc, auc, f1, precision, recall
