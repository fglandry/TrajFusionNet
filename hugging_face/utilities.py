
from datasets import load_metric
import numpy as np
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.timesformer.modeling_timesformer import TimesformerEmbeddings
from hugging_face.utils.focal_loss import FocalLoss, FocalLoss2


def get_class_labels_info():
    class_labels = ["no_cross", "cross"]
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

def compute_huggingface_metrics(eval_pred):
    metric_acc = load_metric("hugging_face/metrics/accuracy.py")
    metric_auc = load_metric("hugging_face/metrics/roc_auc.py")
    metric_f1 = load_metric("hugging_face/metrics/f1.py")
    metric_precision = load_metric("hugging_face/metrics/precision.py")
    metric_recall = load_metric("hugging_face/metrics/recall.py")
    eval_pred_tte = None

    if eval_pred.predictions.shape[-1] > 2: # review: multi-task learning
        # To get crossing metrics ...
        eval_pred_crossing = eval_pred.predictions[...,0:2]
        references = eval_pred.label_ids[0]

        if eval_pred.predictions.shape[-1] == 3:
            # To get tte metrics ...
            _eval_pred_tte = eval_pred.predictions[...,2]
            # predictions_tte = np.argmax(eval_pred_tte, axis=1)
            _references_tte = eval_pred.label_ids[1]

            # Remove non-crossing occurences
            eval_pred_tte = np.array([ele*31.0+30.0 for (idx, ele) in enumerate(list(_eval_pred_tte)) \
                            if _references_tte[idx]!=1])
            references_tte = np.array([ele*31.0+30.0 for ele in list(_references_tte) if ele!=1])
        elif eval_pred.predictions.shape[-1] == 6:
            # To get tte_pos metrics ...
            eval_pred_tte = eval_pred.predictions[..., 2:]
            references_tte = eval_pred.label_ids[1]
            eval_pred_tte = eval_pred_tte.flatten()
            references_tte = references_tte.flatten()
        
        metric_mse = load_metric("hugging_face/metrics/mse.py")
        metric_mae = load_metric("hugging_face/metrics/mae.py")
    else:
        eval_pred_crossing = eval_pred.predictions
        references = eval_pred.label_ids

    # To get crossing metrics
    predictions = np.argmax(eval_pred_crossing, axis=1)
    # prediction_scores = np.argmax(eval_pred_crossing, axis=1) # verify if that makes sense

    try:
        acc = metric_acc.compute(predictions=predictions, references=references)["accuracy"]
    except ValueError:
        acc = 0
    try:
        auc = metric_auc.compute(prediction_scores=predictions, references=references)["roc_auc"]
    except ValueError:
        auc = 0
    try:
        f1 = metric_f1.compute(predictions=predictions, references=references)["f1"]
    except ValueError:
        f1 = 0
    try:
        precision = metric_precision.compute(predictions=predictions, references=references)["precision"]
    except ValueError:
        precision = 0
    try:
        recall = metric_recall.compute(predictions=predictions, references=references)["recall"]
    except ValueError:
        recall = 0
    if type(eval_pred_tte) is np.ndarray:
        try:
            mse = metric_mse.compute(predictions=eval_pred_tte, references=references_tte)["mse"]
        except ValueError:
            mse = 0
        try:
            mae = metric_mae.compute(predictions=eval_pred_tte, references=references_tte)["mae"]
        except ValueError:
            mae = 0

    metrics = {
        "accuracy": acc,
        "auc": auc,
        "f1": f1, 
        "precision": precision,
        "recall": recall
    }
    if type(eval_pred_tte) is np.ndarray:
        metrics.update({
            "mse": mse,
            "mae": mae
        })
    return metrics

def compute_huggingface_forecast_metrics(eval_pred):  
    metric_mse = load_metric("hugging_face/metrics/mse.py")
    metric_mae = load_metric("hugging_face/metrics/mae.py")

    predictions = eval_pred.predictions
    references = eval_pred.label_ids

    # TODO: remove
    #predictions = predictions[:,-1,:]
    #references = references[:,-1,:]

    predictions = predictions.flatten()
    references = references.flatten()

    if len(predictions) > 75000000:
        print("WARNING: MSE and MAE calculation will be calculated on 25 percent of data due to memory constraints")
        predictions = predictions[0::4]
        references = references[0::4]

    try:
        mse = metric_mse.compute(predictions=predictions, references=references)["mse"]
    except ValueError:
        mse = 0
    try:
        mae = metric_mae.compute(predictions=predictions, references=references)["mae"]
    except ValueError:
        mae = 0

    metrics = {
        "mse": mse,
        "mae": mae
    }
    return metrics

def compute_loss(
        outputs,
        logits,
        labels,
        config,
        num_labels,
        return_dict,
        return_loss_only=False,
        tte_labels=None,
        tte_pos_labels=None,
        #tte_pred=None,
        #tte_pos_pred=None
        class_w=None,
        problem_type=""
    ):
    problem_type = config.problem_type if not problem_type else problem_type
    loss = None
    if not config:
        config = dotdict({"problem_type": None})
    if labels is not None:
        if problem_type is None:
            if num_labels == 1:
                problem_type = "regression"
            elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"

        if problem_type == "regression":
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss(weight=class_w)
            # loss_fct = FocalLoss(gamma=0.5, weights=class_w)
            # loss_fct = FocalLoss2(gamma=2, alpha=0.8)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            test = 10
        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        elif problem_type == "multi_objectives_tte":
            #loss_fct = CrossEntropyLoss()
            #num_labels = 2
            #logits = logits[:, 0:num_labels]
            #loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            
            # Calculate loss for crossing prediction
            crossing_loss_fct = CrossEntropyLoss()
            crossing_num_labels = 2
            crossing_logits = logits[:, 0:crossing_num_labels]
            crossing_targets = labels.view(-1)
            crossing_loss = crossing_loss_fct(
                crossing_logits.view(-1, crossing_num_labels), 
                crossing_targets)
            
            # Calculate loss for tte regression
            tte_loss_fct = MSELoss()
            tte_logits = logits[:, 2]
            tte_preds = tte_logits.squeeze().float()
            tte_targets = tte_labels.squeeze().float()
            crossing_indices = crossing_targets == 1
            preds = tte_preds[crossing_indices]
            targets = tte_targets[crossing_indices]
            
            tte_loss = tte_loss_fct(preds, targets)
            loss = crossing_loss + 2*tte_loss

        elif problem_type == "multi_objectives_tte_pos":

            # Calculate loss for crossing prediction
            crossing_loss_fct = CrossEntropyLoss()
            crossing_num_labels = 2
            crossing_logits = logits[:, 0:crossing_num_labels]
            crossing_targets = labels.view(-1)
            crossing_loss = crossing_loss_fct(
                crossing_logits.view(-1, crossing_num_labels), 
                crossing_targets)
            
            # Calculate loss for tte regression
            tte_pos_loss_fct = MSELoss()
            tte_pos_logits = logits[:, 2:] # tte_pos_pred 
            tte_pos_preds = torch.flatten(tte_pos_logits).float()
            tte_pos_targets = torch.flatten(tte_pos_labels).float()
            tte_pos_loss = tte_pos_loss_fct(tte_pos_preds, tte_pos_targets)

            loss = crossing_loss + tte_pos_loss

        elif problem_type == "trajectory":
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
                """
                loss_target_1 = loss_fct(logits[:,-1,:], labels[:,-1,:]) # t = 60
                loss_target_2 = loss_fct(logits[:,29,:], labels[:,29,:]) # t = 30
                loss_pred_horizon = loss_fct(logits, labels)
                loss = 0.5*loss_pred_horizon + 0.25*loss_target_1 + 0.25*loss_target_2
                """
                """
                loss_target_1 = loss_fct(logits[:,-1,:], labels[:,-1,:]) # t = 60
                loss_target_2 = loss_fct(logits[:,29,:], labels[:,29,:]) # t = 30
                loss_pred_horizon = loss_fct(logits, labels)
                loss = 0.5*loss_pred_horizon + 0.25*loss_target_1 + 0.25*loss_target_2
                """
                """
                loss_pre_pred_horizon = loss_fct(logits[:,0:30,:], labels[:,0:30,:])
                loss_pred_horizon = loss_fct(logits[:,30:,:], labels[:,30:,:])
                loss = 0.33*loss_pre_pred_horizon + 0.67*loss_pred_horizon
                """
                """
                traj_logits = logits[:,:-1,:]
                traj_labels = labels[:,:-1,:]
                objective_logit = logits[:,-1,:]
                objective_label = labels[:,-1,:]
                loss_traj = loss_fct(traj_logits, traj_labels)
                loss_objective = loss_fct(objective_logit, objective_label)
                loss = loss_traj + loss_objective
                """
            
            
    if return_loss_only:
        return loss

    if not return_dict:
        output = (logits,) + outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return ImageClassifierOutputWithNoAttention(
        loss=loss, 
        logits=logits, 
        hidden_states=None
    )

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class CustomPreTrainedModel(PreTrainedModel):
    """
    ** Copied from huggingface's transformer lib
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # config_class = TimesformerConfig
    base_model_prefix = "custom_hf"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, TimesformerEmbeddings):
            nn.init.trunc_normal_(module.cls_token, std=self.config.initializer_range)
            nn.init.trunc_normal_(module.position_embeddings, std=self.config.initializer_range)
            module.patch_embeddings.apply(self._init_weights)

    """
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, TimesformerEncoder):
            module.gradient_checkpointing = value
    """

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

