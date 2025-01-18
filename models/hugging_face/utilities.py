from datasets import load_metric
import numpy as np
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import PreTrainedModel, TimeSeriesTransformerConfig
from transformers.trainer_utils import EvalPrediction
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention
from transformers.models.timesformer.modeling_timesformer import TimesformerEmbeddings

METRICS_REL_DIR = "models/hugging_face/metrics"


def get_class_labels_info():
    class_labels = ["no_cross", "cross"]
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label

def compute_huggingface_metrics(eval_pred: EvalPrediction):
    metric_acc = load_metric(f"{METRICS_REL_DIR}/accuracy.py")
    metric_auc = load_metric(f"{METRICS_REL_DIR}/roc_auc.py")
    metric_f1 = load_metric(f"{METRICS_REL_DIR}/f1.py")
    metric_precision = load_metric(f"{METRICS_REL_DIR}/precision.py")
    metric_recall = load_metric(f"{METRICS_REL_DIR}/recall.py")
    eval_pred_tte = None

    if eval_pred.predictions.shape[-1] > 2:

        # To get crossing metrics ...
        eval_pred_crossing = eval_pred.predictions[...,0:2]
        references = eval_pred.label_ids[0]

        if eval_pred.predictions.shape[-1] == 3:
            # To get tte metrics ...
            _eval_pred_tte = eval_pred.predictions[...,2]
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
        
        metric_mse = load_metric(f"{METRICS_REL_DIR}/mse.py")
        metric_mae = load_metric(f"{METRICS_REL_DIR}/mae.py")
    else:
        eval_pred_crossing = eval_pred.predictions
        references = eval_pred.label_ids

    # To get crossing metrics
    predictions = np.argmax(eval_pred_crossing, axis=1)

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

def compute_huggingface_forecast_metrics(
        eval_pred: EvalPrediction
    ):  
    metric_mse = load_metric(f"{METRICS_REL_DIR}/mse.py")
    metric_mae = load_metric(f"{METRICS_REL_DIR}/mae.py")

    predictions = eval_pred.predictions
    references = eval_pred.label_ids

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
        logits: torch.Tensor,
        labels: torch.Tensor,
        config: TimeSeriesTransformerConfig,
        num_labels: int,
        return_dict: bool,
        return_loss_only: bool = False,
        class_w=None,
        problem_type: str = ""
    ):
    problem_type = config.problem_type if not problem_type else problem_type
    loss = None
    if not config:
        config = dotdict({"problem_type": None})

    if labels is not None:
        if problem_type is None:
            if num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"

        if problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss(weight=class_w)
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        elif problem_type == "multi_label_classification":
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)

        elif problem_type == "trajectory":
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
            
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

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
