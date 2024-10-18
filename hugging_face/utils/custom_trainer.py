
from transformers import Trainer

from hugging_face.utilities import compute_loss as _compute_loss


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        self.class_w = kwargs["class_w"]
        del kwargs["class_w"]
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get('logits')
        # compute custom loss
        cmp_loss_ret = _compute_loss(outputs, logits, labels, None, 2, None,
                                     class_w=self.class_w)
        loss = cmp_loss_ret[0]
        return (loss, outputs) if return_outputs else loss
