import math
from typing import Any

from torch import nn
from transformers import Trainer, TrainingArguments
from transformers.dependency_versions_check import dep_version_check
from transformers.utils import is_sagemaker_mp_enabled
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import Trainer
from transformers.trainer_utils import ShardedDDPOption
from transformers.trainer_pt_utils import get_parameter_names


def override_create_optimizer(trainer_obj: Trainer,
                              disable_vam_branch: bool = False,
                              epoch_index: int = 0):
    """ Change implementation of the create_optimizer() method in the Huggingface 
        library's Trainer class to change learning rate in some layers depending
        on the epoch
    """

    opt_model = trainer_obj.model_wrapped if is_sagemaker_mp_enabled() else trainer_obj.model

    if trainer_obj.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        if disable_vam_branch:
            # disable learning in VAM module by setting learning rate to 0 during first epochs
            if epoch_index <= 10:
                params_to_lower_lr = [n for idx, (n, p) in enumerate(opt_model.named_parameters()) if \
                                    (p.requires_grad and "van_output_embed" in n)]
            else:
                params_to_lower_lr = []
        else:
            #params_to_lower_lr = [n for idx, (n, p) in enumerate(opt_model.named_parameters()) if \
            #                      (p.requires_grad and n.startswith("transformer.traj_TF"))]
            params_to_lower_lr = []
            
            
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and not n in params_to_lower_lr)
                ],
                "weight_decay": trainer_obj.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and not n in params_to_lower_lr)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (p.requires_grad and n in params_to_lower_lr)
                ],
                'lr': 0.0
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer_obj.args)

        if trainer_obj.sharded_ddp == ShardedDDPOption.SIMPLE:
            trainer_obj.optimizer = OSS(
                params=optimizer_grouped_parameters,
                optim=optimizer_cls,
                **optimizer_kwargs,
            )
        else:
            trainer_obj.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        # logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        # logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                # logger.info(f"skipped: {skipped/2**20}M params")

    if is_sagemaker_mp_enabled():
        trainer_obj.optimizer = smp.DistributedOptimizer(trainer_obj.optimizer)

    return trainer_obj.optimizer


def get_optimizer(self, model: Any, args: TrainingArguments, 
                  train_dataset: Any, val_dataset: Any,
                  data_train: dict, train_opts: dict,
                  disable_vam_branch: bool = False,
                  epoch_index: int = 0):

    # A second instance of Trainer is instantiated in order to get the optimizer 
    # with the correct model parameters
    # TODO: there is obviously a more efficient way to do that
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=None,
        compute_metrics=self.compute_metrics,
        data_collator=self.collate_fn,
    )

    # Get optimizer
    optimizer = override_create_optimizer(trainer,
                                          disable_vam_branch=disable_vam_branch, 
                                          epoch_index=epoch_index)
    
    num_training_steps = get_number_of_training_steps(data_train, train_opts)
    
    lr_scheduler = trainer.create_scheduler(num_training_steps, optimizer)

    return optimizer, lr_scheduler


def get_number_of_training_steps(data_train, train_opts):
    total_devices = 1 # self.hparams.n_gpus * self.hparams.n_nodes
    train_batches = math.ceil((data_train["count"]["neg_count"]+data_train["count"]["pos_count"]) / train_opts["batch_size"])
    train_batches = train_batches // total_devices
    train_steps = train_opts["epochs"] * train_batches # // self.hparams.accumulate_grad_batches
    return train_steps