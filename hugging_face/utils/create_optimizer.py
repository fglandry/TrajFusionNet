import math
from torch import nn
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available
from transformers.utils import is_sagemaker_mp_enabled
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import Trainer
from transformers.trainer_utils import ShardedDDPOption
from transformers.trainer_pt_utils import get_parameter_names


if is_fairscale_available():
    dep_version_check("fairscale")
    import fairscale
    from fairscale.optim import OSS

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

def override_create_optimizer(trainer_obj, alternate_training=False, epoch_index=0):
    """
    Override the create_optimizer() method in the Huggingface
    library's Trainer class
    """
    opt_model = trainer_obj.model_wrapped if is_sagemaker_mp_enabled() else trainer_obj.model

    if trainer_obj.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        if alternate_training:
            if epoch_index % 2 == 0:
                params_to_lower_lr = [n for idx, (n, p) in enumerate(opt_model.named_parameters()) if \
                                    (p.requires_grad and n.startswith("traj_class_TF"))] # n.startswith("vgg_encoder")
            else:
                params_to_lower_lr = [n for idx, (n, p) in enumerate(opt_model.named_parameters()) if \
                                    (p.requires_grad and not n.startswith("traj_class_TF"))] # n.startswith("vgg_encoder")
        else:
            """
            # ViT 
            params_to_lower_lr = [n for idx, (n, p) in enumerate(opt_model.named_parameters()) if \
                                (p.requires_grad and idx < 84)]
            params_to_lower_lr = [n for idx, (n, p) in enumerate(opt_model.named_parameters()) if \
                                    (p.requires_grad and n.startswith("van"))] # n.startswith("vgg_encoder")
            """
            params_to_lower_lr = []
            params_to_lower_lr = [n for idx, (n, p) in enumerate(opt_model.named_parameters()) if \
                                  (p.requires_grad and n.startswith("transformer.traj_TF"))]
            

        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad and not n in params_to_lower_lr)
                ],
                "weight_decay": trainer_obj.args.weight_decay,
                # 'lr': 5.0e-06
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad and not n in params_to_lower_lr)
                ],
                "weight_decay": 0.0,
                # 'lr': 5.0e-06
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (p.requires_grad and n in params_to_lower_lr)
                ],
                'lr': 5.0e-07
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


def get_optimizer(self, model, args, train_dataset, val_dataset,
                    data_train, train_opts, warmup_ratio, 
                    alternate_training=False, epoch_index=0):

    # A second instance of Trainer is declared in order to get the optimizer 
    # with the correct model parameters
    # ToDo: there is obviously a more efficient way to do that
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=None,
        compute_metrics=self.compute_metrics,
        data_collator=self.collate_fn,
        #optimizers=(optimizer, lr_scheduler)
    )

    # Get optimizer
    # optimizer = trainer.create_optimizer()
    optimizer = override_create_optimizer(trainer, 
                                          alternate_training=alternate_training, 
                                          epoch_index=epoch_index)
    """
    optimizer = AdamW(
        model.parameters(),
        lr=train_opts["lr"],
        no_deprecation_warning=True
    )
    """

    """
    optim.SGD([
            {'params': model.base.parameters()},
            {'params': model.classifier.parameters(), 'lr': 1e-3}
        ], lr=1e-2, momentum=0.9)
    """
    
    num_training_steps = get_number_of_training_steps(data_train, train_opts)
    
    lr_scheduler = trainer.create_scheduler(num_training_steps, optimizer)
    
    # default is get_linear_schedule_with_warmup() 
    #lr_scheduler = get_linear_schedule_with_warmup(optimizer,
    #    num_warmup_steps=math.ceil(num_training_steps * warmup_ratio), 
    #    num_training_steps=num_training_steps)
    #lr_scheduler = get_constant_schedule(optimizer)

    return optimizer, lr_scheduler

def get_number_of_training_steps(data_train, train_opts):
    total_devices = 1 # self.hparams.n_gpus * self.hparams.n_nodes
    train_batches = math.ceil((data_train["count"]["neg_count"]+data_train["count"]["pos_count"]) / train_opts["batch_size"])
    train_batches = train_batches // total_devices
    train_steps = train_opts["epochs"] * train_batches # // self.hparams.accumulate_grad_batches
    return train_steps