"""configs/common.py
"""
import torch
from detectron2.config import LazyCall as L
# from ufo.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from solver.build import build_lr_optimizer_lazy
from solver.build import build_lr_scheduler_lazy


# optimizer = L(L(maybe_add_gradient_clipping)(optimizer=torch.optim.SGD,
#                                              grad_clip_enabled=True))(
#     params=L(get_default_optimizer_params)(
#         base_lr=0.03,
#         weight_decay=0.0001,
#         weight_decay_norm=0.0001,
#         contiguous=True,
#         bias_lr_factor=2.0,
#         momentum=0.9,
#     ),
# )

optimizer = L(build_lr_optimizer_lazy)(
    optimizer=torch.optim.SGD,
    base_lr=0.03,
    weight_decay=0.0001,
    weight_decay_norm=0.0001,
    contiguous=True,
    bias_lr_factor=2.0,
    momentum=0.9,
    grad_clip_enabled=True,
    grad_clip_norm=5.0, #TODO determine exact value of grad_clip
    lr_multiplier=L(build_lr_scheduler_lazy)(
                    iters_per_epoch=503,
                    max_iters=50000,
                    max_epoch=120,
                    warmup_iters=1000,
                    warmup_factor=0.1,
                    warmup_method='linear',
                    delay_epochs=0,
                    solver_steps=[40, 90],
                    solver_gamma=0.1,
                    eta_min=7e-8,
                    base_lr=5e-1,
                    sched='CosineAnnealingLR',
                )
)



train = dict(
    output_dir="output",
    sacred=dict(enabled=True),
    init_checkpoint="",
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    cudnn_benchmark=True,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        # find_unused_parameters=False,
        find_unused_parameters=True, 
        fp16_compression=False,
    ),
    max_iter=90000,
    checkpointer=dict(period=2000, max_to_keep=1),  # options for PeriodicCheckpointer
    eval_period=5000,
    log_period=20,
    device="gpu",
)
