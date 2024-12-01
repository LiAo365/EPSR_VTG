_base_ = ["../_base_/models/model.py", "../_base_/datasets/youtube.py", "../_base_/schedules/default.py", "../_base_/runtime.py"]
# model settings
model = dict(
    strides=(1,),
    adapter_cfg=dict(dropout=0.5, use_tef=False),  # dropout=0.5,
    merge_cls_sal=False,
    coord_head_cfg=None,
    loss_cfg=dict(
        loss_cls=dict(type="DynamicBCELoss"),
        loss_reg=None,
        loss_sal=dict(direction="row"),
        loss_motion=dict(type="Seq2SeqCLLoss", loss_weight=0.1),
    ),
)
# dataset settings
data = dict(train=dict(domain="dog"), val=dict(domain="dog"))
# runtime settings
stages = dict(
    epochs=200,
    lr_schedule=None,  # dict(type="epoch", policy="step", step=[20]),
    optimizer=dict(type="AdamW", lr=5e-4, weight_decay=1e-4),
    warmup=dict(steps=20),
)
