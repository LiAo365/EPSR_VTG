_base_ = ["../_base_/models/model.py", "../_base_/datasets/qvhighlights.py", "../_base_/schedules/default.py", "../_base_/runtime.py"]
model = dict(
    loss_cfg=dict(
        loss_reg=dict(type="L1Loss", loss_weight=0.2),  # default 0.2
        loss_c3l=dict(type="C3LLoss", loss_weight=2.5),
    ),
)
