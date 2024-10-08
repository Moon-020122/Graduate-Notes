_base_ = [
    '../_base_/models/setr_hlg_share_naive_pup.py',
    '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    backbone=dict(
        type='HLGTransformer',
        pretrained='pretrain/pvt_small.pth',
        
        img_size=512,
        in_chans=3,
        patch_size=4,
        num_classes=150,
        
        embed_dims=[96, 192, 384, 768],
        num_heads=[3, 6, 12, 24],
        depths=[2, 2, 14, 2],
        
        use_checkpoint=False,
        h0_att=False,
        proj_dwconv='convbn',
        downsampling=['c', 'sc', 'sc', 'sc'],
        h0_h1_method='mean',
        crs_interval=[8, 4, 2, 1],
        dynamic_position_bias=True,
    ),
    decode_head=dict(
        type='HLGUPHead',
        depth=2,
        sr_ratio=8,
        in_channels=[96, 192, 384, 768],
        num_classes=150,
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150,
    ))
test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(341, 341))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.0,
                 paramwise_cfg=dict(
                     custom_keys={
                         'norm': dict(decay_mult=0.),
                         'head': dict(lr_mult=10.),
                     })
                 )

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# crop_size = (768, 768)
find_unused_parameters = True
data = dict(samples_per_gpu=2)