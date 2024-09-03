auto_scale_lr = dict(base_batch_size=512)
backend_args = dict(backend='local')
batch_size = 16
codec = dict(
    heatmap_size=(
        64,
        64,
    ),
    input_size=(
        256,
        256,
    ),
    sigma=2,
    type='MSRAHeatmap')
custom_hooks = [
    dict(type='SyncBuffersHook'),
]
data_mode = 'topdown'
data_root = '/workspace/mmpose-synthetic-tune/datasets/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    badcase=dict(
        badcase_thr=5,
        enable=False,
        metric_type='loss',
        out_dir='badcase',
        type='BadCaseAnalysisHook'),
    checkpoint=dict(
        interval=10,
        rule='greater',
        save_best='coco/AP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='PoseVisualizationHook'))
default_scope = 'mmpose'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    by_epoch=True, num_digits=6, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        extra=dict(
            stage1=dict(
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_branches=1,
                num_channels=(64, ),
                num_modules=1),
            stage2=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_branches=2,
                num_channels=(
                    48,
                    96,
                ),
                num_modules=1),
            stage3=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_branches=3,
                num_channels=(
                    48,
                    96,
                    192,
                ),
                num_modules=4),
            stage4=dict(
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_branches=4,
                num_channels=(
                    48,
                    96,
                    192,
                    384,
                ),
                num_modules=3)),
        in_channels=3,
        init_cfg=dict(
            checkpoint='checkpoints/hrnet_w48-8ef0771d.pth',
            type='Pretrained'),
        type='HRNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='PoseDataPreprocessor'),
    head=dict(
        decoder=dict(
            heatmap_size=(
                64,
                64,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        deconv_out_channels=None,
        in_channels=48,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        out_channels=20,
        type='HeatmapHead'),
    init_cfg=dict(
        checkpoint=
        'models/_train-2024-08-15_05-08-10-synthetic-0.809/best_coco_AP_epoch_50.pth',
        type='Pretrained'),
    test_cfg=dict(flip_mode='heatmap', flip_test=True, shift_heatmap=True),
    type='TopdownPoseEstimator')
optim_wrapper = dict(optimizer=dict(lr=0.0005, type='Adam'))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=210,
        gamma=0.1,
        milestones=[
            170,
            200,
        ],
        type='MultiStepLR'),
]
resume = True
sample_name = 'sample98'
test_cfg = dict()
test_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='annotations/20kp-sample98-test.json',
        data_mode='topdown',
        data_prefix=dict(img='20kp-sample98-data/'),
        data_root='/workspace/mmpose-synthetic-tune/datasets/',
        metainfo=dict(from_file='custom-configs/20kp.py'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/workspace/mmpose-synthetic-tune/datasets/annotations/20kp-sample98-test.json',
    type='CocoMetric')
train_cfg = dict(by_epoch=True, max_epochs=210, val_interval=10)
train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        ann_file='annotations/20kp-sample98-train.json',
        data_mode='topdown',
        data_prefix=dict(img='20kp-sample98-data/'),
        data_root='/workspace/mmpose-synthetic-tune/datasets/',
        metainfo=dict(from_file='custom-configs/20kp.py'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(direction='horizontal', type='RandomFlip'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(
                encoder=dict(
                    heatmap_size=(
                        64,
                        64,
                    ),
                    input_size=(
                        256,
                        256,
                    ),
                    sigma=2,
                    type='MSRAHeatmap'),
                type='GenerateTarget'),
            dict(type='PackPoseInputs'),
        ],
        type='CocoDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(direction='horizontal', type='RandomFlip'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(
        encoder=dict(
            heatmap_size=(
                64,
                64,
            ),
            input_size=(
                256,
                256,
            ),
            sigma=2,
            type='MSRAHeatmap'),
        type='GenerateTarget'),
    dict(type='PackPoseInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='annotations/20kp-sample98-val.json',
        data_mode='topdown',
        data_prefix=dict(img='20kp-sample98-data/'),
        data_root='/workspace/mmpose-synthetic-tune/datasets/',
        metainfo=dict(from_file='custom-configs/20kp.py'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(input_size=(
                256,
                256,
            ), type='TopdownAffine'),
            dict(type='PackPoseInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/workspace/mmpose-synthetic-tune/datasets/annotations/20kp-sample98-val.json',
    type='CocoMetric')
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(input_size=(
        256,
        256,
    ), type='TopdownAffine'),
    dict(type='PackPoseInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'models/_train-2024-08-15_05-43-51-real-on-synthetic'
