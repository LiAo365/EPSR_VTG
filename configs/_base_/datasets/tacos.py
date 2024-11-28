_base_ = "datasets"
# dataset settings
data_type = "Grounding"
data_root = "/public/home/zyq_202324131079/data/tacos/"
data = dict(
    train=dict(
        type=data_type,
        label_path=data_root + "train.jsonl",
        video_path=data_root + "frames_224_0.5fps",
        cache_path=data_root + "clip_b32_vid_k4",
        query_path=data_root + "clip_b32_txt_k4",
        use_cache=True,
        min_video_len=5,
        fps=0.5,
        loader=dict(batch_size=16, num_workers=4, pin_memory=True, shuffle=True),
    ),
    val=dict(
        type=data_type,
        label_path=data_root + "test.jsonl",
        video_path=data_root + "frames_224_0.5fps",
        cache_path=data_root + "clip_b32_vid_k4",
        query_path=data_root + "clip_b32_txt_k4",
        use_cache=True,
        fps=0.5,
        loader=dict(batch_size=1, num_workers=2, pin_memory=True, shuffle=False),
    ),
)
