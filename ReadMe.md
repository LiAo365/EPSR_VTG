# Efficient Pre-trained Semantics Refinement for Video Temporal Grounding


## Dataset Preparation

We using the pre-extracted features coming from this **awesome** paper [R$^2$-Tuning](https://github.com/yeliudev/R2-Tuning), can be downloaded from [HuggingFace Hub](https://huggingface.co/yeliudev/R2-Tuning) directly. And We express our sincere gratitude for their contribution to the community.

**Please follow our baseline to prepare the dataset and place the corresponding files in the correct directory. And change the config file to the correct path.**

Here are the origin video datasets download links:

- [QVHighlights](https://nlp.cs.unc.edu/data/jielei/qvh/qvhilights_videos.tar.gz)
- [Ego4D-NLQ](https://ego4d-data.org/)
- [Charades-STA](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1.zip)
- [TACoS](https://datasets.d2.mpi-inf.mpg.de/MPII-Cooking-2/MPII-Cooking-2-videos.tar.gz)
- [YouTube Highlights](https://github.com/aliensunmin/DomainSpecificHighlight)
- [TVSum](https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz)

## Training

```bash
# Single GPU
python tools/launch.py <path-to-config>

# Multiple GPUs on a single node (elastic)
torchrun --nproc_per_node=<num-gpus> tools/launch.py <path-to-config>
```

<details>
<summary><i>Arguments of <code>tools/launch.py</code></i></summary>
<br>

- `config` The config file to use
- `--checkpoint` The checkpoint file to load from
- `--resume` The checkpoint file to resume from
- `--work_dir` Working directory
- `--eval` Evaluation only
- `--dump` Dump inference outputs
- `--seed` The random seed to use
- `--amp` Whether to use automatic mixed precision training
- `--debug` Debug mode (detect `nan` during training)
- `--launcher` The job launcher to use

</details>

## Evaluation

```bash
python tools/launch.py <path-to-config> --checkpoint <path-to-checkpoint> --eval
```

## Notes

If problems occur when reproducing the results, please feel free to contact us at github or email.

Maybe you need to change the `config` file to the correct path.

Some issues may be fixed by these issues in [Baseline Repository](https://github.com/yeliudev/R2-Tuning)


## Acknowledgement

We would like to express our sincere gratitude to the following authors for their contributions to the community:

- [R^2-Tuning](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05800.pdf)