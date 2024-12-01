Training on the TVSum dataset is unstable due to its limited size, which makes it challenging to support consistent model training. 

This issue has been noted in the Issues section of the [R2-tuning](https://github.com/yeliudev/R2-Tuning). Therefore, we cannot guarantee that everyone will be able to reliably reproduce experimental results on the TVSum dataset. 

The entire training process is influenced by various environmental factors such as the PyTorch version, CUDA version, and random seeds. If necessary, we recommend readjusting the experimental hyperparameters.