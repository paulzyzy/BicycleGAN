# Image to Image Translation

### Introduction
This is a pytorch implementation of models based on [BicycleGAN](https://arxiv.org/abs/1711.11586) and [Soft-Intro VAE](https://arxiv.org/pdf/2012.13253.pdf) on Image to Image Translation. In this project, we explore the tradeoff between  fidelity and diversity for image translation tasks. 

### Contribution
- We inherit the model architecture from BicycleGAN for the Soft-Intro VAE.
- We explore different hyperparameters for Soft-Intro VAE and obtain the best result of 73 in FID score.
- We explore different loss function for two different architecture configurations.

### Dataset

All of our models are trained on the [Edge2Shoes](https://www.kaggle.com/datasets/balraj98/edges2shoes-dataset) dataset which consists of 50,025 shoe images and their corresponding edges split into train and test subsets.

### Requirement
```
$ pip install requirements.txt
``` 
### Train

#### BicycleGAN

For training please run the following command to train the model, the output image will save under the auto create directory `BCGAN_results/`.
```
$ bash scripts/experiment.sh
``` 

#### Soft-Intro VAE

For training please run the experiment_vae.sh to train the model, the output image will save under the auto create directory `VAE_val_results/`.

```
$ bash scripts/experiment_vae.sh
``` 
### Inference

#### BicycleGAN

To run the inference, you first need to set the `config_name="eval"` in the hydra decorator and then change the pth name you want to run in the variable of best_model_path, then please run the following command:

```
$ bash scripts/ex_infer.sh
``` 
This would also calculate lpips score for each image and create directory for fid score. The results would stored under inference_results/ and name based on the experiment name.

#### Soft-Intro VAE

To run the inference, you first need to set the `config_name="eval_vae"` in the hydra decorator and then change the pth name you want to run in the variable of best_model_path, then please run the following commands.

```
$ bash scripts/ex_infer_vae.sh
``` 
This would also calculate lpips score for each image and create directory for fid score. The results would stored under `inference_results/` and name based on the experiment name

### FID

#TODO

### Results 

#### BicycleGAN
<img src="https://github.com/paulzyzy/BicycleGAN/blob/main/results/best_bcgan.png" alt="BCGAN" height="410"/>

#TODO
### References

[BicycleGAN](https://github.com/junyanz/BicycleGAN)</br>
#TODO
