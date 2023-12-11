We have provide five scripts to run our code under our scripts/ directory. We also config all the best parameters in hydra, 
so only require to run the scripts to train or inference unless you wish to train with different parameters 

BicycleGAN:

Train:
    For training please run the experiment.sh to train the model, the output image will save under the auto create directory
    BCGAN_results/ 
Inference:
    To run the inference, you first need to set the config_name="eval" in the hydra decorator and 
    and then change the pth name you want to run in the variable of best_model_path, then please run the ex_infer.sh.
    This would also calculate lpips score for each image and create directory for fid score. The results would stored 
    under inference_results/ and name based on the experiment name

Soft Intro VAE:

Train:
    For training please run the experiment_vae.sh to train the model, the output image will save under the auto create directory
    VAE_val_results/ 
Inference:
    To run the inference, you first need to set the config_name="eval_vae" in the hydra decorator and 
    and then change the pth name you want to run in the variable of best_model_path, then please run the ex_infer_vae.sh.
    This would also calculate lpips score for each image and create directory for fid score. The results would stored 
    under inference_results/ and name based on the experiment name

FID:
    To run fid.sh you will need to do inference to create corresponding directory, then specify the path of real and fake image
    the inference process create in fid.sh and run it to obtained fid score