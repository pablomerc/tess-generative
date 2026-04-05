I want you to make an experiment in this folder /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/cross_predict_experiment

Basically, make a simple model that has a ResNet18 architecture to extract features similar to the encoders in /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/double_train_fm_neighbors.py

But this time we will just cross-predict properties directly.

So use the neighbors dataset /data/vision/billf/scratch/pablomer/data/neighbours_v2.h5

And the most appropriate dataloader

We might even just want to make a new custom dataloader since we can just read the file sequentially in this case

To train a model where

From each example take the HSC image (this might have been the target or the galaxy pair)
then predict the Legacy survey instrument properties

Explore /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation to see what I called Legacy instrument properties

And derive your loss directly from that

At the end of it report R2 squares and average over channels like we did in /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/final

I want to see the R2 square that i get here and put it side by side with what we got for the model in /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/final/predict_all_zdim16_nogeom_neighbors_table.csv
/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/final/predict_all_zdim16_nogeom_neighbors.csv


For the Resnet architecture use the same package and all as i used in my model. You might only want to change the layers at the very end of the resnet

Make two slurm job scripts. One with a very quick test (few epochs) for me to test that the thing runs end to end
And then one where it runs it for a reasonable amount of time

I want to get loss curves logged onto wandb

And at the end of the training I want to receive a discord message with the plot of the R2squareds obtained by this and by my model


Also what i described above is cross predicting from HSC images to legacy properties. I also want to do another experiment that is the opposite: from legacy image predict HSC properties

To see how to do the discord thing you can look at /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/xi_squared_experiment/test_discord_notify.slurm

Prompt me if you need help to set up the wandb
