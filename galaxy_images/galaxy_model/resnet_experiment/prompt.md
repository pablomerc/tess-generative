Lets implement a new experiment in this folder /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/resnet_experiment

I want to take the same images and sme preprocessing from the neighbors dataset as is being used in /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/neighbours_train.py when it doesnt use the procomputed batches

But this time I want a different model.

It will just take target images that are HSC. Then it will use a ResNet 18 encoder to extract features. And it will directly predict the elipticity


Wait no instead of from the neighbors dataset it should tkae it from the parquet files in /data/vision/billf/scratch/pablomer/legacysurvey_hsc/data

You need to check beforehand that this file has the fluxes and it has the elipticities

import h5py

with h5py.File(path, 'r') as f:
    print(f.keys())

    <KeysViewHDF5 ['EBV', 'FLUX_G', 'FLUX_I', 'FLUX_R', 'FLUX_W1', 'FLUX_W2', 'FLUX_W3', 'FLUX_W4', 'FLUX_Z', 'SHAPE_E1', 'SHAPE_E2', 'SHAPE_R', 'a_g', 'a_i', 'a_r', 'a_y', 'a_z', 'catalog', 'g_cmodel_mag', 'g_cmodel_magerr', 'g_extendedness_value', 'g_sdssshape_psf_shape11', 'g_sdssshape_psf_shape12', 'g_sdssshape_psf_shape22', 'g_sdssshape_shape11', 'g_sdssshape_shape12', 'g_sdssshape_shape22', 'hsc_image', 'hsc_object_id', 'i_cmodel_mag', 'i_cmodel_magerr', 'i_extendedness_value', 'i_sdssshape_psf_shape11', 'i_sdssshape_psf_shape12', 'i_sdssshape_psf_shape22', 'i_sdssshape_shape11', 'i_sdssshape_shape12', 'i_sdssshape_shape22', 'indices', 'legacysurvey_image', 'legacysurvey_object_id', 'r_cmodel_mag', 'r_cmodel_magerr', 'r_extendedness_value', 'r_sdssshape_psf_shape11', 'r_sdssshape_psf_shape12', 'r_sdssshape_psf_shape22', 'r_sdssshape_shape11', 'r_sdssshape_shape12', 'r_sdssshape_shape22', 'y_cmodel_mag', 'y_cmodel_magerr', 'y_extendedness_value', 'y_sdssshape_psf_shape11', 'y_sdssshape_psf_shape12', 'y_sdssshape_psf_shape22', 'y_sdssshape_shape11', 'y_sdssshape_shape12', 'y_sdssshape_shape22', 'z_cmodel_mag', 'z_cmodel_magerr', 'z_extendedness_value', 'z_sdssshape_psf_shape11', 'z_sdssshape_psf_shape12', 'z_sdssshape_psf_shape22', 'z_sdssshape_shape11', 'z_sdssshape_shape12', 'z_sdssshape_shape22']>

    It has to predict SHAPE_E1 and SHAPE_E2

    The loss will be just on predicting those things okay?

You have to do a training and test split

Then save this trained model, and make some plots to show the performance predicting that stuff on the test set

On a second stage I want to take generated images, and see how the model trained on real data performs when evaluated on generated images.

To do this we probably need to crossmatch some example between the neighbors dataset and the MMU dataset, look at the /data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation folder to figure out how we did that in the past please because there was some very important considerations to be taken about the indices
