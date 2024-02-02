# CMC-RSSR

This repository contains the code and models from the paper V. Stojnic, V. Risojevic, ["Self-Supervised Learning of Remote Sensing Scene Representations Using Contrastive Multiview Coding"](https://arxiv.org/abs/2104.07070), In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2021.

This code is buit on top of the official implementation of the [Contrastive Multiview Coding by Y. Tian](https://github.com/HobbitLong/CMC).

To run the code please create Anaconda environment using dependancies defined in *cmc_rssr.yml*.

# Models

Pre-trained CMC models trained in our papaer are available [here](https://www.icloud.com/iclouddrive/05aWsELcprKFwmNnxGti-aaPQ#CMC%5Fmodels.tar).

Trained linear classifiers are available [here](https://www.icloud.com/iclouddrive/040wMl1IAo9pW6qsv9QWnKeuA#linear%5Fmodels).

Finetuned CMC models for our downstream tasks are available [here](https://www.icloud.com/iclouddrive/039f0BRZ3buQdtRR2gbbB02Hw#finetuning%5Fmodels).

We don't provide pre-trained CMC models on the whole ImageNet as they are available on the official CMC implementation.

Pre-trained supervised models are used from torchvision model zoo.

**Given linear classifier and finetuned models can give slightly different results from the ones in the paper, but main conclusions still hold for these models too.**

# CMC training

To run the training of the CMC model use the following python script.

```
python train_CMC.py --data_folder PATH_TO_FOLDER_CONTAINING_THE_IMAGES_OR_LMDB_FOLDER_FOR_MS_IMAGES --image_list PATH_TO_FILE_WITH_IMAGE_LIST --model MODEL_NAME --model_path PATH_TO_DIRECTORY_TO_SAVE_THE_MODEL --batch_size BATCH_SIZE [--multispectral --pca --ben]
```

Use the `--multispectral` flag if you are training on multispectral data and if you want PCA based views use the `--pca` flag.

Use the `--resize_image_aug` flag if you are training on BigEarthNet RGB dataset to resize original images to 256x256 pixels.

Other command line arguments are also possible. Please read the script and parser in *util.py*.

# Feature extraction

To run the feature extraction use the following python script.

```
python extract_features.py --data_folder PATH_TO_FOLDER_CONTAINING_THE_IMAGES_OR_LMDB_FOLDER_FOR_MS_IMAGES --image_list PATH_TO_FILE_WITH_IMAGE_LIST --model MODEL_NAME --resume PATH_TO_TRAINED_CMC_MODEL --features_path PATH_FOR_FEATURE_SAVING [--multilabel_targets PATH_TO_JSON_FILE_WITH_MULTILABEL_TARGETS --multispectral --pca --multispectral_dataset DATASET_NAME]
```

Use ```--multilabel_targets``` if you are extracting features for a multilabel RGB dataset.

Use the `--multispectral` flag if you are extracting features on multispectral data and if you want PCA based views use the `--pca` flag. For multispectral datasets it is necessary to supply ```--multispectral_dataset DATASET_NAME``` dataset name BigEarthNet or So2Sat as these datasets need different preprocessing methods.

# Linear classifier

To run the linear classifier use the following script.

```
python linear_classifier.py --train_data_path PATH_TO_TRAIN_FEATURES --val_data_path PATH_TO_VAL_FEATURES --batch_size BATCH_SIZE --epochs NUM_OF_EPOCHS --learning_rate LEARNING_RATE --lr_decay_epochs DECAY_EPOCHS --lr_decay_rate DECAY_RATE --weight_decay WEIGHT_DECAY --resume PATH_TO_MODEL_SAVE_OR_RESUME [--evaluate]
```

Use ```--evaluate``` flag for evaluation of the trained linear classifier.

# Finetuning

To run the finetuining use the following python script.

```
python finetuning.py --data_folder PATH_TO_FOLDER_CONTAINING_THE_IMAGES_OR_LMDB_FOLDER_FOR_MS_IMAGES --train_image_list PATH_TO_FILE_WITH_TRAIN_IMAGE_LIST --val_image_list PATH_TO_FILE_WITH_VAL_IMAGE_LIST --model MODEL_NAME --model_path PATH_TO_TRAINED_CMC_MODEL --weight_decay WEIGHT_DECAY --epochs NUM_OF_EPOCHS --batch_size BATCH_SIZE --lr_decay_epochs DECAY_EPOCHS --lr_decay_rate DECAY_RATE [--save_path PATH_TO_MODEL_SAVE --learning_rate LEARNING_RATE --resume PATH_TO_MODEL_RESUME --multilabel_targets PATH_TO_JSON_FILE_WITH_MULTILABEL_TARGETS --multispectral --pca --multispectral_dataset DATASET_NAME --evaluate]
```

Use ```--multilabel_targets``` if you are extracting features for a multilabel RGB dataset.

Use the `--multispectral` flag if you are extracting features on multispectral data and if you want PCA based views use the `--pca` flag. For multispectral datasets it is necessary to supply ```--multispectral_dataset DATASET_NAME``` dataset name BigEarthNet or So2Sat as these datasets need different preprocessing methods.

Use ```--evaluate``` flag for evaluation of the trained linear classifier.

# Additional materials

Additional python scripts that can be used to create different dataset formats suitable for this implementation are available in [helper directory](helper).

Dataset splits used in this paper can be found in [data_splits directory](data_splits).

Multilabel targets for every image in BigEarthNet and MLRSNet datasets used in this paper can be found in JSON files in [image_target_mapping directory](image_target_mapping).

# Citation
```
@InProceedings{Stojnic_2021_CVPR_Workshops,
    author = {Stojnic, Vladan and Risojevic, Vladimir},
    title = {Self-Supervised Learning of Remote Sensing Scene Representations Using Contrastive Multiview Coding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month = {June},
    year = {2021}
}
```
