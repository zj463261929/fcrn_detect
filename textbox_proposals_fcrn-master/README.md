# Textbox Proposals Fully Convolutional Regression Network

An implementation of the Fully Convolutional Regression Neural Network (FCRN) Framework in Keras as described by Ankush Gupta et. al in the paper [Synthetic Data for Text Localisation in Natural Images](https://arxiv.org/abs/1604.06646)


Provided in this codebase are python scripts for building a training dataset to train the FCRN and the script to initiate the training of a model which uses the Keras deep learning framework on top of Theano.


## Training Data Format ##

The `build_dataset.py` script will create directories containing H5Py databases that are used as input for the training and validation datasets. Each H5Py database is assumed to have records in a group called "/data" where each record's data is a numpy array containing a 512x512 grayscaled input image with an attribute called 'label' containing a 16x16x7 numpy array representing the 7 output values to regress for each cell of the input image. The 7 dimension feature vector should contain the parameters in the following order:

  (x, y, w, h, sin, cos, c)


## To run the code ##

To build the dataset from SynthText output: `python build_dataset.py /path/to/synth-text-output /path/to/output`

To train the model: `python train_model.py bb-fcrn-model`
