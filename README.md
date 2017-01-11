# Patch based deep learning models 
Train DeepSat dataset [Basu et al., 2015] with Torch

Files SAT4.lua and SAT6.lua can be used to train SAT-4 and SAT-6 airborne datasets [Saikat Basu, Sangram Ganguly, Supratik Mukhopadhyay, Robert Dibiano, Manohar Karki and Ramakrishna Nemani, DeepSat - A Learning framework for Satellite Imagery, ACM SIGSPATIAL 2015]. You can donwload the dataset from the following link: http://csc.lsu.edu/~saikat/deepsat/

To use SAT4.lua and SAT6.lua files, do the following steps:

1. Install Torch in your pc  [ http://torch.ch/docs/getting-started.html#_ ]

2. After installing Torch, make sure that you have the following external Torch packages:

   'nn' 'cutorch' 'cunn' 'xlua' 'optim' 'image' 'trepl' 'mattorch'

    To install them, just type: luarocks install package_name

    It should be noted that in order to install 'mattorch' package  you need to type the following command: MATLAB_ROOT=/path_where_matlab_is_installed luarocks install mattorch

 
3. Create a folder named DeepSat (name is optional) and transfer the following files in it:
    -SAT4.lua
    -SAT6.lua
    -folder 'models'
    -sat-4-full.mat (the file that you downloaded)
    -sat-6-full.mat (the file that you downloaded)

4. Open terminal in folder DeepSat and type

   th SAT4.lua 

   or 

   th SAT6.lua 

   depending on the dataset that you want to use.

File SAT4.lua is used to train and test SAT-4 dataset. File SAT6.lua is used to train and test SAT-6 dataset.You can use ConvNet, AlexNet or VGG model for training. You just have to change the opt options (line 16) of the file that you use (SAT4.lua or SAT6.lua).

After accomplishing the above steps, the training and testing is ready to start. When the training procedure is over, file 'class_predictions_sat4.t7' or 'class_predictions_sat6.t7' will have been created, depending on the dataset you use. It is a one dimensional Tensor that includes the class predictions of the trained model.

The files of this repository were used to produce the accuracy results of the paper 'M. Papadomanolaki, M. Vakalopoulou, S. Zagoruyko, K. Karantzalos, 2016. Benchmarikng Deep Learning Frameworks for the Classification of Very High Resolution Satellite Multispectral Data, ISPRS 2016 – XXIII ISPRS Congress, Prague, Chech republic' 

If you find this code useful in your research, please consider citing:

M. Papadomanolaki, M. Vakalopoulou, S. Zagoruyko, K. Karantzalos, 2016. Benchmarikng Deep Learning Frameworks for the Classification of Very High Resolution Satellite Multispectral Data


