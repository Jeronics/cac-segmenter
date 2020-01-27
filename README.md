cac-segmenter
=============
Image Segmentation using Cage Active Contours(CAC). This project provides a complementary open-source code for the published results in the paper [Cage Active Contours for image warping
and morphing](https://link.springer.com/content/pdf/10.1186%2Fs13640-018-0248-z.pdf) published in EURASIP journal on Image and Video Processing.



Usage
=====

0. Requirements
  Images and ground truth images must be of type .png
1. Creating the initial Contour

  The script mask_from_images creates the initial contours given an initial image. To do this it requires:
  
    a) A .txt file with the path of the image or images to segment
    
    b) A .txt file with the path of the ground truth images so that their file names match the images' file names.
    
    c) A .txt file where the initial information will be written to match the input format required for the segmentation.
    
  Once this is done, and mask_from_images is run, images will appear one by one. The user is required to click on the image twice:
  
    a) First, to mark the center of your initial circular contour
    
    b) Second, to mark a point of the radius of your initial circular contour
    
  Now a the input file is obtained to be able to apply the segmentation
  
2. Running the Segmentation Procedure

    a) Open one of the following classes with the desired energy:
    
      i) MeanCAC
      
      ii) OriginalGaussianCAC
      
      iii) GaussianCAC
      
      iv) MixtureGaussianCAC
      
      v) MultivariateGaussianCAC
      
      vi) MultiMixtureGaussianCAC
      
      vii) HueMeanCAC
      
    b) Change the input file with the one generated previously
    
    c) Run

Requirements
====

1. OpenCV

2. Python 2.7

In the main_directory/apicac/ folder, do the following:

  clean make
  
  make
