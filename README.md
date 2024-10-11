# 3D-scene-Reconstruction
The objective of the project is to create and develop a robust system that can effectively generate
 the 3-D point clouds so that we can reconstruct the 3D scene from a given group of 2D endoscopic
 images. This helps in enhanced spatial understanding, improved diagnostic accuracy, and increased
 quality of patient health care.
## BACKGROUND
 This project uses ResNet18md based encoder and decoders, upsampling them. We also use a
 combination of loss functions like SSIM, L1-consistency, disparity smoothness, etc. It is beneficial
 to understand these concepts so that reading the documentation further will be easier.
## Approach
 In the field of depth estimation and point cloud generation, there are a lot of existing learning
 based methods that show promising results, but the only limitation is that these methods treat the
 problem as a supervised regression problem requiring a lot of depth map(ground truth) data for 3D Scene reconstruction from 2D images
 training. Since it is always not easy to gather high-quality depth map(ground truth) data under a
 variety of conditions, in this project we would like to extend the existing approaches by using the
 more obtainable stereo images instead of the depth data.
 The model that is constructed in this project can learn using a sequence of images to generate
 the depth map without the use of depth map data. The model is trained on image reconstruction
 loss followed by the use of epi-polar geometry to create disparity maps. And then, the generated
 disparity maps are used to enhance the accuracy of the model along with the consistency between
 the generated disparities and the left and right images to improve the performance.
