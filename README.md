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
## REFERENCES
 [1] JonathanT.Barron,BenMildenhall,DorVerbin,PratulP.Srinivasan,andPeterHedman.2022. Mip-NeRF360:Unbounded
 Anti-Aliased Neural Radiance Fields. https://doi.org/10.48550/arXiv.2111.12077 arXiv:2111.12077 [cs].
 
 [2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. 2023. Zip-NeRF: Anti-Aliased
 Grid-Based Neural Radiance Fields. https://doi.org/10.48550/arXiv.2304.06706 arXiv:2304.06706 [cs].
 
 [3] AngelaDai,Christian Diller, and Matthias Nießner. 2020. SG-NN: Sparse Generative Neural Networks forSelf-Supervised
 Scene Completion of RGB-D Scans. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE.
 
 [4] John Flynn, Ivan Neulander, James Philbin, and Noah Snavely. 2015. DeepStereo: Learning to Predict New Views from
 the World’s Imagery. https://doi.org/10.48550/arXiv.1506.06825 arXiv:1506.06825 [cs].
 
 [5] Kyle Genova, Xiaoqi Yin, Abhijit Kundu, Caroline Pantofaru, Forrester Cole, Avneesh Sud, Brian Brewington, Brian
 Shucker, and Thomas Funkhouser. 2021. Learning 3D Semantic Segmentation with only 2D Image Supervision.
 https://doi.org/10.48550/arXiv.2110.11325 arXiv:2110.11325 [cs].
 
 [6] Clément Godard, Oisin Mac Aodha, and Gabriel J. Brostow. 2017. Unsupervised Monocular Depth Estimation with
 Left-Right Consistency. In CVPR.
 
 [7] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. 2020.
 NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. https://doi.org/10.48550/arXiv.2003.08934
 arXiv:2003.08934 [cs].
 
 [8] Kutsev Bengisu Ozyoruk, Guliz Irem Gokceler, Taylor L. Bobrow, Gulfize Coskun, Kagan Incetan, Yasin Almalioglu,
 Faisal Mahmood, Eva Curto, Luis Perdigoto, Marina Oliveira, Hasan Sahin, Helder Araujo, Henrique Alexandrino,
 Nicholas J. Durr, Hunter B. Gilbert, and Mehmet Turan. 2021. EndoSLAM dataset and an unsupervised monocular
 visual odometry and depth estimation approach for endoscopic videos. Medical Image Analysis 71 (July 2021), 102058.
 https://doi.org/10.1016/j.media.2021.102058
 
 [9] David Recasens, José Lamarca, José M. Fácil, J. M. M. Montiel, and Javier Civera. 2021. Endo-Depth-and-Motion:
 Reconstruction and Tracking in Endoscopic Videos using Depth Networks and Photometric Constraints. https:
 //doi.org/10.48550/arXiv.2103.16525 arXiv:2103.16525 [cs].
