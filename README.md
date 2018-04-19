## Multi-scale Single Image Dehazing using Perceptual Pyramid Deep Network (NTIRE-2018-Dehazing-Challenge)
[He Zhang](https://sites.google.com/site/hezhangsprinter), [Vishwanath Sindagi](http://www.vishwanathsindagi.com/), [Vishal M. Patel](http://www.rci.rutgers.edu/~vmp93/)

[[Paper Link](https://arxiv.org/abs/1803.08396)] (CVPRw'18)

We propose a new end-to-end single image dehazing method, called Densely Connected Pyramid Dehazing Network (DCPDN), which can jointly learn the transmission map, atmospheric light and dehazing all together. The end-to-end learning is achieved by directly embedding the atmospheric scattering model into the network, thereby ensuring that the proposed method strictly follows the physics-driven scattering model for dehazing. Inspired by the dense network that can maximize the information flow along features from different levels, we propose a new edge-preserving densely connected encoder-decoder structure with multi-level pyramid pooling module for estimating the transmission map. This network is optimized using a newly introduced edge-preserving loss function. To further incorporate the mutual structural information between the estimated transmission map and the dehazed result, we propose a joint-discriminator based on generative adversarial network framework to decide whether the
corresponding dehazed image and the estimated transmission map are real or fake. An ablation study is conducted to demonstrate the effectiveness of each module evaluated at both estimated transmission map and dehazed result. Extensive experiments demonstrate that the proposed method achieves significant improvements over the state-of-the-art methods.

	@inproceedings{dehaze_zhang_2018w,		
	  title={Densely Connected Pyramid Dehazing Network},
	  author={Zhang, He and Patel, Vishal M},
	  booktitle={CVPR},
	  year={2018}
	} 

And this method rank 1st in the indoor track and 3rd in the outdoor track in [NTIRE-2018 Dehazing Challenge](http://www.vision.ee.ethz.ch/en/ntire18/) (CVPRw'18). 


All the codes can be also be found in the following link
https://drive.google.com/drive/folders/1BjJDM7qCdgqSR356PHcSUi_6fWclrGad?usp=sharing
