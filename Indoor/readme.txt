This is the testing code from Rutgers Unviersity for the NTIRE-2018 Image Dehazing Chanllenge;
This is for Track(Indoor)

Authors: He Zhang (he.zhang92@rutgers.edu); Vishwanath Sindagi (vishwanath.sindagi@rutgers.edu); Vishal M. Patel (vishal.m.patel@rutgers.edu);

Prerequisites:
1. Linux
2. Python 2 or 3
3. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)
4. MATLAB(any version)

Installation:
1. Install PyTorch and dependencies from http://pytorch.org (Ubuntu+Python2.7)
   (conda install pytorch torchvision -c pytorch)

2. Install Torch vision from the source.
   (git clone https://github.com/pytorch/vision
   cd vision
   python setup.py install)

3. Install python package: 
   numpy, scipy, PIL, pdb


Testing:

To run the testing code completely, you need to follow these steps:
1. Run in the terminal: sh run_dehaze.sh
2. cd ./indoor
3. Open matlab and run demo.m (Note: this step takes a couple of minutes to run)


The dehazed results are located in ./indoor/our_cvprw_submitted
