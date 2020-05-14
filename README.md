# Guided Dual Networks For Single Image Super-Resolution
the github repository of GDSR<br/>
This repository is for GDSR, and our presentation slide can be download [here](https://pan.baidu.com/s/1xT9nc6kcLpD9J7ITnDhdGA)   (access code:ipf7)
the results of GDSR can be download [here](https://pan.baidu.com/s/1L_M2Qryd-mfr7N2z3X5PPg) (access code:l2y0)

The code is built on [BasicSR (PyTorch)](https://github.com/xinntao/BasicSR) and tested on Ubuntu 18.04/16.04 environment (Python3.6, PyTorch_1.0.1, CUDA9.0, cuDNN7.4) with Nividia RTX 2080Ti/GTX 1080Ti GPUs.
# Contents
[Introduction](#Introduction)
[codes](#codes)
[results](#results)

# Introduction
The PSNR-oriented super-resolution (SR) methods pursue high reconstruction accuracy, but tend to produce over-smoothed results and lose plenty of high-frequency details. The GAN-based SR methods aim to generate more photo-realistic images, but the hallucinated details are often accompanied with unsatisfying artifacts and noises. To address these problems, we propose a guided dual super-resolution network \relax(GDSR), which exploits the advantages of both the PSNR-oriented and the GAN-based methods to achieve a good trade-off between reconstruction accuracy and perceptual quality. Specifically, our network contains two branches to simultaneously generate SR images with high accuracy and satisfactory visual quality, where one branch is trained to extract global information and the other to focus on detail information. To obtain more high-frequency features, we use the global features extracted from the low-frequency branch to guide the training the high-frequency branch. Besides, our method utilizes a mask generator to adaptively recover the final super-resolved image. Extensive experiments on several standard benchmarks show that our proposed method achieves comparative performance with state-of-the-art methods.

![Network Architecture](https://github.com/wenchen4321/GDSR/tree/master/imgs/network1.png)
# codes

# results
## Quantitative Comparisons
![RMSE_PI figure](https://github.com/wenchen4321/GDSR/tree/master/imgs/RMSE_PI.png)

## Qualitative Results
![visual result1](/imgs/visual_result1.png)  
![visual result2](https://github.com/wenchen4321/GDSR/tree/master/imgs/visual_result2.png)  
![visual result3](https://github.com/wenchen4321/GDSR/tree/master/imgs/visual_result3.png)  

## Acknowledgement
Our work and implementations are inspired by following project:<br/>
[BasicSR] (https://github.com/xinntao/BasicSR)<br/>
