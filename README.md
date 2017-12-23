# Attribute2Sketch2Face
This repository is for our work on Attribute2Sketch2Face.

## Dependencies: 

python 2.7

Pytorch

## Data:
$root$ = attribute2sketch2face

(1) please download the preprocessed [CelebA]and [LFWA] database by link: 

(2) extract them into $root$/dataset/

## Training:
* change the command direction into $root$

* when training on the LFWA dataset. please run: python ./train_lfw.py

* when training on the CelebA dataset. please run: python ./train_celeba.py

## Testing:

* when testingon the LFWA dataset. please run: python ./test_lfw.py

* when testing on the CelebA dataset. please run: python ./test_celeba.py

* All the test result should be in the directory: $root$/result/

## Acknowledgement:

This code is highly inspired by the following:

[cycleGAN and pix2pix:](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

[StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks:](https://github.com/hanzhanggit/StackGAN-Pytorch)

[Generative Adversarial Text-to-Image Synthesis:](https://github.com/reedscot/icml2016)

Also, we highly appreciate the help from [He Zhang](https://github.com/hezhangsprinter) on the Densely Generator "G2" in this code.

