# TrajFusionNet

This repo contains the code for the paper **TrajFusionNet: Pedestrian Crossing Intention Prediction via Fusion of Sequential and Visual Trajectory Representations**

TrajFusionNet is a transformer-based model that integrates future pedestrian trajectory and vehicle speed predictions as priors for predicting crossing intention.

TrajFusionNet is composed of two branches: a Sequence Attention Module (SAM) and a Visual Attention Module (VAM). The SAM branch learns from a sequential representation of the observed and predicted pedestrian trajectory and vehicle speed. Complementarily, the VAM branch enables learning from a visual representation of the predicted pedestrian trajectory by overlaying predicted pedestrian bounding boxes onto scene images.

## Set up

Start by creating a conda environment:

```bash
conda create -n trajfusionnet-env python=3.10 pytorch torchvision pytorchvideo pytorch-cuda accelerate tensorflow -c pytorch -c nvidia -c conda-forge
```

A pytorch-cuda version might need to be specified depending on your NVIDIA driver version.

Then, install the remaining libraries with pip:

```bash
pip install -r requirements.txt
```



