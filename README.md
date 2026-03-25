# TrajFusionNet

This repository contains the code for the paper **TrajFusionNet: Pedestrian Crossing Intention Prediction via Fusion of Sequential and Visual Trajectory Representations**

TrajFusionNet is a transformer-based model for predicting pedestrian crossing intention. The approach is based on fusing sequential and visual representations of pedestrian trajectories. The model first predicts future trajectories and then uses both observed and predicted trajectories to condition the intention prediction.

<img src="docs/architecture.png" alt="TrajFusionNet Architecture" width="500">

TrajFusionNet is composed of two branches: a Sequence Attention Module (SAM) and a Visual Attention Module (VAM). The SAM branch learns from a sequential representation of the observed and predicted pedestrian trajectory and vehicle speed. Complementarily, the VAM branch enables learning from a visual representation of the pedestrian trajectory in relation to the scene context by overlaying observed and predicted pedestrian bounding boxes onto scene images.

## Set up

Start by creating a conda environment:

```bash
conda create -n trajfusionnet-env python=3.10 pytorch torchvision pytorchvideo pytorch-cuda accelerate tensorflow -c pytorch -c nvidia -c conda-forge
```

The pytorch-cuda version might need to be specified depending on your NVIDIA driver version.

Then, install the remaining libraries with pip:

```bash
pip install -r requirements.txt
```

### Downloading the datasets and model weights

The PIE and JAAD datasets need to be downloaded and processed by following instructions provided in the following GitHub repos: [https://github.com/aras62/PIE](https://github.com/aras62/PIE) and [https://github.com/ykotseruba/JAAD](https://github.com/ykotseruba/JAAD)

After downloading the datasets, export environment variables pointing to the datasets' locations:
```bash
export JAAD_PATH=<jaad_dataset_location>
export PIE_PATH=<pie_dataset_location>
```

The model weights are downloaded automatically from Hugging Face when running the code ([https://huggingface.co/efl7126/trajfusionnet](https://huggingface.co/efl7126/trajfusionnet)). Alternatively, the model weights can be downloaded from Google Drive: [https://drive.google.com/drive/folders/1mXQL5W0LoYa5vZOI_SFIkFkSuyaL0ux8](https://drive.google.com/drive/folders/1mXQL5W0LoYa5vZOI_SFIkFkSuyaL0ux8). You will have to extract the zip file to the data/ directory by running:
```bash
unzip <download_location>/weights.zip -d data/
```

## Inference

To perform model inference, execute the following command:
```bash
python3 train_test.py -c config_files/TrajFusionNet.yaml --test_only
```

The dataset to use and other config parameters can be modified in `config_files/TrajFusionNet.yaml`

## Training

To train the model end-to-end, run:
```bash
python3 train_test.py -c config_files/TrajFusionNet.yaml --train_end_to_end
```

## Citation

If you find this repo useful, please cite the following publication:

```bibtex
@article{landry2025trajfusionnet,
  title={TrajFusionNet: Pedestrian Crossing Intention Prediction via Fusion of Sequential and Visual Trajectory Representations},
  author={Landry, Francois G and Akhloufi, Moulay A},
  journal={arXiv preprint arXiv:2508.19866},
  year={2025}
}
```

Depending on your use of the code, please also cite the following:

* A large part of the codebase was forked from [https://github.com/ykotseruba/PedestrianActionBenchmark](https://github.com/ykotseruba/PedestrianActionBenchmark)

* The code for the transformer submodels comes from the TSLib library [https://github.com/thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)


## Authors

## License

This project is licensed under the MIT License
