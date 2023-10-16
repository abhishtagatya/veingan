# VeinGAN
> Synthetic Image Generation of Veins using Generative Adversarial Networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-GCMNedTP3j0v8AhIl7Rg3qGMDB-9lj7?usp=sharing)

### Usage

#### Requirements
Use this script to install the requirements for this implementation.

```python
pip install -r requirements.txt
```

#### Download Dataset
To download the dataset locally, use this script as follows.
```python
python -m scripts.download --dataset=<dataset> --target=<target>
```

Parameter
- `dataset` : A Kaggle Dataset Name or Key
  - `kaggle-fv` Kaggle Finger Vein Dataset
  - `default` Default Dataset
- `target` : Target Directory for Dataset [default: ./veingan-tmp/dataset/]


#### Generate Images
To train the model and generate synthetic images, use the following command and adjust the given parameters.
```python
python -m scripts.generate <model> <dataset> <target> --verbose --epoch=<epoch>
```

Parameter
- `model` : The GAN model to generate the image
- `dataset` : The dataset to train on
- `target` : The target directory to save the images
- `verbose` : Argument to enable verbose output of model progress
- `epoch` : The amount of cycle to train on


#### Evaluate Result
Evaluate the resulting synthetic images on to a more capable classifier to measure the deception rate.
```python
python -m scripts.evaluate <method> <target> --configuration=<configuration>
```

Parameter
- `method` : Method of Evaluation
  - `osvm+vgg` One-Class SVM + VGG Feature Ext. Novelty Score
- `target` : Target Directory of Evaluation
- `configuration` : Specific Configuration for Evaluation Method


### TBA
