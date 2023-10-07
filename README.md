# VeinGAN
> Synthetic Image Generation of Veins using Generative Adversarial Networks

### Usage

#### Requirements
Use this script to install the requirements for this implementation.

```python
pip install -r requirements.txt
```

#### Download Dataset
To download the dataset locally, use this script as follows.
```python
python -m scripts.download <dataset>
```

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
python -m scripts.evaluate <target>
```

### TBA
