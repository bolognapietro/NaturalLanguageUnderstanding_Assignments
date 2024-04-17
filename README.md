## Getting started

We suggest you install [Anaconda](https://www.anaconda.com/download) on your laptop and import the conda environment that we have prepared for you. The reason for this is to give you the same library versions that we used to test the labs. However, if you are not a conda lover, you can manually install on your favourite virtual env the libraries listed in the `requirements.txt` file.

```bash
conda env create -f nlu_env.yaml -n nlu24
conda activate nlu24
```
If you have a Mac or a WindowsA Multimodal Framework for State of Mind Assessment with Sentiment Pre-classification or you do not have a dedicated Nvidia gpu, you can install the environment in this way:

```bash
conda create -n nlu24 python=3.10.13
conda activate nlu24
pip install -r requirements_no_cuda.txt
```

To launch a lab run this line of code:
```bash
jupyter notebook
```


Then, you have to choose the lab that you want to open. 
