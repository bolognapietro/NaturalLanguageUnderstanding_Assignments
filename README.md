## Getting started

First of all, initialize the workspace:
```bash
git clone https://github.com/bolognapietro/NLU_assignments
cd NLU_assignments
```
Then install [Anaconda](https://www.anaconda.com/download) on your laptop and import the conda environment. However, if you are not a conda lover, you can manually install on your favourite virtual env the libraries listed in the `requirements.txt` file.
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

Open VScode and run the code using the conda environment:
