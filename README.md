# SBI-baccoemu
This repository contains codes for performing simulation-based inference (SBI) using emulated power spectra from baccoemu

## Software dependencies and datasets
This code uses `pytorch`, `numpy`, `scipy`, `matplotlib`, `jupyter`, `sbi`[^1], and `baccoemu`[^2] packages.

[^1]: <[https://academic.oup.com/mnras/article-abstract/507/4/5869/6328503?redirectedFrom=fulltext&login=true](https://www.mackelab.org/sbi/)>
[^2]: <https://bitbucket.org/rangulo/baccoemu/src/master/>

## Code description
The code is organized as follows:
    - The `SBI` folder contains the codes employed to perform sbi analysis
    - The `notebooks` folder contains example Jupyter notebooks

## Installation
```bash
conda create -n VE_SBIBE
conda activate VE_SBIBE
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install jupyter numpy scipy matplotlib
pip install sbi wandb

git clone git@github.com:daniellopezcano/SBI-baccoemu.git
cd SBI-baccoemu
pip install -e .

git clone https://DanielLopezCano@bitbucket.org/rangulo/baccoemu.git
cd baccoemu
pip install -e .
```
