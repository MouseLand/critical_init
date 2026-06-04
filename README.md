# critical_init

Code for [Pachitariu et al 2026](https://www.nature.com/articles/s41586-026-10528-1)

For Figures 2 and 3, the data is available on [figshare](https://janelia.figshare.com/articles/dataset/Data_from_A_critical_initialization_for_biological_neural_networks_/27854448), please cite this publication if you use the dataset. For Figure 4 the data is available on [figshare](https://janelia.figshare.com/articles/dataset/Facemap_a_framework_for_modeling_neural_activity_based_on_orofacial_tracking/23712957), please cite Syeda et al 2024 if you use that dataset.

The `demo.ipynb` notebook contains code to run an example simulation of linear dynamics with a random symmetric connectivity matrix. This notebook will take < 5 min to run on an A100 GPU. The expected output is included in the notebooks.

If you use any of this code, please cite the paper:

**Marius Pachitariu, Lin Zhong, Alexa Gracias, Amanda Minisi, Crystall Lopez, Carsen Stringer. [A critical initialization for biological neural networks](https://www.nature.com/articles/s41586-026-10528-1). *Nature*, 2026.**

This code has been tested on Ubuntu 20.04 and Windows 11 with python=3.10, but python>=3.9 should work.

## installation (< 10 min)

Install python, [miniforge](https://conda-forge.org/miniforge/) or [anaconda](https://docs.anaconda.com/anaconda/install/) recommended. Check the box to add anaconda to your path. Then open a new terminal or anaconda prompt (on Windows). Create a new environment and activate it:

```
conda create -n critical_init python=3.10
conda activate critical_init
```

Now clone the github repo and cd into it and install the requirements:

```
git clone https://github.com/mouseland/critical_init.git
cd critical_init
pip install -r requirements.txt
```

If you want to run jupyter notebooks in this environment, you will also need to run:

```
pip install notebook
```

If you want to use an Nvidia GPU to accelerate the code you need to first install Nvidia drivers on your machine. We have not tested the code with MPS from pytorch.

### dependencies

There are no strict version requirements, but listed below are the versions we used on Ubuntu 20.04:

- numpy=2.1.3
- rastermap=1.0
- scikit-learn=1.6.1
- torch=2.6.0
- torchaudio=2.6.0
- tqdm=4.67.1
- matplotlib=3.10.0
- natsort=8.4.0
