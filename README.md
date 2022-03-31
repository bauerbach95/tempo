# Tempo: A Bayesian Algorithm for Circadian Phase Inference in Single-Cell RNA-Sequencing Data

## System Requirements

### Hardware Requirements
Tempo requires only a computer with enough RAM to support in-memory operations. For most datasets, >=8GB RAM should be sufficient.

### Software Requirements

#### OS requirements
Tempo has been tested on macOS 10.14.5 (Mojave), 12.2.1 (Monterey), and CentOS Linux 7.

#### Python dependencies
Tempo requires python >= 3.8 and depends on the following python packages:
  - anndata
  - numpy
  - pandas
  - scanpy>=1.6
  - scikit-image
  - scikit-learn
  - scipy
  - statsmodels
  - torchaudio
  - torchvision
  - tqdm
  - pytorch>=1.9.0

## Installation 

Tempo requires mini-conda for installation. For information on installing mini-conda for your operating system, please view https://docs.conda.io/en/latest/miniconda.html.


After installing mini-conda, run the following commands to install Tempo:
1) git clone https://github.com/bauerbach95/tempo
2) cd tempo
3) source install.sh

After installing, you can activate the conda environment containing the installed Tempo package:
conda activate tempo

To test if Tempo works properly, run the run_test.py file using using the activated environment:
python run_test.py

To deactivate the environment:
conda deactivate

## Running Tempo
Tutorials on how to run Tempo can be viewed in the tutorial folder of the repository.



