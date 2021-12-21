git clone https://github.com/bauerbach95/tempo
cd tempo
conda env create -f tempo.yml
conda activate tempo
source install install_power_spherical.sh
python setup.py install