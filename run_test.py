import os
import tempo
from tempo import run_tempo
os.chdir("test_data")
run_tempo.run(adata_path = 'adata.h5ad', folder_out = 'tempo_test_out', config_dict_path = 'config.txt')




