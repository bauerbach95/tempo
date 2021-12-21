import sys
import argparse
import os
import anndata
from . import unsupervised_alg


# python run_tempo.py -f /Users/benauerbach/Desktop/fitzgerald_male_control_subsample_experiments/cluster_0/subsample_0/adata.h5ad -o '/users/benauerbach/desktop/figure_1/method_preds/tempo' -c '/users/benauerbach/desktop/figure_1/tempo_config.txt'


# python run_tempo.py -f '/Users/benauerbach/Dropbox/BAM Files/brown_scn_data.h5ad' -o '/Users/benauerbach/Dropbox/BAM Files/tempo_exp_1_cells_core_clock' -c '/Users/benauerbach/Desktop/tempo/paper_figures/method_config_dicts/dataset_configs/tempo_brown_clock_only.txt'
# python run_tempo.py -f '/Users/benauerbach/Dropbox/BAM Files/brown_scn_data.h5ad' -o '/Users/benauerbach/Dropbox/BAM Files/tempo_exp_1_cells_core_clock_and_neuron_cyclers' -c '/Users/benauerbach/Desktop/tempo/paper_figures/method_config_dicts/dataset_configs/tempo_brown_clock_and_neuron_cyclers.txt'



# python run_tempo.py -f '/Users/benauerbach/Dropbox/BAM Files/brown_scn_data_exp_2.h5ad' -o '/Users/benauerbach/Dropbox/BAM Files/tempo_exp_2_cells_core_clock' -c '/Users/benauerbach/Desktop/tempo/paper_figures/method_config_dicts/dataset_configs/tempo_brown_clock_only.txt'
# python run_tempo.py -f '/Users/benauerbach/Dropbox/BAM Files/brown_scn_data_exp_2.h5ad' -o '/Users/benauerbach/Dropbox/BAM Files/tempo_exp_2_cells_core_clock_and_neuron_cyclers' -c '/Users/benauerbach/Desktop/tempo/paper_figures/method_config_dicts/dataset_configs/tempo_brown_clock_and_neuron_cyclers.txt'





def run(adata_path, folder_out, config_dict_path):





	# --- LOAD THE CONFIG DICT ---
	with open(config_dict_path) as file_obj:
		config_dict = eval(file_obj.read())



	# --- LOAD ADATA ---
	adata = anndata.read_h5ad(adata_path)



	# --- START THE TIMER ---
	import time
	start_time = time.time()



	# --- RUN TEMPO ---
	unsupervised_alg.run(adata = adata,
		folder_out = folder_out,
		**config_dict)


	# --- END THE TIMER ---
	end_time = time.time()



	# --- WRITE THE RUN TIME OUT --
	run_time = end_time - start_time
	with open("%s/run_time.txt" % folder_out, "wb") as file_obj:
		file_obj.write(str(run_time).encode())


def main(argv):

	# --- PARSE INPUT ARGUMENTS ---

	# ** init **
	parser = argparse.ArgumentParser()

	# ** .h5ad anndata filepath **
	parser.add_argument("-f", help=".h5ad AnnData file", required=True)

	# ** folder out for the results **
	parser.add_argument("-o", help="Folder out", required=True)

	# ** config dict for each method **
	parser.add_argument("-c", help="Filepath to JSON config file", required=True)

	# ** parse **
	args = parser.parse_args()


	# --- CALL ---

	run(adata_path = args.f, folder_out = args.o, config_dict_path = args.c)


if __name__ == "__main__":
	main(sys.argv)







