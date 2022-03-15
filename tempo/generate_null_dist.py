		

import sys
import numpy as np
import torch
import os
import pandas as pd 
import scipy
from scipy import stats
import copy
import statsmodels
from statsmodels import nonparametric
from statsmodels.nonparametric import kernel_regression

# tempo imports
from . import clock_posterior_opt
from . import gene_fit
from . import prep
from . import utils
from . import hvg_selection
from . import params_to_df
from . import cell_posterior
from . import estimate_mean_disp_relationship
from . import generate_null_dist
from . import objective_functions
from . import clock_gene_posterior
from . import est_cell_phase_from_current_cyclers
from . import compute_clock_evidence
import tqdm


def run(cycler_adata, folder_out,
	log_mean_log_disp_coef = None,
	num_null_shuffles = 5,
	gene_acrophase_prior_path=None,
	cell_phase_prior_path = None,
	core_clock_gene_path=None,
	gene_param_grad_dict = None,
	reference_gene = 'Arntl',
	min_gene_prop = 1e-5,
	min_amp = 0.0, # ** prep parameters **
	max_amp = 1.5 / np.log10(np.e),
	init_mesor_scale_val = 0.3,
	prior_mesor_scale_val = 0.5,
	init_amp_loc_val = 0.5,
	init_amp_scale_val = 3,
	prior_amp_alpha_val = 1,
	prior_amp_beta_val = 1,
	known_cycler_init_shift_95_interval = (1.0 / 12.0) * np.pi,
	unknown_cycler_init_shift_95_interval = (1.0 / 12.0) * np.pi,
	known_cycler_prior_shift_95_interval = (2.0 / 12.0) * np.pi,
	init_clock_Q_prob_alpha = 90,
	init_clock_Q_prob_beta = 10,
	init_non_clock_Q_prob_alpha = 1,
	init_non_clock_Q_prob_beta = 9,
	prior_clock_Q_prob_alpha = 90,
	prior_clock_Q_prob_beta = 10,
	prior_non_clock_Q_prob_alpha = 1, 
	prior_non_clock_Q_prob_beta = 9,
	use_noninformative_phase_prior = True,
	use_nb = True,
	mu_loc_lr = 1e-1,
	mu_log_scale_lr = 1e-1,
	A_log_alpha_lr = 1e-1,
	A_log_beta_lr = 1e-1,
	phi_euclid_loc_lr = 5.0,
	phi_log_scale_lr = 1.0,
	Q_prob_log_alpha_lr = 1e-1,
	Q_prob_log_beta_lr = 1e-1,
	num_phase_grid_points = 24,
	num_phase_est_cell_samples = 10,
	num_phase_est_gene_samples = 10,
	num_harmonic_est_cell_samples = 5,
	num_harmonic_est_gene_samples = 3,
	vi_max_epochs = 300,
	vi_print_epoch_loss = True,
	vi_improvement_window = 10,
	vi_convergence_criterion = 1e-3,
	vi_lr_scheduler_patience = 10,
	vi_lr_scheduler_factor = 0.1,
	vi_batch_size = 3000,
	vi_num_workers = 0,
	vi_pin_memory = False,
	test_mode = False,
	use_clock_input_only = False,
	use_clock_output_only = True,
	frac_pos_cycler_samples_threshold = 0.1,
	A_loc_pearson_residual_threshold = 0.5,
	confident_cell_interval_size_threshold = 12.0,
	opt_phase_est_gene_params = True,
	init_variational_dist_to_prior = False,
	**kwargs):


	


	# --- TURN OFF VERBOSE ---
	vi_print_epoch_loss = False


	# --- SET CONFIG DICT ---
	config_dict = locals()
	keys_to_drop = ['cycler_adata', 'log_mean_log_disp_coef', 'folder_out']
	for key in keys_to_drop:
		del config_dict[key]




	# --- INIT ---
	null_log_evidence_list = []

	# -- GENERATE NULL DIST ---
	for shuffle_index in tqdm.tqdm(range(0,num_null_shuffles)):

		# ** randomly shuffle the expression of individual genes across cells **
		try:
			X = np.array(cycler_adata.X.todense())
		except:
			X = np.array(cycler_adata.X)
		X = X[:, np.random.permutation(X.shape[1])]
		cycler_adata.X = X



		# ** estimate cell phase **


		# ** get the gene param folder out **
		shuffle_folder_out = '%s/shuffle_%s' % (folder_out, shuffle_index)
		opt_cycler_theta_posterior_likelihood, opt_cycler_gene_param_dict_unprepped = est_cell_phase_from_current_cyclers.run(cycler_adata, shuffle_folder_out, log_mean_log_disp_coef, **config_dict)






		# ** compute evidence and add **
		clock_log_evidence = compute_clock_evidence.run(cycler_adata, opt_cycler_gene_param_dict_unprepped,
			opt_cycler_theta_posterior_likelihood, log_mean_log_disp_coef, **config_dict)
		null_log_evidence_list.append(clock_log_evidence)




	# --- TURN INTO NUMPY ----
	null_log_evidence_list = np.array(null_log_evidence_list)


	# --- WRITE THE NULL LL LIST TO THE FOLDER ---
	fileout = '%s/null_log_evidence_vec.txt' % (folder_out)
	np.savetxt(fileout,null_log_evidence_list)


	return null_log_evidence_list













