	
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
import random



def run(cycler_adata, folder_out,
	log_mean_log_disp_coef = None,
	bulk_cycler_info_path=None,
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


	# --- MAKE FOLDER OUT ---
	if not os.path.exists(folder_out):
		os.makedirs(folder_out)



	# --- SET CONFIG DICT ---
	config_dict = locals()
	keys_to_drop = ['cycler_adata', 'log_mean_log_disp_coef']
	for key in keys_to_drop:
		del config_dict[key]


	# --- MAKE THE LR DICT ---
	vi_gene_param_lr_dict = {
		"mu_loc" : mu_loc_lr,
		"mu_log_scale" : mu_log_scale_lr,
		"A_log_alpha" : A_log_alpha_lr,
		"A_log_beta" : A_log_beta_lr,
		"phi_euclid_loc" : phi_euclid_loc_lr,
		"phi_log_scale" : phi_log_scale_lr,
		'Q_prob_log_alpha' : Q_prob_log_alpha_lr,
		'Q_prob_log_beta' : Q_prob_log_beta_lr
	}



	# --- TURN ON / OFF TEST MODE ---
	if test_mode:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(0)   
		random.seed(0)
		detect_anomaly = True
		config_dict['detect_anomaly'] = detect_anomaly
	else:
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True
		detect_anomaly = False
		config_dict['detect_anomaly'] = detect_anomaly



	# --- PREP THE TENSORS NEEDED ---
	cycler_gene_X, log_L, cycler_gene_param_dict, cell_prior_dict, cycler_gene_prior_dict = prep.unsupervised_prep(cycler_adata,**config_dict)
	if not opt_phase_est_gene_params or init_variational_dist_to_prior: # set variational to priors if opt_phase_est_gene_params is False
		cycler_gene_param_dict = prep.get_zero_kl_gene_param_dict_from_gene_prior_dict(cycler_gene_prior_dict)
	if use_clock_output_only:
		clock_indices = np.where(cycler_adata.var['is_clock'])[0]
	else:
		clock_indices = np.arange(0,cycler_adata.shape[1])
	non_clock_indices = np.setdiff1d(np.arange(0,cycler_adata.shape[1]), np.where(cycler_adata.var['is_clock'])[0])



	# ** get prior theta euclid dist **
	prior_theta_euclid_dist = utils.init_distributions_from_param_dicts(cell_prior_dict = cell_prior_dict)['prior_theta_euclid']



	# --- ESTIMATE THE CELL PHASE POSTERIOR AT INIT, AND WRITE OUT ---
	
	# get cell posterior df at init
	cycler_gene_posterior_obj_init = clock_gene_posterior.ClockGenePosterior(cycler_gene_param_dict,None,num_phase_grid_points,clock_indices,use_nb=use_nb,log_mean_log_disp_coef=log_mean_log_disp_coef,min_amp=min_amp,max_amp=max_amp)
	theta_posterior_likelihood_init = cycler_gene_posterior_obj_init.compute_cell_phase_posterior_likelihood(cycler_gene_X,log_L,prior_theta_euclid_dist,num_gene_samples=100)


	# get df's
	cell_posterior_df = params_to_df.cell_multinomial_params_to_param_df(np.array(cycler_adata.obs.index), theta_posterior_likelihood_init)
	cell_prior_df = params_to_df.cell_powerspherical_params_dict_to_param_df(np.array(cycler_adata.obs.index), cell_prior_dict)
	gene_param_df = params_to_df.gene_param_dicts_to_param_df(list(cycler_adata.var_names), utils.prep_gene_params(cycler_gene_param_dict), cycler_gene_prior_dict, min_amp, max_amp)


	# write
	if not os.path.exists('%s/cell_phase_estimation' % folder_out):
		os.makedirs('%s/cell_phase_estimation' % folder_out)
	cell_posterior_fileout = '%s/cell_phase_estimation/cell_posterior_init.tsv' % folder_out # % (folder_out)
	cell_posterior_df.to_csv(cell_posterior_fileout,sep='\t')
	cell_prior_fileout = '%s/cell_phase_estimation/cell_prior_init.tsv' % folder_out #  % (folder_out)
	cell_prior_df.to_csv(cell_prior_fileout,sep='\t')
	gene_param_df_fileout = '%s/cell_phase_estimation/gene_prior_and_posterior_init.tsv' % folder_out #  (folder_out)
	gene_param_df.to_csv(gene_param_df_fileout,sep='\t')


	# --- ESTIMATE CELL PHASE ---


	if opt_phase_est_gene_params:
		opt_cycler_theta_posterior_likelihood, opt_cycler_gene_param_dict_unprepped = clock_posterior_opt.run(gene_X = cycler_gene_X,
			clock_indices = clock_indices,
			log_L = log_L,
			gene_param_dict = cycler_gene_param_dict,
			gene_prior_dict = cycler_gene_prior_dict,
			min_amp = min_amp,
			max_amp = max_amp,
			prior_theta_euclid_dist = prior_theta_euclid_dist, # clock_posterior_dist
			folder_out = '%s/cell_phase_estimation' % folder_out, # '%s/clock_and_confident_hv_inference' % folder_out,
			learning_rate_dict = vi_gene_param_lr_dict,
			gene_param_grad_dict = gene_param_grad_dict, # None,
			use_nb = use_nb,
			log_mean_log_disp_coef = log_mean_log_disp_coef,
			num_grid_points = num_phase_grid_points,
			num_cell_samples = num_phase_est_cell_samples,
			num_gene_samples = num_phase_est_gene_samples,
			vi_max_epochs = vi_max_epochs,
			vi_print_epoch_loss = vi_print_epoch_loss,
			vi_improvement_window = vi_improvement_window,
			vi_convergence_criterion = vi_convergence_criterion,
			vi_lr_scheduler_patience = vi_lr_scheduler_patience,
			vi_lr_scheduler_factor = vi_lr_scheduler_factor,
			vi_batch_size = vi_batch_size,
			vi_num_workers = vi_num_workers,
			vi_pin_memory = vi_pin_memory,
			batch_indicator_mat = None,
			detect_anomaly = detect_anomaly,
			use_clock_output_only = use_clock_output_only)
	else:
		opt_cycler_theta_posterior_likelihood = theta_posterior_likelihood_init
		opt_cycler_gene_param_dict_unprepped = cycler_gene_param_dict



	# --- WRITE OUT CELL PHASE / GENE PARAMETER ESTIMATES ---

	# get df's
	cell_posterior_df = params_to_df.cell_multinomial_params_to_param_df(np.array(cycler_adata.obs.index), opt_cycler_theta_posterior_likelihood)
	cell_prior_df = params_to_df.cell_powerspherical_params_dict_to_param_df(np.array(cycler_adata.obs.index), cell_prior_dict)
	gene_param_df = params_to_df.gene_param_dicts_to_param_df(list(cycler_adata.var_names), utils.prep_gene_params(opt_cycler_gene_param_dict_unprepped), cycler_gene_prior_dict, min_amp, max_amp)


	# write
	cell_posterior_fileout = '%s/cell_phase_estimation/cell_posterior.tsv' % folder_out
	cell_posterior_df.to_csv(cell_posterior_fileout,sep='\t')
	cell_prior_fileout = '%s/cell_phase_estimation/cell_prior.tsv' % folder_out
	cell_prior_df.to_csv(cell_prior_fileout,sep='\t')
	gene_param_df_fileout = '%s/cell_phase_estimation/gene_prior_and_posterior.tsv' % folder_out
	gene_param_df.to_csv(gene_param_df_fileout,sep='\t')




	return opt_cycler_theta_posterior_likelihood, opt_cycler_gene_param_dict_unprepped








