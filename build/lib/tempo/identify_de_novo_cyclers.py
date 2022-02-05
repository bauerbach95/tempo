
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



# adata, opt_cycler_theta_posterior_likelihood, algorithm_step, alg_result_head_folder, config_dict
def run(hv_adata, opt_cycler_theta_posterior_likelihood, alg_step_subfolder, log_mean_log_disp_coef,
	bulk_cycler_info_path,
	core_clock_gene_path,
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
	use_nb = True,
	mu_loc_lr = 1e-1, # ** conditional posterior opt parameters **
	mu_log_scale_lr = 1e-1,
	A_log_alpha_lr = 1e-1,
	A_log_beta_lr = 1e-1,
	phi_euclid_loc_lr = 5.0,
	phi_log_scale_lr = 1.0,
	Q_prob_log_alpha_lr = 1e-1,
	Q_prob_log_beta_lr = 1e-1,
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
	frac_pos_cycler_samples_threshold = 0.1,
	A_loc_pearson_residual_threshold = 0.5,
	confident_cell_interval_size_threshold = 12.0,
	**kwargs):







	# --- MAKE FOLDER OUT ---
	if not os.path.exists(alg_step_subfolder):
		os.makedirs(alg_step_subfolder)



	# --- SET CONFIG DICT ---
	config_dict = locals()
	keys_to_drop = ['hv_adata', 'log_mean_log_disp_coef']
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






	# --- GET HIGHLY CONFIDENT CELLS ---

	print("--- IDENTIFYING HIGHLY CONFIDENT CELLS BASED ON CELL PHASE POSTERIOR ---")

	if confident_cell_interval_size_threshold is None:
		confident_cell_indices = np.arange(0,hv_adata.shape[0])
	else:
		# ** make the dist **
		cell_posterior_dist = cell_posterior.ThetaPosteriorDist(opt_cycler_theta_posterior_likelihood.detach())

		# ** compute the confidence intervals **
		cell_confidence_intervals = cell_posterior_dist.compute_confidence_interval(confidence=0.90)
		cell_confidence_intervals = (np.sum(cell_confidence_intervals,axis=1) / cell_posterior_dist.num_grid_points) * 24.0

		# ** get confident cell indices **
		confident_cell_indices = np.where(cell_confidence_intervals <= confident_cell_interval_size_threshold)[0]





	# --- RESTRICT ADATA TO CONFIDENT CELLS ---
	hv_adata = hv_adata[confident_cell_indices,:]







	# --- BURNING IN HVG PARAMETERS WHEN Q FIXED TO 1 ---

	print("--- BURNING IN HARMONIC PARAMETERS FOR HIGHLY VARIABLE GENES THAT ARE NON-CYCLERS ---")


	# ** prep **
	hv_gene_X, log_L, hv_gene_param_dict, cell_prior_dict, hv_gene_prior_dict = prep.unsupervised_prep(hv_adata,**config_dict)
	hv_gene_prior_dict['prior_Q_prob_alpha'] = 999.0 * torch.ones(hv_adata.shape[1])
	hv_gene_prior_dict['prior_Q_prob_beta'] = torch.ones(hv_adata.shape[1])
	hv_gene_param_dict['Q_prob_log_alpha'] = torch.nn.Parameter(torch.log(999.0 * torch.ones(hv_adata.shape[1])).detach(),requires_grad=True)
	hv_gene_param_dict['Q_prob_log_beta'] = torch.nn.Parameter(torch.log(torch.ones(hv_adata.shape[1])).detach(),requires_grad=True)
	gene_param_grad_dict = {
		"mu_loc" : True, "mu_log_scale" : True,
		"phi_euclid_loc" : True, "phi_log_scale" : True,
		"A_log_alpha" : True, "A_log_beta" : True,
		"Q_prob_log_alpha" : False, "Q_prob_log_beta" : False,
	}


	# ** run **
	if num_harmonic_est_gene_samples > 1:
		_, opt_hv_gene_param_dict_unprepped = gene_fit.gene_fit(gene_X = hv_gene_X, 
			log_L = log_L, 
			gene_param_dict = hv_gene_param_dict, 
			gene_prior_dict = hv_gene_prior_dict,
			folder_out = "%s/de_novo_cycler_id_preinference_burn_in" % (alg_step_subfolder),  # '%s/hv_preinference' % folder_out,
			learning_rate_dict = vi_gene_param_lr_dict,
			theta_posterior_likelihood = opt_cycler_theta_posterior_likelihood[confident_cell_indices,:], # opt_clock_theta_posterior_likelihood
			gene_param_grad_dict = gene_param_grad_dict,
			max_iters = vi_max_epochs, 
			num_cell_samples = num_harmonic_est_cell_samples,
			num_gene_samples = 1,
			max_amp = max_amp,
			min_amp = min_amp,
			print_epoch_loss = vi_print_epoch_loss,
			improvement_window = vi_improvement_window,
			convergence_criterion = vi_convergence_criterion,
			lr_scheduler_patience = vi_lr_scheduler_patience,
			lr_scheduler_factor = vi_lr_scheduler_factor,
			use_flat_model = False,
			batch_size = vi_batch_size,
			num_workers = vi_num_workers,
			pin_memory = vi_pin_memory,
			use_nb = use_nb,
			log_mean_log_disp_coef = log_mean_log_disp_coef,
			batch_indicator_mat = None,
			detect_anomaly = detect_anomaly,
			expectation_point_est_only = False)
	else:
		opt_hv_gene_param_dict_unprepped = hv_gene_param_dict









	# --- FIT HVG PARAMETERS WHEN Q FIXED TO 1 ---

	print("--- FITTING HARMONIC PARAMETERS FOR HIGHLY VARIABLE GENES THAT ARE NON-CYCLERS ---")



	# ** run **
	_, opt_hv_gene_param_dict_unprepped = gene_fit.gene_fit(gene_X = hv_gene_X, 
		log_L = log_L, 
		gene_param_dict = opt_hv_gene_param_dict_unprepped, # hv_gene_param_dict, 
		gene_prior_dict = hv_gene_prior_dict,
		folder_out = "%s/de_novo_cycler_id_preinference" % (alg_step_subfolder),  # '%s/hv_preinference' % folder_out,
		learning_rate_dict = vi_gene_param_lr_dict,
		theta_posterior_likelihood = opt_cycler_theta_posterior_likelihood[confident_cell_indices,:], # opt_clock_theta_posterior_likelihood
		gene_param_grad_dict = gene_param_grad_dict,
		max_iters = vi_max_epochs, 
		num_cell_samples = num_harmonic_est_cell_samples,
		num_gene_samples = num_harmonic_est_gene_samples,
		max_amp = max_amp,
		min_amp = min_amp,
		print_epoch_loss = vi_print_epoch_loss,
		improvement_window = vi_improvement_window,
		convergence_criterion = vi_convergence_criterion,
		lr_scheduler_patience = vi_lr_scheduler_patience,
		lr_scheduler_factor = vi_lr_scheduler_factor,
		use_flat_model = False,
		batch_size = vi_batch_size,
		num_workers = vi_num_workers,
		pin_memory = vi_pin_memory,
		use_nb = use_nb,
		log_mean_log_disp_coef = log_mean_log_disp_coef,
		batch_indicator_mat = None,
		detect_anomaly = detect_anomaly)



	# --- BURN IN Q FOR HVG ---

	print("--- BURNING IN CYCLING INDICATOR PARAMETER FOR HIGHLY VARIABLE GENES THAT ARE NON-CYCLERS ---")


	# ** prep **
	hv_gene_X, log_L, _, _, hv_gene_prior_dict = prep.unsupervised_prep(hv_adata,**config_dict)
	gene_param_grad_dict = {
		"mu_loc" : False, "mu_log_scale" : False,
		"phi_euclid_loc" : False, "phi_log_scale" : False,
		"A_log_alpha" : False, "A_log_beta" : False,
		"Q_prob_log_alpha" : True, "Q_prob_log_beta" : True,
	}



	# ** run **
	if num_harmonic_est_gene_samples > 1:
		_, opt_hv_gene_param_dict_unprepped = gene_fit.gene_fit(gene_X = hv_gene_X, 
			log_L = log_L, 
			gene_param_dict = opt_hv_gene_param_dict_unprepped, 
			gene_prior_dict = hv_gene_prior_dict,
			folder_out = "%s/de_novo_cycler_id_burn_in" % (alg_step_subfolder), # '%s/hv_preinference_Q_fit' % folder_out,
			learning_rate_dict = vi_gene_param_lr_dict,
			theta_posterior_likelihood = opt_cycler_theta_posterior_likelihood[confident_cell_indices,:], # opt_clock_theta_posterior_likelihood
			gene_param_grad_dict = gene_param_grad_dict,
			max_iters = vi_max_epochs, 
			num_cell_samples = num_harmonic_est_cell_samples,
			num_gene_samples = 1,
			max_amp = max_amp,
			min_amp = min_amp,
			print_epoch_loss = vi_print_epoch_loss,
			improvement_window = vi_improvement_window,
			convergence_criterion = vi_convergence_criterion,
			lr_scheduler_patience = vi_lr_scheduler_patience,
			lr_scheduler_factor = vi_lr_scheduler_factor,
			use_flat_model = False,
			batch_size = vi_batch_size,
			num_workers = vi_num_workers,
			pin_memory = vi_pin_memory,
			use_nb = use_nb,
			log_mean_log_disp_coef = log_mean_log_disp_coef,
			batch_indicator_mat = None,
			detect_anomaly = detect_anomaly,
			expectation_point_est_only = False) # True
	else:
		opt_hv_gene_param_dict_unprepped = opt_hv_gene_param_dict_unprepped





	# --- FIT Q FOR HVG ---

	print("--- FITTING CYCLING INDICATOR PARAMETER FOR HIGHLY VARIABLE GENES THAT ARE NON-CYCLERS ---")


	# ** prep **
	hv_gene_X, log_L, _, _, hv_gene_prior_dict = prep.unsupervised_prep(hv_adata,**config_dict)
	gene_param_grad_dict = {
		"mu_loc" : False, "mu_log_scale" : False,
		"phi_euclid_loc" : False, "phi_log_scale" : False,
		"A_log_alpha" : False, "A_log_beta" : False,
		"Q_prob_log_alpha" : True, "Q_prob_log_beta" : True,
	}



	# ** run **
	_, opt_hv_gene_param_dict_unprepped = gene_fit.gene_fit(gene_X = hv_gene_X, 
		log_L = log_L, 
		gene_param_dict = opt_hv_gene_param_dict_unprepped, 
		gene_prior_dict = hv_gene_prior_dict,
		folder_out = "%s/de_novo_cycler_id" % (alg_step_subfolder), # '%s/hv_preinference_Q_fit' % folder_out,
		learning_rate_dict = vi_gene_param_lr_dict,
		theta_posterior_likelihood = opt_cycler_theta_posterior_likelihood[confident_cell_indices,:], # opt_clock_theta_posterior_likelihood
		gene_param_grad_dict = gene_param_grad_dict,
		max_iters = vi_max_epochs, 
		num_cell_samples = num_harmonic_est_cell_samples,
		num_gene_samples = num_harmonic_est_gene_samples,
		max_amp = max_amp,
		min_amp = min_amp,
		print_epoch_loss = vi_print_epoch_loss,
		improvement_window = vi_improvement_window,
		convergence_criterion = vi_convergence_criterion,
		lr_scheduler_patience = vi_lr_scheduler_patience,
		lr_scheduler_factor = vi_lr_scheduler_factor,
		use_flat_model = False,
		batch_size = vi_batch_size,
		num_workers = vi_num_workers,
		pin_memory = vi_pin_memory,
		use_nb = use_nb,
		log_mean_log_disp_coef = log_mean_log_disp_coef,
		batch_indicator_mat = None,
		detect_anomaly = detect_anomaly)






	# --- IDENTIFY THE HVG THAT ARE HIGHLY CONFIDENT CYCLERS ---

	print("--- IDENTIFYING DE NOVO CYCLERS ---")


	# ** compute the pearson residuals of the amplitude loc's (based on mesor - amplitude relationship across all genes) **
	opt_hv_gene_param_loc_scale_dict = utils.get_distribution_loc_and_scale(gene_param_dict=opt_hv_gene_param_dict_unprepped,min_amp=min_amp,max_amp=max_amp,prep=True)
	mu_loc = opt_hv_gene_param_loc_scale_dict['mu_loc'].detach().numpy()
	A_loc = opt_hv_gene_param_loc_scale_dict['A_loc'].detach().numpy()
	kernel_model = statsmodels.nonparametric.kernel_regression.KernelReg(A_loc, mu_loc.reshape(-1,1), var_type = ['c'], bw = [0.1 / np.log10(np.e)]) 
	pred_A,marginal_effects = kernel_model.fit()
	est_std = (np.mean((A_loc - pred_A)**2)) ** 0.5
	A_loc_pearson_residuals = (A_loc - pred_A) / est_std



	# ** compute the fraction of Q = 1 for cyclers **
	num_Q_samples = 100
	hvg_gene_param_dist_dict = utils.init_distributions_from_param_dicts(gene_param_dict = opt_hv_gene_param_dict_unprepped, prep = True)
	Q_samples = utils.get_is_cycler_samples_from_dist(hvg_gene_param_dist_dict['Q_prob'],num_gene_samples=num_Q_samples,rsample=False)
	num_pos_cycler_samples = torch.sum(Q_samples,dim=0).detach().numpy()
	frac_pos_cycler_samples = num_pos_cycler_samples / num_Q_samples



	# ** get HV gene param df **
	hv_gene_param_df = params_to_df.gene_param_dicts_to_param_df(list(hv_adata.var_names), utils.prep_gene_params(opt_hv_gene_param_dict_unprepped), hv_gene_prior_dict, min_amp, max_amp)


	# ** add A_loc_pearson_residuals and frac_pos_cycler_samples to HV adata and gene parameter DF **
	hv_gene_param_df['A_loc_pearson_residual'] = A_loc_pearson_residuals
	hv_gene_param_df['frac_pos_cycler_samples'] = frac_pos_cycler_samples
	hv_adata.var['A_loc_pearson_residual'] = A_loc_pearson_residuals
	hv_adata.var['frac_pos_cycler_samples'] = frac_pos_cycler_samples



	# ** write out gene param DF **
	hv_gene_param_df_fileout = '%s/de_novo_cycler_id/gene_prior_and_posterior.tsv' % alg_step_subfolder # % (folder_out)
	hv_gene_param_df.to_csv(hv_gene_param_df_fileout,sep='\t')



	# ** get cycler genes based on frac_pos_cycler_samples and A_loc_pearson_residual_threshold **
	confident_hv_gene_param_df = hv_gene_param_df[(hv_gene_param_df['frac_pos_cycler_samples'] >= frac_pos_cycler_samples_threshold) & (hv_gene_param_df['A_loc_pearson_residual'] >= A_loc_pearson_residual_threshold)]
	new_de_novo_cycler_genes = np.array(list(confident_hv_gene_param_df.index))
	# adata.var.loc[new_de_novo_cycler_genes,'is_cycler'] = True
		

	return new_de_novo_cycler_genes







