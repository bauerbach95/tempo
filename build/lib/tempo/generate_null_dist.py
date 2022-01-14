import torch
import copy
import numpy as np


# tempo imports
from . import gene_fit
from . import utils
from . import objective_functions
from . import prep




def generate(adata,null_head_folder_out,learning_rate_dict,log_mean_log_disp_coef,min_amp,max_amp,num_gene_samples,use_nb,num_shuffles,config_dict):


	# ** drop the config dict key for folder out **
	del config_dict['folder_out']

	

	# --- INIT ---
	null_ll_list = []

	# -- GENERATE NULL DIST ---
	for shuffle_index in range(0,num_shuffles):


		# ** do the prep **
		gene_X, log_L, gene_param_dict, cell_prior_dict, gene_prior_dict = prep.unsupervised_prep(adata,**config_dict)


		# ** get theta rand **
		theta_rand = torch.rand(gene_X.shape[0]) * 2 * np.pi


		# ** get the gene param folder out **
		shuffle_folder_out = '%s/shuffle_%s' % (null_head_folder_out, shuffle_index)


		# ** compute the conditional gene posterior **
		optimizer, rand_gene_param_dict = gene_fit.gene_fit(gene_X = gene_X, 
			log_L = log_L, 
			gene_param_dict = gene_param_dict, 
			gene_prior_dict = gene_prior_dict,
			folder_out = shuffle_folder_out,
			theta = theta_rand,
			learning_rate_dict = learning_rate_dict,
			gene_param_grad_dict = None,
			log_mean_log_disp_coef = log_mean_log_disp_coef,
			**config_dict)



		# # ** get the loc scale dict **
		# gene_param_loc_scale_dict = utils.get_distribution_loc_and_scale(gene_param_dict = rand_gene_param_dict, min_amp = min_amp, max_amp = max_amp, prep = True)

		# # ** compute expectation of the LL **
		# random_harmonic_expectation_cell_log_likelihood = objective_functions.compute_expectation_log_likelihood(gene_X, log_L,
		# 	theta_sampled = theta_rand.reshape(-1,1), mu_loc = gene_param_loc_scale_dict['mu_loc'], A_loc = gene_param_loc_scale_dict['A_loc'],
		# 	phi_euclid_loc = gene_param_loc_scale_dict['phi_euclid_loc'], Q_prob_loc = gene_param_loc_scale_dict['Q_prob_loc'],
		# 	use_is_cycler_indicators = gene_param_loc_scale_dict['Q_prob_loc'] is not None, exp_over_cells = True, use_flat_model = False,
		# 	use_nb = use_nb, log_mean_log_disp_coef = log_mean_log_disp_coef, rsample = False)



		# ** get distribution dict **
		distrib_dict = utils.init_distributions_from_param_dicts(gene_param_dict = rand_gene_param_dict, max_amp = max_amp, min_amp = min_amp, prep = True)

		# ** compute expectation of the LL **
		random_harmonic_expectation_cell_log_likelihood = objective_functions.compute_expectation_log_likelihood(gene_X = gene_X, log_L = log_L, theta_sampled = theta_rand.reshape(-1,1),
			mu_dist = distrib_dict['mu'], A_dist = distrib_dict['A'], phi_euclid_dist = distrib_dict['phi_euclid'], Q_prob_dist = distrib_dict['Q_prob'],
			num_gene_samples = num_gene_samples, exp_over_cells = True, use_flat_model = False,
			use_nb = use_nb, log_mean_log_disp_coef = log_mean_log_disp_coef, rsample = False, use_is_cycler_indicators = distrib_dict['Q_prob'] is not None)



		
		# ** compute expectation over cells **
		random_harmonic_expectation_ll = torch.mean(random_harmonic_expectation_cell_log_likelihood)

		
		# --- ADD TO LIST ---
		null_ll_list.append(random_harmonic_expectation_ll.item())


	# --- TURN INTO NUMPY ----
	null_ll_vec = np.array(null_ll_list)


	# --- WRITE THE NULL LL LIST TO THE FOLDER ---
	fileout = '%s/null_ll_vec.txt' % (null_head_folder_out)
	np.savetxt(fileout,null_ll_vec)


	return null_ll_vec






	

