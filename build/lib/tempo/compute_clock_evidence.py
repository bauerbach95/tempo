
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



def run(cycler_adata, opt_cycler_gene_param_dict_unprepped, opt_cycler_theta_posterior_likelihood, log_mean_log_disp_coef, max_amp, min_amp,
	num_phase_est_cell_samples, num_phase_est_gene_samples, use_nb, **kwargs):

	
	# ** get clock X and log_L **
	clock_adata = cycler_adata[:,cycler_adata.var['is_clock']]
	try:
		clock_X = torch.Tensor(np.array(clock_adata.X.todense()))
	except:
		clock_X = torch.Tensor(np.array(clock_adata.X))
	log_L = torch.Tensor(np.array(cycler_adata.obs['log_L']))


	# ** restrict opt_cycler_gene_param_dict_unprepped to just the core clock genes **
	clock_indices = np.where(np.array(cycler_adata.var['is_clock']))[0]
	opt_clock_gene_param_dict_unprepped = {}
	for key in opt_cycler_gene_param_dict_unprepped:
		opt_clock_gene_param_dict_unprepped[key] = opt_cycler_gene_param_dict_unprepped[key][clock_indices]


	
	# ** init gene distrib dict **
	distrib_dict = utils.init_distributions_from_param_dicts(gene_param_dict = opt_clock_gene_param_dict_unprepped, max_amp = max_amp, min_amp = min_amp, prep = True)
	
	# ** sample theta distribution **
	theta_sampled = cell_posterior.ThetaPosteriorDist(opt_cycler_theta_posterior_likelihood).sample(num_samples=num_phase_est_cell_samples) # ** get theta sampled **
	

	# ** compute gene ll in each cell and over cell / gene sample **
	clock_cell_gene_ll_sampled = objective_functions.compute_sample_log_likelihood(clock_X, log_L,
	    theta_sampled = theta_sampled,
	    mu_dist = distrib_dict['mu'], A_dist = distrib_dict['A'], phi_euclid_dist = distrib_dict['phi_euclid'], Q_prob_dist = distrib_dict['Q_prob'],
	    num_cell_samples = num_phase_est_cell_samples, num_gene_samples = num_phase_est_gene_samples, use_flat_model = False,
	    use_nb = use_nb, log_mean_log_disp_coef = log_mean_log_disp_coef, rsample = False, use_is_cycler_indicators = distrib_dict['Q_prob'] is not None)
	
	# ** get the mc log evidence **
	clock_log_evidence_sampled = torch.sum(torch.sum(clock_cell_gene_ll_sampled,dim=0),dim=0).flatten()
	clock_log_evidence = torch.mean(clock_log_evidence_sampled).item()

	return clock_log_evidence






