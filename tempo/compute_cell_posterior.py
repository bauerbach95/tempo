import sys
import os
import anndata
import copy
import torch
import pandas as pd
import numpy as np
import power_spherical


# tempo imports
from . import objective_functions
from . import cell_posterior
from . import utils



def compute_cell_posterior(gene_X, log_L, num_grid_points, prior_theta_euclid_dist, mu_sampled, A_sampled, phi_sampled, Q_sampled = None, B_sampled = None, use_nb = True, log_mean_log_disp_coef = None):


	# --- DO INPUT CHECKS ---

	# ** check that sampled gene parameters are all of the same number of samples **
	gene_params_sampled = [mu_sampled, A_sampled, phi_sampled]
	if Q_sampled is not None:
		gene_params_sampled += [Q_sampled]
	if B_sampled is not None:
		gene_params_sampled += [B_sampled]
	num_samples_per_gene_param = set(list(map(lambda x: x.shape[0], gene_params_sampled)))
	if len(num_samples_per_gene_param) > 1:
		raise Exception("Error: gene parameters do not contain equal samples. Unique num samples per gene param: %s" % str(num_samples_per_gene_param))
	num_gene_samples,num_genes = mu_sampled.shape


	# ** make sure that log_mean_log_disp_coef supplied if use_nb = True **
	if use_nb and log_mean_log_disp_coef is None:
		raise Exception("Error: us_nb = True, but no log_mean_log_disp_coef supplied.")


	# --- RESHAPE SAMPLED GENE PARAMETERS to [1,1,num_gene_samples,num_genes] ---

	mu_sampled = mu_sampled.unsqueeze(0).unsqueeze(0)
	A_sampled = A_sampled.unsqueeze(0).unsqueeze(0)
	phi_sampled = phi_sampled.unsqueeze(0).unsqueeze(0)
	if Q_sampled is not None:
		Q_sampled = Q_sampled.unsqueeze(0).unsqueeze(0)
	if B_sampled is not None:
		B_sampled = B_sampled.unsqueeze(0).unsqueeze(0)


	# --- SET UP THE THETA GRID ---

	theta_grid = torch.linspace(0,2*np.pi,num_grid_points)
	theta_euclid_grid = torch.zeros((theta_grid.shape[0],2))
	theta_euclid_grid[:,0] = torch.cos(theta_grid)
	theta_euclid_grid[:,1] = torch.sin(theta_grid)
	theta_grid_reshaped = theta_grid.unsqueeze(1).unsqueeze(2).unsqueeze(0) # [1,num_cell_samples,num_gene_samples,num_genes]



	# --- COMPUTE THE PRIOR PROBABILITIES OVER THETA GRID: # [num_sampled_points x num_cells]---
	
	if prior_theta_euclid_dist is not None:
		# print("NEED TO CHANGE THIS TO HANDLE EUCLID GRID")
		if prior_theta_euclid_dist.__class__ == power_spherical.distributions.HypersphericalUniform or prior_theta_euclid_dist.__class__ == power_spherical.distributions.PowerSpherical:
			prior_theta_grid_prob = prior_theta_euclid_dist.log_prob(theta_euclid_grid.unsqueeze(1))
		elif prior_theta_euclid_dist.__class__ == cell_posterior.ThetaPosteriorDist:
			prior_theta_grid_prob = prior_theta_euclid_dist.log_prob(theta_grid.reshape(-1,1))
		else:
			raise Exception("Cell prior distribution choice not recognized.")





	# ---- COMPUTE THE LOG PROP: [num_cells, num_cell_samples, num_gene_samples, num_genes] ---

	if Q_sampled is None:
		log_prop_sampled = mu_sampled + A_sampled * torch.cos(theta_grid_reshaped - phi_sampled)
	else:
		log_prop_sampled = mu_sampled + (Q_sampled * A_sampled * torch.cos(theta_grid_reshaped - phi_sampled))
	if B_sampled is not None:
		log_prop_sampled += B_sampled


	# --- COMPUTE LOG LAMBDA SAMPLED [num_cells, num_cell_samples, num_gene_samples, num_genes] ---

	log_lambda_sampled = log_prop_sampled + log_L.unsqueeze(1).unsqueeze(2).unsqueeze(3)


	# --- COMPUTE THE LL OVER THE SAMPLED CELL-GENE GRID: [num_cells, num_cell_samples, num_gene_samples, num_genes] ---


	if use_nb:
		ll_cell_gene_grid_sampled = objective_functions.compute_nb_ll(gene_X.unsqueeze(1).unsqueeze(2),
			log_prop_sampled,
			log_lambda_sampled,
			log_mean_log_disp_coef)
	else:
		ll_cell_gene_grid_sampled = torch.distributions.poisson.Poisson(rate = torch.exp(log_lambda_sampled)).log_prob(gene_X.unsqueeze(1).unsqueeze(2))


	# --- COMPUTE LL OVER CELLS: [num_cells x num_cell_samples x num_gene_samples] ---
	ll_cell_grids = torch.sum(ll_cell_gene_grid_sampled,dim=3) # [num_cells x num_cell_samples x num_gene_samples]



	# --- RESHAPE prior_theta_grid_prob: [num_cells x num_cell_samples x 1] ---
	if prior_theta_euclid_dist is not None:
		prior_theta_grid_prob_reshaped = prior_theta_grid_prob.T.unsqueeze(2)



	# ---- COMPUTE THE PHASE POSTERIOR (AVERAGING THE INDIVIDAUL SAMPLE PHASE POSTERIORS) ----
	posterior_ll_cell_grid = ll_cell_grids
	if prior_theta_euclid_dist is not None:
		posterior_ll_cell_grid += prior_theta_grid_prob_reshaped
	posterior_ll_cell_grid_max_norm = posterior_ll_cell_grid - torch.max(posterior_ll_cell_grid,dim=1).values.unsqueeze(1)
	posterior_ll_cell_grid_max_norm = posterior_ll_cell_grid_max_norm.to(torch.float64)
	posterior_likelihood_cell_grid_max_norm = torch.exp(posterior_ll_cell_grid_max_norm)
	posterior_likelihoods_sampled = posterior_likelihood_cell_grid_max_norm / torch.sum(posterior_likelihood_cell_grid_max_norm,dim=1).unsqueeze(1)
	posterior_likelihood = torch.sum(posterior_likelihoods_sampled,dim=2) / float(num_gene_samples)

	return posterior_likelihood








