import sys
import os
import anndata
import copy
import torch
import pandas as pd
import numpy as np
import power_spherical
import hyperspherical_vae


# tempo imports
from . import objective_functions
from . import cell_posterior
from . import utils




# [num_gene_samples x num_cells x num_cell_samples x num_genes]
def compute_nonparametric_ll_mat(gene_X,
							log_L,
							phases_sampled,
							gene_log_alpha,
							gene_log_beta,
							gene_min_log_prop,
							gene_max_log_prop,
							num_gene_samples,
							use_nb,
							log_mean_log_disp_coef):
	
	
	# ** get num grid points from de_novo_log_alpha **
	num_grid_points = gene_log_alpha.shape[0]


	# ** discretize the cell phases **
	phases_discretized_indices = torch.round((phases_sampled / (2 * np.pi)) * num_grid_points) # convert [0,1] to [0,num_grid_points] (float), and then discretize
	phases_discretized_indices[torch.where(phases_discretized_indices == num_grid_points)] = 0 # since bins actually go from [0,num_grid_points - 1], let's wrap back around
	phases_discretized_indices = phases_discretized_indices.int().long() # [num_cells x num_cell_samples]


	# ** get the cell-gene alpha and beta's ** 
	cell_gene_alpha = torch.exp(gene_log_alpha)[phases_discretized_indices.long(),:] # [num_cells x num_cell_samples x num_genes]
	cell_gene_beta = torch.exp(gene_log_beta)[phases_discretized_indices.long(),:] # [num_cells x num_cell_samples x num_genes]

	# **  get the corresponding distribution
	cell_gene_dist = torch.distributions.beta.Beta(cell_gene_alpha,cell_gene_beta)

	# ** sample gene proportions: [num_gene_samples x num_cells x num_cell_samples x num_genes] ** 
	cell_gene_log_prop_sampled = cell_gene_dist.rsample((num_gene_samples,)) # [num_]
	cell_gene_log_prop_sampled = cell_gene_log_prop_sampled * (gene_max_log_prop.unsqueeze(0).unsqueeze(0) - gene_min_log_prop.unsqueeze(0).unsqueeze(0)) + gene_min_log_prop.unsqueeze(0).unsqueeze(0) # put in [min to max prop for each gene]


	# **  get the cell_gene_log_mean (expected log prop scaled by the library size): [num_gene_samples x num_cells x num_cell_samples x num_genes] ** 
	cell_gene_log_mean = cell_gene_log_prop_sampled + log_L.unsqueeze(0).unsqueeze(2).unsqueeze(2)
	cell_gene_mean = torch.exp(cell_gene_log_mean)

	#  ** compute the MC expected log likelihood for gene: # [num_gene_samples x num_cells x num_cell_samples x num_genes] ** 
	gene_X_reshaped = gene_X.unsqueeze(0).unsqueeze(2)
	if use_nb:
		ll_mat = objective_functions.compute_nb_ll(gene_X_reshaped,
		cell_gene_log_prop_sampled,
		cell_gene_log_mean,
		log_mean_log_disp_coef)
	else:
		ll_mat = torch.distributions.poisson.Poisson(cell_gene_mean).log_prob(gene_X_reshaped)
		
	return ll_mat
	


# inputs:
#	- mu_sampled ([num_gene_samples x num_genes]) 
#	- A_sampled ([num_gene_samples x num_genes]) 
#	- phi_sampled ([num_gene_samples x num_genes]) 
#	- Q_sampled ([num_gene_samples x num_genes]) 
#	- phases_sampled ([num_cells x num_cell_samples])

# output:
#	- [num_gene_samples x num_cells x num_cell_samples x num_genes]
def compute_clock_ll_mat(clock_X,
						log_L,
						use_nb,
						log_mean_log_disp_coef,
						clock_min_amp,
						clock_max_amp,
						mu_sampled = None,
						A_sampled = None,
						phi_sampled = None,
						Q_sampled = None,
						phases_sampled = None):
	

	
	# **  get corresponding samples for log prop's: [num_gene_samples x num_cells x num_cell_samples x num_genes] ** 
	if Q_sampled is None:
		clock_log_prop_sampled = mu_sampled + (A_sampled * torch.cos(phases_sampled - phi_sampled))
	else:
		clock_log_prop_sampled = mu_sampled + (Q_sampled * A_sampled * torch.cos(phases_sampled - phi_sampled))

	
	#  ** get corresponding samples for log means: [num_gene_samples x num_cells x num_cell_samples x num_genes] ** 
	clock_log_mean_sampled = clock_log_prop_sampled + log_L.unsqueeze(0).unsqueeze(2).unsqueeze(3)


	# ** compute the LL **
	if use_nb:
		ll_mat = objective_functions.compute_nb_ll(clock_X.unsqueeze(0).unsqueeze(2),
			clock_log_prop_sampled,
			clock_log_mean_sampled,
			log_mean_log_disp_coef)

	else:
		ll_mat = torch.distributions.poisson.Poisson(rate = torch.exp(clock_log_mean_sampled)).log_prob(clock_X.unsqueeze(0).unsqueeze(2))
	
	return ll_mat





def grid_sample_posterior_cell_phase(log_L,
	prior_theta_euclid_dist,
	num_grid_points = 12,
	clock_X = None,
	clock_param_dict = None,
	clock_min_amp = None,
	clock_max_amp = None,
	clock_mu_sampled = None,
	clock_A_sampled = None,
	clock_phi_sampled = None,
	clock_Q_sampled = None,
	de_novo_X = None,
	de_novo_log_alpha = None,
	de_novo_log_beta = None,
	de_novo_min_log_prop = None,
	de_novo_max_log_prop = None,                             
	use_nb = False,
	log_mean_log_disp_coef = None
	):

	
	# --- IF CHECK IF WE ARE USING THE CLOCK AND/OR THE DE NOVO IN THE LIKELIHOOD ---
	use_clock_ll = (clock_X is not None) & (clock_mu_sampled is not None) & (clock_A_sampled is not None) & (clock_phi_sampled is not None) & (clock_min_amp is not None) & (clock_max_amp is not None)
	use_de_novo_ll = (de_novo_X is not None) & (de_novo_log_alpha is not None) & (de_novo_log_beta is not None) & (de_novo_min_log_prop is not None) & (de_novo_max_log_prop is not None)
	if not use_clock_ll and not use_de_novo_ll:
		raise Exception("Error: neither clock nor de novo genes given as input for grid sampling.")
	

	# --- GET THE PHASE GRID: [1 x num_grid_points] ----
	phase_grid = torch.arange(0,2*np.pi,(2 * np.pi) / num_grid_points)
	phase_grid_euclid = torch.zeros((num_grid_points,2))
	phase_grid_euclid[:,0] = torch.cos(phase_grid)
	phase_grid_euclid[:,1] = torch.sin(phase_grid)


	

	# --- GET THE LL MAT FOR THE DE NOVO AND CLOCK GENES ---- **

	# [num_gene_samples x num_cells x num_cell_samples x num_genes]
	if use_de_novo_ll:
		de_novo_ll_mat = compute_nonparametric_ll_mat(gene_X = de_novo_X,
									log_L = log_L,
									phases_sampled = phase_grid.unsqueeze(0),
									gene_log_alpha = de_novo_log_alpha,
									gene_log_beta = de_novo_log_beta,
									gene_min_log_prop = de_novo_min_log_prop,
									gene_max_log_prop = de_novo_max_log_prop,
									num_gene_samples = num_gene_samples,
									use_nb = use_nb,
									log_mean_log_disp_coef = log_mean_log_disp_coef)

	# [num_gene_samples x num_cells x num_cell_samples x num_genes]
	if use_clock_ll:
		clock_ll_mat = compute_clock_ll_mat(clock_X = clock_X,
								log_L = log_L,
								use_nb = use_nb,
								log_mean_log_disp_coef = log_mean_log_disp_coef,
								clock_min_amp = clock_min_amp,
								clock_max_amp = clock_max_amp,
								mu_sampled = clock_mu_sampled,
								A_sampled = clock_A_sampled,
								phi_sampled = clock_phi_sampled,
								Q_sampled = clock_Q_sampled,
								phases_sampled = phase_grid.unsqueeze(0))




	# --- CONCATENATE THE CLOCK AND DE NOVO LIKELIHOODS IF WE NEED TO ---
	
	if use_de_novo_ll and use_clock_ll:
		# [num_gene_samples x num_cells x num_cell_samples x (num_clock_genes + num_de_novo_genes)]
		ll_cell_gene_grid_sampled = torch.cat((clock_ll_mat,de_novo_ll_mat), dim=3)
	elif use_de_novo_ll:
		# [num_gene_samples x num_cells x num_cell_samples x num_de_novo_genes]
		ll_cell_gene_grid_sampled = de_novo_ll_mat
	else:
		# [num_gene_samples x num_cells x num_cell_samples x num_clock_genes]
		ll_cell_gene_grid_sampled = clock_ll_mat


	# --- COMPUTE THE PHASE LIKELIHOODS: [num_cells x num_grid_points]---

	phase_ll_un_norm = torch.mean(torch.sum(ll_cell_gene_grid_sampled,dim=3),dim=0) # [num_cells x num_grid_points]
	phase_ll_max_norm = phase_ll_un_norm - torch.max(phase_ll_un_norm,axis=1).values.unsqueeze(1)
	phase_ll_max_norm = phase_ll_max_norm.to(torch.float64)
	phase_likelihood_max_norm = torch.exp(phase_ll_max_norm)
	phase_likelihood = phase_likelihood_max_norm / torch.sum(phase_likelihood_max_norm,axis=1).unsqueeze(1)
	phase_ll = torch.log(phase_likelihood)


	# --- COMPUTE THE PRIOR PROBABILITIES OVER THETA GRID: # [1 x num_grid_points] or [num_cells x num_grid_points]---

	if prior_theta_euclid_dist is not None:
		if prior_theta_euclid_dist.__class__ == power_spherical.distributions.HypersphericalUniform:
			prior_theta_grid_ll_un_norm = prior_theta_euclid_dist.log_prob(phase_grid_euclid).unsqueeze(0) #[1 x num_grid_points]     
		elif prior_theta_euclid_dist.__class__ == hyperspherical_vae.distributions.von_mises_fisher.VonMisesFisher:
			prior_theta_grid_ll_un_norm = prior_theta_euclid_dist.log_prob(phase_grid_euclid.unsqueeze(1)).T #[num_cells x num_grid_points]    
		elif prior_theta_euclid_dist.__class__ == cell_posterior.ThetaPosteriorDist:
			prior_theta_grid_ll_un_norm = prior_theta_euclid_dist.log_prob(phase_grid.unsqueeze(1)).T #[num_cells x num_grid_points]            
		else:
			raise Exception("Cell prior distribution choice not recognized.")
	prior_theta_grid_ll_max_norm = prior_theta_grid_ll_un_norm - torch.max(prior_theta_grid_ll_un_norm,dim=1).values.unsqueeze(1)
	prior_theta_grid_ll_max_norm = prior_theta_grid_ll_max_norm.to(torch.float64)
	prior_theta_grid_likelihood_max_norm = torch.exp(prior_theta_grid_ll_max_norm)
	prior_theta_grid_likelihood = prior_theta_grid_likelihood_max_norm / torch.sum(prior_theta_grid_likelihood_max_norm,dim=1).unsqueeze(1)
	prior_phase_ll = torch.log(prior_theta_grid_likelihood)



	# ---- COMPUTE THE PHASE POSTERIOR (AVERAGING THE INDIVIDUAL SAMPLE PHASE POSTERIORS) ----
	posterior_phase_ll_un_norm = phase_ll + prior_phase_ll
	posterior_phase_ll_max_norm = posterior_phase_ll_un_norm - torch.max(posterior_phase_ll_un_norm,axis=1).values.unsqueeze(1)
	posterior_phase_ll_max_norm = posterior_phase_ll_max_norm.to(torch.float64)
	posterior_phase_likelihood_max_norm = torch.exp(posterior_phase_ll_max_norm)
	posterior_phase_likelihood = posterior_phase_likelihood_max_norm / torch.sum(posterior_phase_likelihood_max_norm,axis=1).unsqueeze(1)




	return posterior_phase_likelihood, phase_likelihood, prior_theta_grid_likelihood










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

	theta_grid = torch.arange(0,2*np.pi,(2 * np.pi) / num_grid_points) # theta_grid = torch.linspace(0,2*np.pi,num_grid_points)
	theta_euclid_grid = torch.zeros((theta_grid.shape[0],2))
	theta_euclid_grid[:,0] = torch.cos(theta_grid)
	theta_euclid_grid[:,1] = torch.sin(theta_grid)
	theta_grid_reshaped = theta_grid.unsqueeze(1).unsqueeze(2).unsqueeze(0) # [1,num_cell_samples,num_gene_samples,num_genes]



	# --- COMPUTE THE PRIOR PROBABILITIES OVER THETA GRID: # [num_sampled_points x num_cells]---
	
	if prior_theta_euclid_dist is not None:
		# print("NEED TO CHANGE THIS TO HANDLE EUCLID GRID")
		if prior_theta_euclid_dist.__class__ == power_spherical.distributions.HypersphericalUniform or prior_theta_euclid_dist.__class__ == hyperspherical_vae.distributions.von_mises_fisher.VonMisesFisher:
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








