import pandas as pd
import power_spherical
import torch
import numpy as np

# tempo imports
from . import cell_posterior



# - Description: given current variational and prior parameters for the genes, returns a pandas dataframe
#	with their values
# - Parameters:
#		- gene_names (str): name of genes
#		- optimal_gene_param_dict (dict of Tensors): dictionary storing the current gene variational distribution parameters
#		- gene_prior_dict (dict of Tensors): dictionary storing the gene prior distribution parameters
#		- flat_model (bool): if false, does not include sinusoidal parameters in the dataframe
# - Outputs:
#		- gene_param_df (pandas dataframe): datafarme storing parameter values
def gene_param_dicts_to_param_df(gene_names, gene_param_dict, gene_prior_dict, min_amp, max_amp):


	# fill gene param df
	gene_param_df = pd.DataFrame()
	gene_param_df['gene'] = gene_names
	gene_param_df = gene_param_df.set_index("gene")



	# variational params
	if gene_param_dict is not None:
		gene_param_df['mu_loc'] = gene_param_dict['mu_loc'].detach().numpy()
		gene_param_df['mu_scale'] = gene_param_dict['mu_scale'].detach().numpy()
		if 'A_alpha' in gene_param_dict and 'A_beta' in gene_param_dict and 'phi_euclid_loc' in gene_param_dict and 'phi_scale' in gene_param_dict:
			gene_param_df['A_alpha'] = gene_param_dict['A_alpha'].detach().numpy()
			gene_param_df['A_beta'] = gene_param_dict['A_beta'].detach().numpy()
			gene_param_df['phi_euclid_cos'] = gene_param_dict['phi_euclid_loc'][:,0].detach().numpy()
			gene_param_df['phi_euclid_sin'] = gene_param_dict['phi_euclid_loc'][:,1].detach().numpy()
			gene_param_df['phi_scale'] = gene_param_dict['phi_scale'].detach().numpy()
			A_loc = gene_param_dict['A_alpha'] / (gene_param_dict['A_alpha'] + gene_param_dict['A_beta'])
			A_loc = (A_loc * (max_amp - min_amp)) + min_amp
			A_loc = A_loc.detach().numpy()
			phi_loc = torch.atan2(gene_param_dict['phi_euclid_loc'][:,1],gene_param_dict['phi_euclid_loc'][:,0]).detach().numpy()
			gene_param_df['A_loc'] = A_loc
			gene_param_df['phi_loc'] = phi_loc % (2 * np.pi)
			if 'Q_prob_alpha' in gene_param_dict and 'Q_prob_beta' in gene_param_dict:
				gene_param_df['Q_prob_alpha'] = gene_param_dict['Q_prob_alpha'].detach().numpy()
				gene_param_df['Q_prob_beta'] = gene_param_dict['Q_prob_beta'].detach().numpy()
				Q_prob_loc = gene_param_dict['Q_prob_alpha'] / (gene_param_dict['Q_prob_alpha'] + gene_param_dict['Q_prob_beta'])
				Q_prob_loc = Q_prob_loc.detach().numpy()
				gene_param_df['Q_prob_loc'] = Q_prob_loc


	# prior params
	if gene_prior_dict is not None:
		gene_param_df['prior_mu_loc'] = gene_prior_dict['prior_mu_loc'].detach().numpy()
		gene_param_df['prior_mu_scale'] = gene_prior_dict['prior_mu_scale'].detach().numpy()
		if 'prior_A_alpha' in gene_prior_dict and 'prior_A_beta' in gene_prior_dict and 'prior_phi_euclid_loc' in gene_prior_dict and 'prior_phi_scale' in gene_prior_dict:
			gene_param_df['prior_A_alpha'] = gene_prior_dict['prior_A_alpha'].detach().numpy()
			gene_param_df['prior_A_beta'] = gene_prior_dict['prior_A_beta'].detach().numpy()
			gene_param_df['prior_phi_euclid_cos'] = gene_prior_dict['prior_phi_euclid_loc'][:,0].detach().numpy()
			gene_param_df['prior_phi_euclid_sin'] = gene_prior_dict['prior_phi_euclid_loc'][:,1].detach().numpy()
			gene_param_df['prior_phi_scale'] = gene_prior_dict['prior_phi_scale'].detach().numpy()
			if 'prior_Q_prob_alpha' in gene_prior_dict and 'prior_Q_prob_beta' in gene_prior_dict:
				gene_param_df['prior_Q_prob_alpha'] = gene_prior_dict['prior_Q_prob_alpha'].detach().numpy()
				gene_param_df['prior_Q_prob_beta'] = gene_prior_dict['prior_Q_prob_beta'].detach().numpy()


	return gene_param_df
	



def cell_powerspherical_params_dict_to_param_df(cell_barcodes, cell_prior_dict):


	cell_prior_df = pd.DataFrame()
	cell_prior_df['barcode'] = cell_barcodes
	cell_prior_df = cell_prior_df.set_index("barcode")
	cell_prior_df['prior_uniform_angle'] = cell_prior_dict['prior_uniform_angle']
	if not cell_prior_dict['prior_uniform_angle']:
		cell_prior_df['prior_theta_euclid_cos'] = cell_prior_dict['prior_theta_euclid_loc'][:,0].detach().numpy()
		cell_prior_df['prior_theta_euclid_sin'] = cell_prior_dict['prior_theta_euclid_loc'][:,1].detach().numpy()
		cell_prior_df['prior_theta_scale'] = cell_prior_dict['prior_theta_scale'].detach().numpy()
	else:
		cell_prior_df['prior_theta_euclid_cos'] = None
		cell_prior_df['prior_theta_euclid_sin'] = None
		cell_prior_df['prior_theta_scale'] = None



	return cell_prior_df


def cell_multinomial_params_to_param_df(cell_barcodes, clock_theta_posterior_likelihood):

	cell_posterior_df = pd.DataFrame()
	cell_posterior_df['barcode'] = cell_barcodes
	cell_posterior_df = cell_posterior_df.set_index("barcode")
	num_bins = clock_theta_posterior_likelihood.shape[1]
	for bin_num in range(0,num_bins):
		col = 'bin_%s' % bin_num
		cell_posterior_df[col] = clock_theta_posterior_likelihood[:,bin_num].detach().numpy()


	return cell_posterior_df




