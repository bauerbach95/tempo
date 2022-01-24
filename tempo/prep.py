import torch
import numpy as np
import scipy
from scipy import stats
import pandas as pd
import scanpy
import anndata



# tempo imports
from . import utils



# --- PARAMETER FUNCTIONS TO GET VARIATIONAL AND PRIOR DISTRIBUTIONS ---

# ** mesor **

# variational
def get_mesor_variational_params(adata, init_mesor_scale_val = 0.1):

	# ** check if mesor_loc and mesor_scale found in adata.var; otherwise init using prop **
	if "mu_loc" in adata.var.columns and "mu_scale" in adata.var.columns:
		mesor_loc = np.array(adata.var['mu_loc'])
		mesor_scale = np.array(adata.var['mu_scale'])
	else:
		mesor_loc = np.log(np.array(adata.var['prop']))
		mesor_scale = np.array([init_mesor_scale_val] * adata.shape[1])


	# ** mesor log scale **
	mesor_log_scale = np.log(mesor_scale)

	# ** init variational pytorch parameters **
	mu_loc = torch.nn.Parameter(torch.Tensor(mesor_loc), requires_grad = True)
	mu_log_scale = torch.nn.Parameter(torch.Tensor(mesor_log_scale), requires_grad = True)

	return mu_loc, mu_log_scale

# prior
def get_mesor_prior_params(adata, prior_mesor_scale_val = 0.5):

	# ** check if mesor_loc and mesor_scale found in adata.var; otherwise init using prop **
	if "prior_mu_loc" in adata.var.columns and "prior_mu_scale" in adata.var.columns:
		prior_mesor_loc = np.array(adata.var['prior_mu_loc'])
		prior_mesor_scale = np.array(adata.var['prior_mu_scale'])
	else:
		prior_mesor_loc = np.log(np.array(adata.var['prop']))
		prior_mesor_scale = np.array([prior_mesor_scale_val] * adata.shape[1])

	# ** torch tensors **
	prior_mesor_loc = torch.Tensor(prior_mesor_loc)
	prior_mesor_scale = torch.Tensor(prior_mesor_scale)

	return prior_mesor_loc, prior_mesor_scale

# variational and prior
def get_mesor_variational_and_prior_params(adata, init_mesor_scale_val = 0.1, prior_mesor_scale_val = 0.5):

	# ** init variational pytorch parameters **
	mu_loc, mu_log_scale = get_mesor_variational_params(adata, init_mesor_scale_val = init_mesor_scale_val)
	
	# ** init prior pytorch distributions **
	prior_mesor_loc, prior_mesor_scale = get_mesor_prior_params(adata, prior_mesor_scale_val = prior_mesor_scale_val)

	return mu_loc, mu_log_scale, prior_mesor_loc, prior_mesor_scale


# ** amplitude **

# variational
def get_amp_variational_params(adata, max_amp = 1.0 / np.log10(np.e), min_amp = 0.2 / np.log10(np.e), init_amp_loc_val = 0.4 / np.log10(np.e), init_amp_scale_val = 400):


	# ** check if A_alpha and A_beta are supplied **
	if 'A_alpha' in adata.var.columns and 'A_beta' in adata.var.columns:
		amp_log_alpha = np.log(np.array(adata.var['A_alpha']))
		amp_log_beta = np.log(np.array(adata.var['A_beta']))
		A_alpha = torch.nn.Parameter(torch.Tensor(amp_log_alpha), requires_grad = True)
		A_beta = torch.nn.Parameter(torch.Tensor(amp_log_beta), requires_grad = True)
		return A_alpha, A_beta


	# ** raise exception if amp variational loc is not greater than min **
	if not (init_amp_loc_val > min_amp and init_amp_loc_val < max_amp):
		raise Exception("Error: init amp loc must be within the (min_amp,max_amp) range -- exclusive.")


	# ** compute z value ([0,1] which refers to [min_amp, max_amp]) of init_amp_loc_val **
	z_val = float((init_amp_loc_val - min_amp) / (max_amp - min_amp))


	# ** init variational params as numpy arrays **
	amp_alpha = np.array([1.0] * adata.shape[1]) # unscaled alpha and beta
	amp_beta = np.array([(1.0 / z_val) - 1.0] * adata.shape[1]) # beta = (1 - z) / z = (1 / z) - 1
	amp_alpha = amp_alpha * init_amp_scale_val # scaled alpha and beta
	amp_beta = amp_beta * init_amp_scale_val


	# ** convert amp alpha and beta to log values **
	amp_log_alpha = np.log(amp_alpha)
	amp_log_beta = np.log(amp_beta)

	# ** init variational pytorch parameters **
	A_log_alpha = torch.nn.Parameter(torch.Tensor(amp_log_alpha), requires_grad = True)
	A_log_beta = torch.nn.Parameter(torch.Tensor(amp_log_beta), requires_grad = True)

	return A_log_alpha, A_log_beta



# prior
def get_amp_prior_params(adata, prior_amp_alpha_val = 1.0, prior_amp_beta_val = 1.0):

	# ** check if A_alpha and A_beta are supplied **
	if 'prior_A_alpha' in adata.var.columns and 'prior_A_beta' in adata.var.columns:
		prior_amp_alpha = np.array(adata.var['prior_A_alpha'])
		prior_amp_beta = np.array(adata.var['prior_A_beta'])
		prior_A_alpha = torch.Tensor(prior_amp_alpha)
		prior_A_beta = torch.Tensor(prior_amp_beta)
		return prior_A_alpha, prior_A_beta


	# ** set prior params as numpy arrays **
	prior_amp_alpha = np.array([prior_amp_alpha_val] * adata.shape[1])
	prior_amp_beta = np.array([prior_amp_beta_val] * adata.shape[1])

	# ** make torch tensors **
	prior_A_alpha = torch.Tensor(prior_amp_alpha)
	prior_A_beta = torch.Tensor(prior_amp_beta)


	return prior_A_alpha, prior_A_beta

# variational and prior
def get_amp_variational_and_prior_params(adata, init_amp_loc_val = 0.4 / np.log10(np.e), init_amp_scale_val = 400,
	max_amp = 1.0 / np.log10(np.e), min_amp = 0.2 / np.log10(np.e), prior_amp_alpha_val = 1.0, prior_amp_beta_val = 1.0):

	# ** variational params as pytorch tensors **
	A_log_alpha, A_log_beta = get_amp_variational_params(adata, max_amp = max_amp, min_amp = min_amp, init_amp_loc_val = init_amp_loc_val, init_amp_scale_val = init_amp_scale_val)

	# ** prior dist **
	prior_amp_alpha, prior_amp_beta = get_amp_prior_params(adata, prior_amp_alpha_val = prior_amp_alpha_val, prior_amp_beta_val = prior_amp_beta_val)


	return A_log_alpha, A_log_beta, prior_amp_alpha, prior_amp_beta


# ** shift **

# variational
def get_variational_shift_params(adata,
	bulk_shift_df,
	known_cycler_init_shift_95_interval = (1.0 / 12.0) * np.pi,
	unknown_cycler_init_shift_95_interval = (3.0 / 12.0) * np.pi,
	reference_gene = 'Arntl',
	shift_noise_scale = 0.3):
	# shift_noise_scale was: 0.5 before


	# ** check if phi_euclid_cos, phi_euclid_sin, phi_scale are supplied **
	if ('phi_euclid_cos' in adata.var.columns) and ('phi_euclid_sin' in adata.var.columns) and ('phi_scale' in adata.var.columns):
		phi_euclid_loc = np.zeros((adata.shape[1],2))
		phi_euclid_loc[:,0] = np.array(adata.var['phi_euclid_cos'])
		phi_euclid_loc[:,1] = np.array(adata.var['phi_euclid_sin'])
		phi_log_scale = np.log(np.array(adata.var['phi_scale']))

		# get parameters
		phi_euclid_loc = torch.nn.Parameter(torch.Tensor(phi_euclid_loc), requires_grad = True)
		phi_log_scale = torch.nn.Parameter(torch.Tensor(phi_log_scale), requires_grad = True)
		return phi_euclid_loc, phi_log_scale



	# --- CONVERT 95 INTERVALS TO CONCENTRATIONS ---
	known_cycler_shift_concentration = utils.powerspherical_95_radian_interval_to_concentration(known_cycler_init_shift_95_interval)
	unknown_cycler_shift_concentration = utils.powerspherical_95_radian_interval_to_concentration(unknown_cycler_init_shift_95_interval)


	# --- SET SHIFT VARIATIONAL AND PRIOR DISTRIBUTION PARAMETERS AS NUMPY ARRAYS ----

	# ** init variational params as numpy arrays **

	# init variational shift locs and scales to the uninformative case
	shift_loc = np.zeros((adata.shape[1]))
	shift_concentration = np.ones((adata.shape[1])) * unknown_cycler_shift_concentration

	# update shift locs to the mean value in the reference if found
	if bulk_shift_df is not None:
		for i, gene in enumerate(list(adata.var_names)):
			if gene in bulk_shift_df.index:
				shift_loc[i] = bulk_shift_df.loc[gene]['prior_acrophase_loc']
				shift_concentration[i] = known_cycler_shift_concentration
			

			
	# for reference gene only -- update variational concentration to pre-specified high value (basically a point) and set loc to 0 radians
	if reference_gene is not None and reference_gene in adata.var_names:
		reference_gene_index = np.where(adata.var_names == reference_gene)[0]
		shift_concentration[reference_gene_index] = 10000.0
		shift_loc[reference_gene_index] = 0.0



	# --- OTHER THAN THE REFERENCE GENE, IF SHIFT LOC IS EQUAL TO ZERO, LET'S ADD SOME NOISE TO IT ---
	zero_loc_indices = np.where(shift_loc == 0)[0]
	if reference_gene is not None:
		reference_gene_index = np.where(adata.var_names == reference_gene)[0]
		zero_loc_indices = np.setdiff1d(zero_loc_indices, np.array([reference_gene_index]))
	shift_loc[zero_loc_indices] = shift_loc[zero_loc_indices] + np.random.normal(size = zero_loc_indices.size) * shift_noise_scale



	# --- INIT VARIATIONAL PYTORCH PARAMETERS ---

	# set tensors
	phi_loc = torch.Tensor(shift_loc)
	phi_euclid_loc = torch.zeros((phi_loc.shape[0], 2))
	phi_euclid_loc[:,0] = torch.cos(phi_loc)
	phi_euclid_loc[:,1] = torch.sin(phi_loc)
	phi_scale = torch.Tensor(shift_concentration)
	phi_log_scale = torch.log(phi_scale)

	# get parameters
	phi_euclid_loc = torch.nn.Parameter(torch.Tensor(phi_euclid_loc), requires_grad = True)
	phi_log_scale = torch.nn.Parameter(torch.Tensor(phi_log_scale), requires_grad = True)


	return phi_euclid_loc, phi_log_scale

# prior
def get_prior_shift_params(adata,
	bulk_shift_df,
	known_cycler_prior_shift_95_interval = (2.0 / 12.0) * np.pi,
	unknown_cycler_prior_shift_95_interval = (11.99 / 12.0) * np.pi,
	reference_gene = 'Arntl'):


	# ** check if phi_euclid_cos, phi_euclid_sin, phi_scale are supplied **
	if ('prior_phi_euclid_cos' in adata.var.columns) and ('prior_phi_euclid_sin' in adata.var.columns) and ('prior_phi_scale' in adata.var.columns):
		prior_phi_euclid_loc = np.zeros((adata.shape[1],2))
		prior_phi_euclid_loc[:,0] = np.array(adata.var['prior_phi_euclid_cos'])
		prior_phi_euclid_loc[:,1] = np.array(adata.var['prior_phi_euclid_sin'])
		prior_phi_scale = np.array(adata.var['prior_phi_scale']).reshape(-1,1)

		# get parameters
		prior_phi_euclid_loc = torch.Tensor(prior_phi_euclid_loc)
		prior_phi_scale = torch.Tensor(prior_phi_scale)
		return prior_phi_euclid_loc, prior_phi_scale




	# --- GET THE SHIFT CONCENTRATIONS FROM THE INTERVALS ---
	known_cycler_shift_concentration = utils.convert_ps_to_vmf_scale(utils.powerspherical_95_radian_interval_to_concentration(known_cycler_prior_shift_95_interval))
	unknown_cycler_shift_concentration = utils.convert_ps_to_vmf_scale(utils.powerspherical_95_radian_interval_to_concentration(unknown_cycler_prior_shift_95_interval))




	# --- SET SHIFT VARIATIONAL AND PRIOR DISTRIBUTION PARAMETERS AS NUMPY ARRAYS ----

	# ** init variational params as numpy arrays **

	# init variational shift locs and scales to the uninformative case
	shift_loc = np.zeros((adata.shape[1]))
	shift_concentration = np.ones((adata.shape[1])) * unknown_cycler_shift_concentration

	# update shift locs to the val in the reference if found
	if bulk_shift_df is not None:
		for i, gene in enumerate(list(adata.var_names)):
			if gene in bulk_shift_df.index:
				shift_loc[i] = bulk_shift_df.loc[gene]['prior_acrophase_loc']
				if 'prior_acrophase_95_interval' in bulk_shift_df:
					shift_concentration[i] = utils.convert_ps_to_vmf_scale(utils.powerspherical_95_radian_interval_to_concentration(bulk_shift_df.loc[gene]['prior_acrophase_95_interval']))
				else:
					shift_concentration[i] = known_cycler_shift_concentration
			

			
	# for reference gene only -- update variational concentration to pre-specified high value (basically a point) and set loc to 0 radians
	if reference_gene is not None and reference_gene in adata.var_names:
		reference_gene_index = np.where(adata.var_names == reference_gene)[0]
		shift_concentration[reference_gene_index] = 10000.0
		shift_loc[reference_gene_index] = 0.0



	# --- INIT PYTORCH TENSORS ---

	# set tensors
	phi_loc = torch.Tensor(shift_loc)
	phi_euclid_loc = torch.zeros((phi_loc.shape[0], 2))
	phi_euclid_loc[:,0] = torch.cos(phi_loc)
	phi_euclid_loc[:,1] = torch.sin(phi_loc)
	phi_scale = torch.Tensor(shift_concentration).reshape(-1,1)


	return phi_euclid_loc, phi_scale


def load_bulk_shift_reference(bulk_cycler_info_path, reference_gene = None):



	# --- LOAD BULK CCG REFERENCE ---
	bulk_shift_df = pd.read_table(bulk_cycler_info_path,sep=',')
	bulk_shift_df = bulk_shift_df.set_index("gene")



	# --- CHECK THAT REFERENCE GENE IN BULK REFERENCE AND UPDATE GENE SHIFTS S.T. REFERENCE GENE SHIFT = 0 ---

	if reference_gene not in bulk_shift_df.index:
		print("Warning: reference gene chosen not found to be a cycler in reference bulk datasets.")

	# ** update mean shifts in bulk reference w.r.t. the reference gene if it is found in the bulk reference **
	if (reference_gene is not None) and (reference_gene in bulk_shift_df.index):

		# ** update gene shifts s.t. reference gene has a shift of 0 by default **

		# shift in terms of the reference
		mean_reference_shift = np.array(bulk_shift_df['prior_acrophase_loc'] - bulk_shift_df['prior_acrophase_loc'].loc[reference_gene])

		# restrict [0,2*np.pi]
		mean_reference_shift[np.where(mean_reference_shift < 0)] = mean_reference_shift[np.where(mean_reference_shift < 0)] + 2 * np.pi
		bulk_shift_df['prior_acrophase_loc'] = mean_reference_shift

	else:

		# set reference shift to be the same as the mean_shift in the bulk reference
		bulk_shift_df['prior_acrophase_loc'] = bulk_shift_df['prior_acrophase_loc']


	return bulk_shift_df




# variational and prior
def get_shift_variational_and_prior_params(adata,
	known_cycler_init_shift_95_interval = (1.0 / 12.0) * np.pi,
	unknown_cycler_init_shift_95_interval = (1.0 / 12.0) * np.pi,
	known_cycler_prior_shift_95_interval = (2.0 / 12.0) * np.pi,
	unknown_cycler_prior_shift_95_interval = (11.99 / 12.0) * np.pi,
	reference_gene = 'Arntl',
	bulk_cycler_info_path = '/Users/benauerbach/Desktop/tempo/utils/BHTC_cyclers.csv'):


	# ** load the bulk shift reference **
	bulk_shift_df = None
	if bulk_cycler_info_path is not None:
		bulk_shift_df = load_bulk_shift_reference(bulk_cycler_info_path, reference_gene)



	# ** variational params as pytorch tensors **
	phi_euclid_loc, phi_log_scale = get_variational_shift_params(adata,
		known_cycler_init_shift_95_interval = known_cycler_init_shift_95_interval,
		unknown_cycler_init_shift_95_interval = unknown_cycler_init_shift_95_interval,
		reference_gene = reference_gene,
		bulk_shift_df = bulk_shift_df)



	# ** prior dist **
	prior_phi_euclid_loc, prior_phi_scale = get_prior_shift_params(adata,
		known_cycler_prior_shift_95_interval = known_cycler_prior_shift_95_interval,
		unknown_cycler_prior_shift_95_interval = unknown_cycler_prior_shift_95_interval,
		reference_gene = reference_gene,
		bulk_shift_df = bulk_shift_df)


	return phi_euclid_loc, phi_log_scale, prior_phi_euclid_loc, prior_phi_scale




def get_phase_variational_params(adata = None):


	if not ('theta_euclid_cos' in adata.obs.columns) and ('theta_euclid_sin' in adata.obs.columns) and ('theta_95_interval' in adata.obs.columns):
		raise Exception("Error: called get_phaes_variational_params but proper columns not found in adata")

	theta_euclid_loc = np.zeros((adata.shape[0],2))
	theta_euclid_loc[:,0] = np.array(adata.obs['theta_euclid_cos'])
	theta_euclid_loc[:,1] = np.array(adata.obs['theta_euclid_sin'])
	theta_scale = utils.powerspherical_95_radian_interval_to_concentration(np.array(adata.obs['theta_95_interval']))
	theta_euclid_loc = torch.Tensor(theta_euclid_loc)
	theta_scale = torch.Tensor(theta_scale)


	return theta_euclid_loc, theta_scale




# prior 
def get_phase_prior_params(adata = None, use_noninformative_phase_prior = True):

	
	if use_noninformative_phase_prior:
		uniform_angle, prior_theta_euclid_loc, prior_theta_scale = True, None, None
	elif ('prior_theta_euclid_cos' in adata.obs.columns) and ('prior_theta_euclid_sin' in adata.obs.columns) and ('prior_theta_95_interval' in adata.obs.columns):
		prior_theta_euclid_loc = np.zeros((adata.shape[0],2))
		prior_theta_euclid_loc[:,0] = np.array(adata.obs['prior_theta_euclid_cos'])
		prior_theta_euclid_loc[:,1] = np.array(adata.obs['prior_theta_euclid_sin'])
		prior_theta_scale = utils.convert_ps_to_vmf_scale(utils.powerspherical_95_radian_interval_to_concentration(np.array(adata.obs['prior_theta_95_interval'])))
		prior_theta_euclid_loc = torch.Tensor(prior_theta_euclid_loc)
		prior_theta_scale = torch.Tensor(prior_theta_scale).reshape(-1,1)
		uniform_angle = False
	else:
		print("use_noninformative_phase_prior = False, but improper cell phase prior columns specified in adata.")



	return uniform_angle, prior_theta_euclid_loc, prior_theta_scale



def get_cycler_prob_variational_and_prior_params(adata, init_clock_Q_prob_alpha, init_clock_Q_prob_beta, init_non_clock_Q_prob_alpha, init_non_clock_Q_prob_beta,
	prior_clock_Q_prob_alpha, prior_clock_Q_prob_beta, prior_non_clock_Q_prob_alpha, prior_non_clock_Q_prob_beta):


	# ** Q prob alpha and beta **
	if 'Q_prob_alpha' in adata.var and 'Q_prob_beta' in adata.var:
		Q_prob_alpha = torch.Tensor(np.array(adata.var['Q_prob_alpha']))
		Q_prob_beta = torch.Tensor(np.array(adata.var['Q_prob_beta']))


	else:

		# get the Q_prob_alpha and Q_prob_beta tensors
		Q_prob_alpha = torch.Tensor(np.array([init_non_clock_Q_prob_alpha] * adata.shape[1]))
		Q_prob_beta = torch.Tensor(np.array([init_non_clock_Q_prob_beta] * adata.shape[1]))


		# add update Q prob alpha and beta (and priors) for clock genes
		if "is_clock" in adata.var:
			clock_indices = np.where(adata.var['is_clock'])[0]
			Q_prob_alpha[clock_indices] = init_clock_Q_prob_alpha
			Q_prob_beta[clock_indices] = init_clock_Q_prob_beta



	# ** prior: Q prob alpha and beta **
	if 'prior_Q_prob_alpha' in adata.var and 'prior_Q_prob_beta' in adata.var:
		prior_Q_prob_alpha = torch.Tensor(np.array(adata.var['prior_Q_prob_alpha']))
		prior_Q_prob_beta = torch.Tensor(np.array(adata.var['prior_Q_prob_beta']))
	else:

		# get the prior_Q_prob_alpha and prior_Q_prob_beta tensors
		prior_Q_prob_alpha = torch.Tensor(np.array([prior_non_clock_Q_prob_alpha] * adata.shape[1]))
		prior_Q_prob_beta = torch.Tensor(np.array([prior_non_clock_Q_prob_beta] * adata.shape[1]))


		# add update Q prob alpha and beta (and priors) for clock genes
		if "is_clock" in adata.var:
			clock_indices = np.where(adata.var['is_clock'])[0]
			prior_Q_prob_alpha[clock_indices] = prior_clock_Q_prob_alpha
			prior_Q_prob_beta[clock_indices] = prior_clock_Q_prob_beta




	# ** get the variational parameters **
	Q_prob_log_alpha = torch.nn.Parameter(torch.log(Q_prob_alpha).detach(),requires_grad=True)
	Q_prob_log_beta = torch.nn.Parameter(torch.log(Q_prob_beta).detach(),requires_grad=True)


	return Q_prob_log_alpha, Q_prob_log_beta, prior_Q_prob_alpha, prior_Q_prob_beta










# --- PREP FUNCTION ---


def unsupervised_prep(adata,
	bulk_cycler_info_path,
	core_clock_gene_path,
	reference_gene = 'Arntl',
	min_amp = 0.0,
	max_amp = 1.5,
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
	non_clock_Q_prob_alpha = 1,
	non_clock_Q_prob_beta = 9,
	use_noninformative_phase_prior = True,
	min_gene_prop = 1e-5,
	**kwargs):



	# --- SET IS_CLOCK = TRUE FOR GENES IN CORE_CLOCK_GENE_PATH ---
	with open(core_clock_gene_path) as file_obj:
		clock_genes = list(map(lambda x: x.replace('\n', ''), file_obj.readlines()))
		clock_genes = list(filter(lambda x: x in adata.var_names, clock_genes))
	adata.var['is_clock'] = False
	adata.var.loc[clock_genes,'is_clock'] = True




	# --- GET MESOR VARIATIONAL PARAMETERS AND PRIOR DISTRIBUTION ---
	mu_loc, mu_log_scale, prior_mu_loc, prior_mu_scale = get_mesor_variational_and_prior_params(adata = adata,
		init_mesor_scale_val = init_mesor_scale_val, 
		prior_mesor_scale_val = prior_mesor_scale_val)


	# --- GET AMP VARIATIONAL PARAMETERS AND PRIOR DISTRIBUTION ---
	A_log_alpha, A_log_beta, prior_A_alpha, prior_A_beta = get_amp_variational_and_prior_params(
		adata = adata,
		init_amp_loc_val = init_amp_loc_val, 
		init_amp_scale_val = init_amp_scale_val,
		max_amp = max_amp,
		min_amp = min_amp,
		prior_amp_alpha_val = prior_amp_alpha_val, 
		prior_amp_beta_val = prior_amp_beta_val)


	# --- GET SHIFT VARIATIONAL PARAMETERS AND PRIOR DISTRIBUTION ---
	phi_euclid_loc, phi_log_scale, prior_phi_euclid_loc, prior_phi_scale = get_shift_variational_and_prior_params(adata = adata,
		known_cycler_init_shift_95_interval = known_cycler_init_shift_95_interval,
		unknown_cycler_init_shift_95_interval = unknown_cycler_init_shift_95_interval,
		known_cycler_prior_shift_95_interval = known_cycler_prior_shift_95_interval,
		reference_gene = reference_gene,
		bulk_cycler_info_path = bulk_cycler_info_path)




	# --- GET PHASE VARIATIONAL PARAMETERS AND PRIOR DISTRIBUTION ---
	prior_uniform_angle, prior_theta_euclid_loc, prior_theta_scale = get_phase_prior_params(adata = adata, use_noninformative_phase_prior = use_noninformative_phase_prior)



	# --- Q PARAMETERS ---
	Q_prob_log_alpha, Q_prob_log_beta, prior_Q_prob_alpha, prior_Q_prob_beta = get_cycler_prob_variational_and_prior_params(adata = adata,
		init_clock_Q_prob_alpha = init_clock_Q_prob_alpha,
		init_clock_Q_prob_beta = init_clock_Q_prob_beta,
		init_non_clock_Q_prob_alpha = init_non_clock_Q_prob_alpha,
		init_non_clock_Q_prob_beta = init_non_clock_Q_prob_beta,
		prior_clock_Q_prob_alpha = prior_clock_Q_prob_alpha,
		prior_clock_Q_prob_beta = prior_clock_Q_prob_beta,
		prior_non_clock_Q_prob_alpha = prior_non_clock_Q_prob_alpha, 
		prior_non_clock_Q_prob_beta = prior_non_clock_Q_prob_beta)

	



	# --- MAKE THE PARAMETER AND PRIOR DICTS ---


	# ** parameter dictionaries **

	# gene
	gene_param_dict = {
		"mu_loc" : mu_loc,
		"mu_log_scale" : mu_log_scale,
		"A_log_alpha" : A_log_alpha,
		"A_log_beta" : A_log_beta,
		"phi_euclid_loc" : phi_euclid_loc,
		"phi_log_scale" : phi_log_scale,
		'Q_prob_log_alpha' : Q_prob_log_alpha,
		'Q_prob_log_beta' : Q_prob_log_beta
	}


	# ** priors **

	# cell
	cell_prior_dict = {
		"prior_theta_euclid_loc" : prior_theta_euclid_loc,
		"prior_theta_scale" : prior_theta_scale,
		"prior_uniform_angle" : prior_uniform_angle
	}

	# gene
	gene_prior_dict = {
		"prior_mu_loc" : prior_mu_loc,
		"prior_mu_scale" : prior_mu_scale,
		"prior_A_alpha" : prior_A_alpha,
		"prior_A_beta" : prior_A_beta,
		"prior_phi_euclid_loc" : prior_phi_euclid_loc,
		"prior_phi_scale" : prior_phi_scale,
		'prior_Q_prob_alpha' : prior_Q_prob_alpha,
		'prior_Q_prob_beta' : prior_Q_prob_beta
	}


	# --- MAKE INPUT TENSORS ---

	# ** Gene X **
	try:
		gene_X = torch.Tensor(adata.X.todense())
	except:
		gene_X = torch.Tensor(adata.X)
	log_L = torch.log(torch.Tensor(adata.obs['lib_size']))



	return gene_X, log_L, gene_param_dict, cell_prior_dict, gene_prior_dict




def get_zero_kl_gene_param_dict_from_gene_prior_dict(gene_prior_dict):
	gene_param_dict = {}
	gene_param_dict['mu_loc'] = torch.nn.Parameter(gene_prior_dict['prior_mu_loc'].detach(), requires_grad = True)
	gene_param_dict['mu_log_scale'] = torch.nn.Parameter(torch.log(gene_prior_dict['prior_mu_scale']).detach(), requires_grad = True)
	gene_param_dict['A_log_alpha'] = torch.nn.Parameter(torch.log(gene_prior_dict['prior_A_alpha']).detach(), requires_grad = True)
	gene_param_dict['A_log_beta'] = torch.nn.Parameter(torch.log(gene_prior_dict['prior_A_beta']).detach(), requires_grad = True)
	gene_param_dict['phi_euclid_loc'] = torch.nn.Parameter(gene_prior_dict['prior_phi_euclid_loc'].detach(), requires_grad = True)
	gene_param_dict['phi_log_scale'] = torch.nn.Parameter(torch.log(utils.convert_vmf_to_ps_scale(gene_prior_dict['prior_phi_scale'])).detach(), requires_grad = True)
	gene_param_dict['Q_prob_log_alpha'] = torch.nn.Parameter(torch.log(gene_prior_dict['prior_Q_prob_alpha']).detach(), requires_grad = True)
	gene_param_dict['Q_prob_log_beta'] = torch.nn.Parameter(torch.log(gene_prior_dict['prior_Q_prob_beta']).detach(), requires_grad = True)
	return gene_param_dict






def harmonic_regression_prep(adata,
	bulk_cycler_info_path,
	reference_gene = None,
	min_amp = 0.0,
	max_amp = 1.5,
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
	non_clock_Q_prob_alpha = 1,
	non_clock_Q_prob_beta = 9,
	min_gene_prop = 1e-5,
	**kwargs):





	# --- GET MESOR VARIATIONAL PARAMETERS AND PRIOR DISTRIBUTION ---
	mu_loc, mu_log_scale, prior_mu_loc, prior_mu_scale = get_mesor_variational_and_prior_params(adata = adata,
		init_mesor_scale_val = init_mesor_scale_val, 
		prior_mesor_scale_val = prior_mesor_scale_val)


	# --- GET AMP VARIATIONAL PARAMETERS AND PRIOR DISTRIBUTION ---
	A_log_alpha, A_log_beta, prior_A_alpha, prior_A_beta = get_amp_variational_and_prior_params(
		adata = adata,
		init_amp_loc_val = init_amp_loc_val, 
		init_amp_scale_val = init_amp_scale_val,
		max_amp = max_amp,
		min_amp = min_amp,
		prior_amp_alpha_val = prior_amp_alpha_val, 
		prior_amp_beta_val = prior_amp_beta_val)


	# --- GET SHIFT VARIATIONAL PARAMETERS AND PRIOR DISTRIBUTION ---
	phi_euclid_loc, phi_log_scale, prior_phi_euclid_loc, prior_phi_scale = get_shift_variational_and_prior_params(adata = adata,
		known_cycler_init_shift_95_interval = known_cycler_init_shift_95_interval,
		unknown_cycler_init_shift_95_interval = unknown_cycler_init_shift_95_interval,
		known_cycler_prior_shift_95_interval = known_cycler_prior_shift_95_interval,
		reference_gene = reference_gene,
		bulk_cycler_info_path = bulk_cycler_info_path)




	# --- Q PARAMETERS ---
	Q_prob_log_alpha, Q_prob_log_beta, prior_Q_prob_alpha, prior_Q_prob_beta = get_cycler_prob_variational_and_prior_params(adata = adata,
		init_clock_Q_prob_alpha = init_clock_Q_prob_alpha,
		init_clock_Q_prob_beta = init_clock_Q_prob_beta,
		init_non_clock_Q_prob_alpha = init_non_clock_Q_prob_alpha,
		init_non_clock_Q_prob_beta = init_non_clock_Q_prob_beta,
		prior_clock_Q_prob_alpha = prior_clock_Q_prob_alpha,
		prior_clock_Q_prob_beta = prior_clock_Q_prob_beta,
		prior_non_clock_Q_prob_alpha = prior_non_clock_Q_prob_alpha, 
		prior_non_clock_Q_prob_beta = prior_non_clock_Q_prob_beta)





	# --- MAKE THE PARAMETER AND PRIOR DICTS ---


	# ** parameter dictionaries **

	# gene
	gene_param_dict = {
		"mu_loc" : mu_loc,
		"mu_log_scale" : mu_log_scale,
		"A_log_alpha" : A_log_alpha,
		"A_log_beta" : A_log_beta,
		"phi_euclid_loc" : phi_euclid_loc,
		"phi_log_scale" : phi_log_scale,
		'Q_prob_log_alpha' : Q_prob_log_alpha,
		'Q_prob_log_beta' : Q_prob_log_beta
	}


	# ** priors **

	# gene
	gene_prior_dict = {
		"prior_mu_loc" : prior_mu_loc,
		"prior_mu_scale" : prior_mu_scale,
		"prior_A_alpha" : prior_A_alpha,
		"prior_A_beta" : prior_A_beta,
		"prior_phi_euclid_loc" : prior_phi_euclid_loc,
		"prior_phi_scale" : prior_phi_scale,
		'prior_Q_prob_alpha' : prior_Q_prob_alpha,
		'prior_Q_prob_beta' : prior_Q_prob_beta
	}


	# --- MAKE INPUT TENSORS ---

	# ** Gene X **
	try:
		gene_X = torch.Tensor(adata.X.todense())
	except:
		gene_X = torch.Tensor(adata.X)
	log_L = torch.log(torch.Tensor(adata.obs['lib_size']))


	return gene_X, log_L, gene_param_dict, gene_prior_dict














