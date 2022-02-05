import torch
import power_spherical
import numpy as np
import pandas as pd
import hyperspherical_vae

# tempo
# import cell_posterior
from . import cell_posterior



# dist is Beta
def get_is_cycler_samples_from_dist(dist, num_gene_samples, rsample = True):
	
	# get num genes
	num_genes = dist.concentration0.shape[0]
	
	# sample bernoulli distribution probs from the beta: [num_samples x num_genes]
	if rsample:
		bernoulli_success_prob_samples = dist.rsample((num_gene_samples,))
	else:
		bernoulli_success_prob_samples = dist.sample((num_gene_samples,))
	
	# sample cycler indicators from the bernoulli
	logits = torch.zeros(num_gene_samples, num_genes,2)
	logits[:,:,0] = bernoulli_success_prob_samples # success prob
	logits[:,:,1] = 1 - logits[:,:,0] # failure prob
	logits = torch.log(logits)
	Q_sampled = torch.nn.functional.gumbel_softmax(logits = logits,
									  tau = 1,
									  hard = True,
									  dim = -1)   
	Q_sampled = Q_sampled[:,:,0] # [num_samples, num_genes]
	
	return Q_sampled


def convert_ps_to_vmf_scale(ps_scale):
	vmf_scale = ps_scale * 0.5
	return vmf_scale
def convert_vmf_to_ps_scale(vmf_scale):
	ps_scale = vmf_scale * 2.0
	return ps_scale



def get_transformed_beta_param_loc_and_scale(alpha, beta, min_val=0.0, max_val=1.0):

	# loc
	A_loc_unscaled = alpha / (alpha + beta)
	A_loc = (A_loc_unscaled * (max_val - min_val)) + min_val

	# scale
	num = alpha * beta
	denom = ((alpha + beta) ** 2) * (alpha + beta + 1)
	variance = num / denom
	A_scale = variance * ((max_val - min_val) ** 2)

	return A_loc, A_scale



def init_transformed_beta_dist(amp_alpha, amp_beta, min_val = 0.0, max_val = 1.0):

	# ** define transformations **
	
	# scale
	scale = max_val - min_val
	transform1 = torch.distributions.transforms.AffineTransform(loc = 0.0, scale = scale) # [convert [0,1]] to [0,max_val-min_val]

	# add min_val
	transform2 = torch.distributions.transforms.AffineTransform(loc = min_val, scale = 1.0) # [0,max_val-min_val] to [min_val,ma_amp]
	
	# ** init base distribution **
	base_distrib = torch.distributions.beta.Beta(concentration1 = amp_alpha, concentration0 = amp_beta)

	# ** init transformed dist **
	transformed_dist = torch.distributions.transformed_distribution.TransformedDistribution(base_distribution = base_distrib,
																					   transforms = [transform1,
																									transform2])
	return transformed_dist



class TransformedBeta():
	def __init__(self,min_val,max_val,alpha,beta):
		self.alpha = alpha
		self.beta = beta
		self.min_val = min_val
		self.max_val = max_val
		self.loc, self.scale = get_transformed_beta_param_loc_and_scale(self.alpha,self.beta,self.min_val,self.max_val)
		self.dist = init_transformed_beta_dist(alpha, beta, min_val = 0.0, max_val = 1.0)



	def sample(self, shape):
		samples = self.dist.sample(shape)
		return samples
	def rsample(self, shape):
		samples = self.dist.rsample(shape)
		return samples
	def log_prob(self, data):
		return self.dist.log_prob(data)







def get_norm_euclid_loc(angle_euclid_loc):

	# ensure powerspherical loc has L2 norm of 1
	angle_euclid_loc_norms = (angle_euclid_loc ** 2).sum(dim=1) ** 0.5
	angle_euclid_loc_norm_1 = angle_euclid_loc / angle_euclid_loc_norms.reshape(-1,1)

	return angle_euclid_loc_norm_1







def get_distribution_loc_and_scale(gene_param_dict=None, gene_prior_dict=None, cell_prior_dict=None, max_amp = 1.0, min_amp = 0.0, prep = True):


	# --- PREP THE VARIATIONAL GENE PARAMETERS IF THEY'RE NOT ---
	if prep and gene_param_dict is not None:
		gene_param_dict = prep_gene_params(gene_param_dict)


	# --- GENE LOC/SCALE ---
	mu_loc,mu_scale,A_loc,A_scale,phi_euclid_loc,phi_scale,Q_prob_loc,Q_prob_scale = None,None,None,None,None,None,None,None
	if gene_param_dict is not None:

		# ** mu **
		if 'mu_loc' in gene_param_dict and 'mu_scale' in gene_param_dict:
			mu_loc, mu_scale = gene_param_dict['mu_loc'], gene_param_dict['mu_scale']


		# ** A **
		if 'A_alpha' in gene_param_dict and 'A_beta' in gene_param_dict:
			A_loc, A_scale = get_transformed_beta_param_loc_and_scale(gene_param_dict['A_alpha'], gene_param_dict['A_beta'], min_val = min_amp, max_val = max_amp)

		# ** phi **
		if 'phi_euclid_loc' in gene_param_dict and 'phi_scale' in gene_param_dict:
			phi_euclid_loc, phi_scale= gene_param_dict['phi_euclid_loc'],gene_param_dict['phi_scale']

		# ** Q **
		if 'Q_prob_alpha' in gene_param_dict and 'Q_prob_beta' in gene_param_dict:
			Q_prob_loc, Q_prob_scale = get_transformed_beta_param_loc_and_scale(gene_param_dict['Q_prob_alpha'], gene_param_dict['Q_prob_beta'], min_val = 0.0, max_val = 1.0)





	# --- GENE PRIOR LOC / SCALE ---
	prior_mu_loc,prior_mu_scale,prior_A_loc,prior_A_scale,prior_phi_euclid_loc,prior_phi_scale,prior_Q_prob_loc,prior_Q_prob_scale = None,None,None,None,None,None,None,None
	if gene_prior_dict is not None:

		# ** mu **
		if 'prior_mu_loc' in gene_prior_dict and 'prior_mu_scale' in gene_prior_dict:
			prior_mu_loc, prior_mu_scale = gene_prior_dict['prior_mu_loc'], gene_prior_dict['prior_mu_scale']
		
		# ** A **
		if 'prior_A_alpha' in gene_prior_dict and 'prior_A_beta' in gene_prior_dict:
			prior_A_loc, prior_A_scale = get_transformed_beta_param_loc_and_scale(gene_prior_dict['prior_A_alpha'], gene_prior_dict['prior_A_beta'], min_val = min_amp, max_val = max_amp)

		# ** phi **
		if 'prior_phi_euclid_loc' in gene_prior_dict and 'prior_phi_scale' in gene_prior_dict:
			prior_phi_euclid_loc, prior_phi_scale = gene_prior_dict['prior_phi_euclid_loc'], gene_prior_dict['prior_phi_scale']

		# ** Q **
		if 'prior_Q_prob_alpha' in gene_prior_dict and 'prior_Q_prob_beta' in gene_prior_dict:
			prior_Q_prob_loc, prior_Q_prob_scale = get_transformed_beta_param_loc_and_scale(gene_prior_dict['Q_prob_alpha'], gene_prior_dict['Q_prob_beta'], min_val = 0.0, max_val = 1.0)


	# --- CELL PRIOR LOC / SCALE ---
	prior_theta_euclid_loc, prior_theta_scale = None, None
	if cell_prior_dict is not None and not cell_prior_dict['prior_uniform_angle']:
		if 'prior_theta_euclid_loc' in cell_prior_dict and 'prior_theta_scale' in cell_prior_dict:
			prior_theta_euclid_loc, prior_theta_scale = cell_prior_dict['prior_theta_euclid_loc'], cell_prior_dict['prior_theta_scale']


	# --- MAKE THE LOC/SCALE DICT ---
	loc_scale_dict = {
		"mu_loc" : mu_loc,
		"mu_scale" : mu_scale,
		"A_loc" : A_loc,
		"A_scale" : A_scale,
		"phi_euclid_loc" : phi_euclid_loc,
		"phi_scale" : phi_scale,
		'Q_prob_loc' : Q_prob_loc,
		'Q_prob_scale' : Q_prob_scale,
		"prior_mu_loc" : prior_mu_loc,
		"prior_mu_scale" : prior_mu_scale,
		"prior_A_loc" : prior_A_loc,
		"prior_A_scale" : prior_A_scale,
		"prior_phi_euclid_loc" : prior_phi_euclid_loc,
		"prior_phi_scale" : prior_phi_scale,
		'prior_Q_prob_loc' : prior_Q_prob_loc,
		'prior_Q_prob_scale' : prior_Q_prob_scale,
		"prior_theta_euclid_loc" : prior_theta_euclid_loc,
		"prior_theta_scale" : prior_theta_scale
	}


	return loc_scale_dict




def prep_gene_params(gene_param_dict):

	# --- PREP GENE VARIATIONAL PARAMETERS ---
	mu_loc,mu_scale,A_alpha,A_beta,phi_euclid_loc,phi_scale,Q_prob_alpha,Q_prob_beta=gene_param_dict['mu_loc'],None,None,None,None,None,None,None
	if gene_param_dict:
		if 'mu_log_scale' in gene_param_dict:
			mu_scale = torch.exp(gene_param_dict['mu_log_scale'])
		if 'A_log_alpha' in gene_param_dict and 'A_log_beta' in gene_param_dict:
			A_alpha = torch.exp(gene_param_dict['A_log_alpha'])
			A_beta = torch.exp(gene_param_dict['A_log_beta'])
		if 'phi_euclid_loc' in gene_param_dict and 'phi_log_scale' in gene_param_dict:
			phi_euclid_loc = get_norm_euclid_loc(gene_param_dict['phi_euclid_loc'])
			phi_scale = torch.exp(gene_param_dict['phi_log_scale'])
		if 'Q_prob_log_alpha' in gene_param_dict and 'Q_prob_log_beta' in gene_param_dict:
			Q_prob_alpha = torch.exp(gene_param_dict['Q_prob_log_alpha'])
			Q_prob_beta = torch.exp(gene_param_dict['Q_prob_log_beta'])


	prepped_gene_param_dict = {
		"mu_loc" : mu_loc,
		"mu_scale" : mu_scale,
		"A_alpha" : A_alpha,
		"A_beta" : A_beta,
		'phi_euclid_loc' : phi_euclid_loc,
		'phi_scale' : phi_scale,
		'Q_prob_alpha' : Q_prob_alpha,
		'Q_prob_beta' : Q_prob_beta
	}

	return prepped_gene_param_dict


def init_distributions_from_param_dicts(gene_param_dict = None, cell_prior_dict = None, gene_prior_dict = None, max_amp = 1.0, min_amp = 0.0, prep = True):


	# --- PREP THE VARIATIONAL GENE PARAMETERS ---
	if prep and gene_param_dict is not None:
		gene_param_dict = prep_gene_params(gene_param_dict)



	# --- GET GENE DISTRIBUTIONS ---
	mu_dist, A_dist, phi_euclid_dist, Q_prob_dist = None, None, None, None
	if gene_param_dict is not None:

		# mu
		if 'mu_loc' in gene_param_dict and 'mu_scale' in gene_param_dict:
			mu_dist = torch.distributions.normal.Normal(loc = gene_param_dict['mu_loc'], scale = gene_param_dict['mu_scale'])

		# A
		if 'A_alpha' in gene_param_dict and 'A_beta' in gene_param_dict:
			A_dist = init_transformed_beta_dist(gene_param_dict['A_alpha'], gene_param_dict['A_beta'], min_val = min_amp, max_val = max_amp)
			# A_dist = TransformedBeta(min_val = min_amp, max_val = max_amp, alpha = gene_param_dict['A_alpha'], beta = gene_param_dict['A_beta'])

		# phi
		if 'phi_euclid_loc' in gene_param_dict and 'phi_scale' in gene_param_dict:
			phi_euclid_dist = power_spherical.PowerSpherical(loc = gene_param_dict['phi_euclid_loc'],
															scale = gene_param_dict['phi_scale'])

		# Q
		if 'Q_prob_alpha' in gene_param_dict and 'Q_prob_beta' in gene_param_dict:
			Q_prob_dist = torch.distributions.beta.Beta(gene_param_dict['Q_prob_alpha'],gene_param_dict['Q_prob_beta'])




	# --- GENE PRIOR DISTRIBUTIONS ---
	prior_mu_dist, prior_A_dist, prior_phi_euclid_dist, prior_Q_prob_dist = None, None, None, None
	if gene_prior_dict is not None:

		# mu
		if 'prior_mu_loc' in gene_prior_dict and 'prior_mu_scale' in gene_prior_dict:
			prior_mu_dist = torch.distributions.normal.Normal(loc = gene_prior_dict['prior_mu_loc'], scale = gene_prior_dict['prior_mu_scale'])

		# A
		if 'prior_A_alpha' in gene_prior_dict and 'prior_A_beta' in gene_prior_dict:
			prior_A_dist = init_transformed_beta_dist(gene_prior_dict['prior_A_alpha'], gene_prior_dict['prior_A_beta'], min_val = min_amp, max_val = max_amp)
			# prior_A_dist = TransformedBeta(min_val = min_amp, max_val = max_amp, alpha = gene_prior_dict['prior_A_alpha'], beta = gene_prior_dict['prior_A_beta'])

		# phi
		if 'prior_phi_euclid_loc' in gene_prior_dict and 'prior_phi_scale' in gene_prior_dict:
			prior_phi_euclid_dist = hyperspherical_vae.distributions.von_mises_fisher.VonMisesFisher(loc = gene_prior_dict['prior_phi_euclid_loc'], scale = gene_prior_dict['prior_phi_scale'])

		# Q
		if 'prior_Q_prob_alpha' in gene_prior_dict and 'prior_Q_prob_beta' in gene_prior_dict:
			prior_Q_prob_dist = torch.distributions.beta.Beta(gene_prior_dict['prior_Q_prob_alpha'],gene_prior_dict['prior_Q_prob_beta'])




	# --- CELL PRIOR DISTRIBUTIONS ---
	prior_theta_euclid_dist = None
	if cell_prior_dict is not None:

		# non-informative prior
		if cell_prior_dict['prior_uniform_angle']:
			prior_theta_euclid_dist = power_spherical.HypersphericalUniform(dim = 2) # set to hyperspherical uniform

		# informative prior
		elif 'prior_theta_euclid_loc' in cell_prior_dict and 'prior_theta_scale' in cell_prior_dict:
			prior_theta_euclid_dist = hyperspherical_vae.distributions.von_mises_fisher.VonMisesFisher(loc = cell_prior_dict['prior_theta_euclid_loc'],
															scale = cell_prior_dict['prior_theta_scale'])
		else:
			raise Exception("Error: prior_uniform_angle = False, but no informative parameters provided for cell prior.")




	# --- MAKE THE DISTRIB DICT ---
	distrib_dict = {
		"mu" : mu_dist,
		"A" : A_dist,
		"phi_euclid" : phi_euclid_dist,
		'Q_prob' : Q_prob_dist,
		"prior_mu" : prior_mu_dist,
		"prior_A" : prior_A_dist,
		"prior_phi_euclid" : prior_phi_euclid_dist,
		'prior_Q_prob' : prior_Q_prob_dist,
		"prior_theta_euclid" : prior_theta_euclid_dist
	}


	return distrib_dict





# https://discuss.pytorch.org/t/how-to-convert-a-dense-matrix-to-a-sparse-one/7809/2
def to_sparse_tensor(x):
	""" converts dense tensor x to sparse format """
	x_typename = torch.typename(x).split('.')[-1]
	sparse_tensortype = getattr(torch.sparse, x_typename)

	indices = torch.nonzero(x)
	if len(indices.shape) == 0:  # if all elements are zeros
		return sparse_tensortype(*x.shape)
	indices = indices.t()
	values = x[tuple(indices[i] for i in range(indices.shape[0]))]
	return sparse_tensortype(indices, values, x.size())


# given expected molecule difference, library size, and log mesor (mu), returns the log amplitude
# required to at least exceed the expected molecule difference
# - possible modes:
#	- 'min_to_max'
#	- 'mesor_to_max'
def mesor_to_amp_threshold(expected_molecule_dif, L, mu,mode):
	if mode == 'mesor_to_max':
		log_amp_thresh = np.log((expected_molecule_dif / L) + np.exp(mu)) - mu
	elif mode == "min_to_max":
		raise Exception("Error: min_to_max not yet implemented")
	else:
		raise Exception("Mode '%s' not implemented." % (mode))
	return log_amp_thresh


# given expected molecule difference, library size, log mesor (mu), and log amp (A),
# returns whether or not the expected molecule difference is met by the mesor and amplitude combo
# - possible modes:
#	- 'min_to_max'
#	- 'mesor_to_max'
def passes_map_amp_threshold(expected_molecule_dif, L, mu, A, mode):
	if mode == 'mesor_to_max':
		return (L * np.exp(mu + A)) - (L * np.exp(mu)) >= expected_molecule_dif
	elif mode == "min_to_max":
		return (L * np.exp(mu + A)) - (L * np.exp(mu - A)) >= expected_molecule_dif
	else:
		raise Exception("Mode '%s' not implemented." % (mode))



def passes_credible_interval_amp_threshold(expected_molecule_dif, L, mu_samples, A_samples, mode, quantile_threshold = 0.975):
	if mode == 'mesor_to_max':
		dif_samples = (L * torch.exp(mu_samples + A_samples)) - (L * torch.exp(mu_samples)) # [num_samples x num_genes]
	elif mode == 'min_to_max':
		dif_samples = (L * torch.exp(mu_samples + A_samples)) - (L * torch.exp(mu_samples - A_samples)) # [num_samples x num_genes]
	else:
		raise Exception("Mode '%s' not implemented." % (mode))
	quantile_vals = torch.quantile(dif_samples,torch.Tensor([quantile_threshold]),dim=0)[0,:] # [num_genes]
	return quantile_vals >= expected_molecule_dif


# interval specified is in radains [0,pi]
def powerspherical_95_radian_interval_to_concentration(interval):

	# interval must be from [0,pi]
	if interval > np.pi or interval < 0:
		raise Exception("Error: interval cannot be larger than pi or smaller than zero.")

	# scale from 0 to 1
	interval = interval / np.pi

	# get beta_0 and beta_1
	beta_0 = 0.71972177
	beta_1 = -1.61950415

	# compute z val
	z_val = np.log(interval / (1 - interval))

	# compute the log10_concentration
	log10_concentration = (z_val - beta_0) / beta_1

	# compute the concentration
	concentration = np.power(10.0,log10_concentration)

	return concentration


def log10_concentration_to_radian_resolution_99(log10_concentration):

	beta_0 = 1.35332452
	beta_1 = -1.87457903
	input_to_logistic = beta_0 + (beta_1 * log10_concentration)
	outputs = 1.0 / (1.0 + np.exp(-1 * input_to_logistic))
	resolution_in_radians = outputs * np.pi

	return resolution_in_radians


def log10_concentration_to_hour_resolution_99(log10_concentration):

	beta_0 = 1.35332452
	beta_1 = -1.87457903
	input_to_logistic = beta_0 + (beta_1 * log10_concentration)
	outputs = 1.0 / (1.0 + np.exp(-1 * input_to_logistic))
	resolution_in_hours = outputs * 12.0

	return resolution_in_hours

def log10_concentration_to_radian_resolution_95(log10_concentration):

	beta_0 = 0.71972177
	beta_1 = -1.61950415
	input_to_logistic = beta_0 + (beta_1 * log10_concentration)
	outputs = 1.0 / (1.0 + np.exp(-1 * input_to_logistic))
	resolution_in_radians = outputs * np.pi

	return resolution_in_radians


def log10_concentration_to_hour_resolution_95(log10_concentration):

	beta_0 = 0.71972177
	beta_1 = -1.61950415
	input_to_logistic = beta_0 + (beta_1 * log10_concentration)
	outputs = 1.0 / (1.0 + np.exp(-1 * input_to_logistic))
	resolution_in_hours = outputs * 12.0

	return resolution_in_hours



def compute_angle_dif(angle_1,angle_2):
	angle_1 = angle_1 % (2 * np.pi)
	angle_2 = angle_2 % (2 * np.pi)
	angle_dif = np.abs(angle_2 - angle_1)
	angle_dif[np.where(angle_dif > np.pi)] = (2 * np.pi) - angle_dif[np.where(angle_dif > np.pi)]
	return angle_dif
def compute_hour_dif(angle_1,angle_2):
	angle_dif = compute_angle_dif(angle_1,angle_2)
	hour_dif = (angle_dif / np.pi) * 12.0
	return hour_dif




def get_optimal_predicted_phase_when_reference_gene_unknown(true_phase, predicted_phase, viz = False):


	# ** compute the estimated phase at all possible shifts **
	shift_vec = np.linspace(0,2*np.pi,100)
	predicted_phase_shifted_mat = (predicted_phase - shift_vec.reshape(-1,1)) % (2 * np.pi) # [num_shifts, num_cells]

	# ** compute shift of the pseudotime with the minimum loss **
	radian_error_shifted = compute_angle_dif(true_phase,predicted_phase_shifted_mat)
	radian_error_shifted = np.mean(radian_error_shifted,axis=1)
	min_error_shift_index = np.argmin(radian_error_shifted)
	min_error_shift = shift_vec[min_error_shift_index]


	# ** viz **
	if viz:
		import matplotlib.pyplot as plt
		
		plt.clf()
		plt.scatter(shift_vec,radian_error_shifted)
		plt.title("Mean error at all shifts of predicted pseudotime")
		plt.xlabel("Shift applied to predicted pseudotime (radians)")
		plt.ylabel("Mean error (radians)")
		plt.axvline(x = shift_vec[min_error_shift_index],c='r')
		plt.show()

	# ** shift the pseudotime to the min error shift **
	predicted_phase_shifted = (predicted_phase + min_error_shift) % (2 * np.pi)
	
	return predicted_phase_shifted



def init_cycler_adata_variational_and_prior_dist_from_prev_round(cycler_adata, previous_alg_step_subfolder, enforce_de_novo_cycler_flat_Q_prior = False):


	# load gene param df for previous round's cycler genes
	previous_cycler_gene_param_df_fileout = '%s/cell_phase_estimation/gene_prior_and_posterior.tsv' % previous_alg_step_subfolder
	previous_cycler_gene_param_df = pd.read_table(previous_cycler_gene_param_df_fileout,sep='\t',index_col='gene')

	# load the gene param df for the previous round's de novo cyclers
	previous_de_novo_cycler_gene_param_df_fileout = '%s/de_novo_cycler_id/gene_prior_and_posterior.tsv' % previous_alg_step_subfolder
	previous_de_novo_cycler_gene_param_df = pd.read_table(previous_de_novo_cycler_gene_param_df_fileout,sep='\t',index_col='gene')
	previous_de_novo_cyclers = np.intersect1d(np.array(cycler_adata.var_names), np.array(previous_de_novo_cycler_gene_param_df.index))
	previous_de_novo_cycler_gene_param_df = previous_de_novo_cycler_gene_param_df.loc[previous_de_novo_cyclers]

	# concat previous cycler and de novo cycler param df's
	current_cycler_gene_param_df = pd.concat((previous_cycler_gene_param_df, previous_de_novo_cycler_gene_param_df))

	# make sure current_cycler_gene_param_df is in the same order as cycler_adata
	current_cycler_gene_param_df = current_cycler_gene_param_df.loc[np.array(cycler_adata.var_names)] 

	# add parameters to adata to initialize distribution parameters
	cols_to_keep = list(current_cycler_gene_param_df.columns) # drop prior columns
	for col in cols_to_keep:
		cycler_adata.var[col] = np.array(current_cycler_gene_param_df[col])


	if enforce_de_novo_cycler_flat_Q_prior:
		non_clock_indices = np.where(~cycler_adata.var['is_clock'])[0]
		cycler_gene_prior_dict['prior_Q_prob_alpha'][non_clock_indices] = torch.ones((non_clock_indices.shape[0])) # set the non-clock cycler Q priors to flat
		cycler_gene_prior_dict['prior_Q_prob_beta'][non_clock_indices] = torch.ones((non_clock_indices.shape[0]))

	return cycler_adata




def init_hv_adata_variational_and_prior_dist_from_prev_round(hv_adata, previous_alg_step_subfolder):



	# load the gene param df for the previous round's de novo cyclers
	previous_de_novo_cycler_gene_param_df_fileout = '%s/de_novo_cycler_id/gene_prior_and_posterior.tsv' % previous_alg_step_subfolder
	previous_de_novo_cycler_gene_param_df = pd.read_table(previous_de_novo_cycler_gene_param_df_fileout,sep='\t',index_col='gene')
	
	# get the current_non_cycler_gene_param_df
	current_non_cycler_gene_param_df = previous_de_novo_cycler_gene_param_df.loc[np.array(hv_adata.var_names)]

	# # filter cols relevant for initializing variational parameters
	# cols_to_keep = list(filter(lambda x: "prior" not in x, current_non_cycler_gene_param_df.columns)) # drop prior columns
	# current_non_cycler_gene_param_df = current_non_cycler_gene_param_df[cols_to_keep]


	# add parameters to adata to initialize cycler genes 
	cols_to_keep = list(current_non_cycler_gene_param_df.columns)
	for col in cols_to_keep:
		hv_adata.var[col] = np.array(current_non_cycler_gene_param_df[col])

	return hv_adata











