import torch
import power_spherical
import hyperspherical_vae


# tempo impoorts
from . import utils







def compute_nb_ll(gene_X,log_prop,log_lambda,log_mean_log_disp_coef):

	# get the gene dispersions based on the mean estimates
	log_disp = log_mean_log_disp_coef[0]
	if len(log_mean_log_disp_coef) > 1:
		power = 1
		for coef in log_mean_log_disp_coef[1:]:
			log_disp = log_disp + (coef * (log_prop ** power))
			power += 1


	# clip the log disp
	log_disp = torch.clamp(log_disp,min=-5,max=5)


	# compute the negative binomial LL over the samples
	inv_disp = torch.exp(-1 * log_disp)
	mean_param = torch.exp(log_lambda)
	eps = 1e-8
	log_theta_mu_eps = torch.log(inv_disp + mean_param + eps)
	ll = (
		inv_disp * (torch.log(inv_disp + eps) - log_theta_mu_eps)
		+ gene_X * (torch.log(mean_param + eps) - log_theta_mu_eps)
		+ torch.lgamma(gene_X + inv_disp)
		- torch.lgamma(inv_disp)
		- torch.lgamma(gene_X + 1)
	)

	return ll



# - Description: given observed gene count matrix, cell library sizes, distributions / samples of gene parameters,
#	and distributions / samples of cell parameters, returns sample log likelihood of the data
# - Note: for each parameter, users must supply either a Tensor of samples or the PyTorch distribution. If both supplied,
#	the samples are used by default.
# - Parameters:
#		- gene_X ([num_cells x num_genes] Tensor of ints): gene count matrix 
#		- log_L ([num_cells] Tensor floats): natural log of cell library sizes
#		- theta_sampled ([num_cells x num_samples] Tensor of floats): matrix of cell phase samples in radians
#		- mu_sampled ([num_samples x num_genes] Tensor of floats): matrix of gene mesor samples
#		- A_sampled ([num_samples x num_genes] Tensor of floats): matrix of gene amplitude samples
#		- phi_sampled ([num_samples x num_genes] Tensor of floats): matrix of gene acrophase samples in radians
#		- theta_euclid_dist (PyTorch PowerSpherical distribution): PyTorch distribution of cell phases
#		- mu_dist (PyTorch normal distribution): PyTorch distribution of gene mesors
#		- A_dist (PyTorch transformed Beta distribution): PyTorch distribution of gene amplitudes
#		- phi_euclid_dist (PyTorch PowerSpherical distribution): PyTorch distribution of gene acrophases
#		- num_cell_samples (int): number of samples to use for cell distributions (if cell samples not provided and cell distribution provided)
#		- num_gene_samples (int): number of samples to use for gene distributions (if gene samples not provided and gene distribution provided)
#		- use_flat_model (bool): if true, assumes flat model of expression (i.e. amplitude = 0), otherwise assumes harmonic model
#		- use_nb (bool): if true, uses Negative Binomial likelihood model, otherwise uses Poisson
#		- mean_disp_coef (numpy array of floats): if use_nb = True, describes the log mean - log dispersion relationship shared by all genes
#		- rsample (bool): whether to track gradients or not when sampling (used for learning parameters)
#		- batch_indicator_mat ([num_cells x num_batches] Tensor of 0/1's): cell indicator for batch
#		- batch_indicator_effect_mat ([num_batches x num_genes] Tensor of effects): batch effects for each gene
# - Outputs:
#		- log_likelihood_sampled ([num_genes, num_cells, num_gene_samples, num_cell_samples] Tensor of floats): Tensor of sample log likelihoods

def compute_sample_log_likelihood(gene_X, log_L,
	theta_sampled = None, mu_sampled = None, A_sampled = None, phi_sampled = None, Q_sampled = None,
	theta_euclid_dist = None, mu_dist = None, A_dist = None, phi_euclid_dist = None, Q_prob_dist = None,
	num_cell_samples = 5, num_gene_samples = 5, use_flat_model = False, use_nb = False,
	log_mean_log_disp_coef = None, rsample = True, batch_indicator_mat = None, batch_indicator_effect_mat = None, use_is_cycler_indicators = False):

	# --- GET PARAMETER SAMPLES NEEDED FOR FLAT MODEL ----

	# ** for each variable (theta, mu, A, phi), make sure either samples were given or that a distribution was given **

	# mu
	if mu_sampled is None and mu_dist is None:
		raise Exception("Error: neither samples nor a distribution was provided for mu.")

	# ** for each variable, obtain samples if samples not already provided **

	# mu: [num_samples x num_genes]
	if mu_sampled is None:
		if rsample:
			mu_sampled = mu_dist.rsample(sample_shape = (num_gene_samples,))
		else:
			mu_sampled = mu_dist.sample(sample_shape = (num_gene_samples,))


	# --- GET PARAMETER SAMPLES NEEDED FOR HARMONIC MODEL ----
	if not use_flat_model:

		# ** for each variable (theta, mu, A, phi), make sure either samples were given or that a distribution was given **

		# theta
		if theta_sampled is None and theta_euclid_dist is None:
			raise Exception("Error: neither samples nor a distribution was provided for theta.")

		# A
		if A_sampled is None and A_dist is None:
			raise Exception("Error: neither samples nor a distribution was provided for A.")

		# phi
		if phi_sampled is None and phi_euclid_dist is None:
			raise Exception("Error: neither samples nor a distribution was provided for phi.")


		# ** for each variable, obtain samples if samples not already provided **


		# A: [num_samples x num_genes]
		if A_sampled is None:
			if rsample:
				A_sampled = A_dist.rsample(sample_shape = (num_gene_samples,))
			else:
				A_sampled = A_dist.sample(sample_shape = (num_gene_samples,))


		# phi: [num_samples x num_genes]; phi_euclid_sampled: [num_samples x num_genes x 2]
		if phi_sampled is None:
			if rsample:
				phi_euclid_sampled = phi_euclid_dist.rsample(sample_shape = (num_gene_samples,))
			else:
				phi_euclid_sampled = phi_euclid_dist.sample(sample_shape = (num_gene_samples,))
			phi_sampled = torch.atan2(phi_euclid_sampled[:,:,1], phi_euclid_sampled[:,:,0])

		
		# is_cycler: [num_samples x num_genes]
		if use_is_cycler_indicators:
			if Q_sampled is None:
				Q_sampled = utils.get_is_cycler_samples_from_dist(Q_prob_dist, num_gene_samples, rsample = rsample)            
			

		# ** compute pairwise differences among theta and phi **
		mat_A = phi_sampled.T
		mat_B = theta_sampled
		num_genes, num_samples_1 = mat_A.shape
		num_cells, num_samples_2 = mat_B.shape
		theta_phi_dif = mat_A.view(num_genes,1,-1,1) - mat_B.view(1,num_cells,1,-1) # [num_genes,num_cells,num_gene_samples,num_cell_samples]




	# ** compute the expectation **

	# compute log_prop_sampled: [num_genes, num_cells, num_gene_samples, num_cell_samples]
	if use_flat_model:
		num_genes = gene_X.shape[1]
		log_prop_sampled = mu_sampled.T.view(num_genes,1,-1,1)
	else:
		if use_is_cycler_indicators:
			log_prop_sampled = mu_sampled.T.view(num_genes,1,-1,1) + (Q_sampled.T.view(num_genes,1,-1,1) * A_sampled.T.view(num_genes,1,-1,1) * torch.cos(theta_phi_dif))
		else:
			log_prop_sampled = mu_sampled.T.view(num_genes,1,-1,1) + (A_sampled.T.view(num_genes,1,-1,1) * torch.cos(theta_phi_dif))



	# compute lambda sampled: [num_genes, num_cells, num_gene_samples, num_cell_samples]
	log_lambda_sampled = log_prop_sampled + log_L.view(1,log_L.shape[0],1,1)

	# add batch effect if specified
	if batch_indicator_effect_mat is not None and batch_indicator_mat is not None:
		batch_effect = torch.matmul(batch_indicator_mat,batch_indicator_effect_mat) # [num_cells x num_genes]
		batch_effect = batch_effect.T.unsqueeze(2).unsqueeze(3) # [num_genes, num_cells, 1, 1]
		log_lambda_sampled = log_lambda_sampled + batch_effect

	# reshape the GE
	gene_X_reshaped = gene_X.unsqueeze(0).unsqueeze(0).T # [num_genes, num_cells, 1, 1]


	# ** compute the LL over samples, either under NB or Poisson **
	if use_nb:
		log_likelihood_sampled = compute_nb_ll(gene_X_reshaped,log_prop_sampled,log_lambda_sampled,log_mean_log_disp_coef)
	else:
		log_likelihood_sampled = torch.distributions.poisson.Poisson(rate = torch.exp(log_lambda_sampled)).log_prob(gene_X_reshaped)


	return log_likelihood_sampled



# - Description: given [num_genes, num_cells, num_gene_samples, num_cell_samples] sample log likelihood of the data,
#	computes the expected log likelihood over genes or cells
# - Parameters:
#		- gene_X ([num_cells x num_genes] Tensor of ints): gene count matrix 
#		- log_L ([num_cells] Tensor floats): natural log of cell library sizes
#		- theta_sampled ([num_cells x num_samples] Tensor of floats): matrix of cell phase samples in radians
#		- mu_sampled ([num_samples x num_genes] Tensor of floats): matrix of gene mesor samples
#		- A_sampled ([num_samples x num_genes] Tensor of floats): matrix of gene amplitude samples
#		- phi_sampled ([num_samples x num_genes] Tensor of floats): matrix of gene acrophase samples in radians
#		- theta_euclid_dist (PyTorch PowerSpherical distribution): PyTorch distribution of cell phases
#		- mu_dist (PyTorch normal distribution): PyTorch distribution of gene mesors
#		- A_dist (PyTorch transformed Beta distribution): PyTorch distribution of gene amplitudes
#		- phi_euclid_dist (PyTorch PowerSpherical distribution): PyTorch distribution of gene acrophases
#		- num_cell_samples (int): number of samples to use for cell distributions (if cell samples not provided and cell distribution provided)
#		- num_gene_samples (int): number of samples to use for gene distributions (if gene samples not provided and gene distribution provided)
#		- exp_over_cells (bool): True if computing the expectation over cells, otherwise computes over genes
#		- use_flat_model (bool): if true, assumes flat model of expression (i.e. amplitude = 0), otherwise assumes harmonic model
#		- use_nb (bool): if true, uses Negative Binomial likelihood model, otherwise uses Poisson
#		- mean_disp_coef (numpy array of floats): if use_nb = True, describes the log mean - log dispersion relationship shared by all genes
#		- rsample (bool): whether to track gradients or not when sampling (used for learning parameters)
# - Outputs:
#		- expectation_log_likelihood ([num_cells] or [num_genes] Tensor of floats): Tensor of expected log likelihoods for genes or cells (depending on exp_over_cells)

def compute_mc_expectation_log_likelihood(gene_X, log_L,
	theta_sampled = None, mu_sampled = None, A_sampled = None, phi_sampled = None, Q_sampled = None,
	theta_euclid_dist = None, mu_dist = None, A_dist = None, phi_euclid_dist = None, Q_prob_dist = None,
	num_cell_samples = 5, num_gene_samples = 5, exp_over_cells = True, use_flat_model = False,
	use_nb = False, log_mean_log_disp_coef = None, rsample = True, batch_indicator_mat = None, batch_indicator_effect_mat = None, use_is_cycler_indicators = False):

		  
	# --- COMPUTE LOG LIKELIHOOD OVER SAMPLES (RETURNS [NUM_GENES X NUM_CELLS X NUM_GENE_SAMPLES X NUM_CELL_SAMPLES]) ---
	log_likelihood_sampled = compute_sample_log_likelihood(gene_X, log_L,
		theta_sampled = theta_sampled, mu_sampled = mu_sampled, A_sampled = A_sampled, phi_sampled = phi_sampled, Q_sampled = Q_sampled,
		theta_euclid_dist = theta_euclid_dist, mu_dist = mu_dist, A_dist = A_dist, phi_euclid_dist = phi_euclid_dist, Q_prob_dist = Q_prob_dist,
		num_cell_samples = num_cell_samples, num_gene_samples = num_gene_samples, use_flat_model = use_flat_model,
		use_nb = use_nb, log_mean_log_disp_coef = log_mean_log_disp_coef, rsample = rsample,
		batch_indicator_mat = batch_indicator_mat, batch_indicator_effect_mat = batch_indicator_effect_mat, use_is_cycler_indicators = use_is_cycler_indicators)




	# --- COMPUTE EXPECTATION OF LOG LIKELIHOOD W.R.T CELLS OR GENES ---
	if exp_over_cells:
		expectation_log_likelihood = torch.mean(torch.sum(log_likelihood_sampled,dim=0).view(-1, log_likelihood_sampled.shape[2] * log_likelihood_sampled.shape[3]),dim=1)
	else:
		expectation_log_likelihood = torch.mean(torch.sum(log_likelihood_sampled,dim=1).view(-1, log_likelihood_sampled.shape[2] * log_likelihood_sampled.shape[3]),dim=1)



	return expectation_log_likelihood



def compute_expectation_log_likelihood(gene_X, log_L,
		theta_sampled, mu_loc, A_loc = None, phi_euclid_loc = None, Q_prob_loc = None,
		use_is_cycler_indicators = False, exp_over_cells = True, use_flat_model = False,
		use_nb = False, log_mean_log_disp_coef = None, batch_indicator_mat = None, B_loc = None, rsample = True):

	




	# --- COMPUTE LOG LIKELIHOOD OVER SAMPLES CONDITIONAL ON Q = 0 ---
	Q_0_log_likelihood_sampled = compute_sample_log_likelihood(gene_X, log_L,
		theta_sampled = theta_sampled,
		mu_sampled = mu_loc.reshape(1,-1),
		A_sampled = None if A_loc is None else A_loc.reshape(1,-1),
		phi_sampled = None if phi_euclid_loc is None else torch.atan2(phi_euclid_loc[:,1],phi_euclid_loc[:,0]).reshape(1,-1),
		Q_sampled = torch.zeros((1,gene_X.shape[1])),
		use_flat_model = False,
		use_nb = use_nb,log_mean_log_disp_coef = log_mean_log_disp_coef, rsample = rsample,
		batch_indicator_mat = batch_indicator_mat, batch_indicator_effect_mat = B_loc, use_is_cycler_indicators = True)



	# --- COMPUTE LOG LIKELIHOOD OVER SAMPLES CONDITIONAL ON Q = 1 ---
	Q_1_log_likelihood_sampled = compute_sample_log_likelihood(gene_X, log_L,
		theta_sampled = theta_sampled,
		mu_sampled = mu_loc.reshape(1,-1),
		A_sampled = None if A_loc is None else A_loc.reshape(1,-1),
		phi_sampled = None if phi_euclid_loc is None else torch.atan2(phi_euclid_loc[:,1],phi_euclid_loc[:,0]).reshape(1,-1),
		Q_sampled = torch.ones((1,gene_X.shape[1])),
		use_flat_model = False,
		use_nb = use_nb,log_mean_log_disp_coef = log_mean_log_disp_coef, rsample = rsample,
		batch_indicator_mat = batch_indicator_mat, batch_indicator_effect_mat = B_loc, use_is_cycler_indicators = True)



	# --- WEIGH THE SAMPLED LL BASED ON Q_LOC ---
	Q_prob_loc_reshaped = Q_prob_loc.unsqueeze(1).unsqueeze(2).unsqueeze(3)
	log_likelihood_sampled = (Q_prob_loc_reshaped * Q_1_log_likelihood_sampled) + ((1.0 - Q_prob_loc_reshaped) * Q_0_log_likelihood_sampled)



	# --- COMPUTE EXPECTATION OF LOG LIKELIHOOD W.R.T CELLS OR GENES ---
	if exp_over_cells:
		expectation_log_likelihood = torch.mean(torch.sum(log_likelihood_sampled,dim=0).view(-1, log_likelihood_sampled.shape[2] * log_likelihood_sampled.shape[3]),dim=1)
	else:
		expectation_log_likelihood = torch.mean(torch.sum(log_likelihood_sampled,dim=1).view(-1, log_likelihood_sampled.shape[2] * log_likelihood_sampled.shape[3]),dim=1)



	return expectation_log_likelihood



# - Description: given lists of corresponding variational and prior distributions, 
# computes the KL divergence among distribution pairs and then computes the sum of the KL's
# - Notes:
#		- Assumes parameters are independent in distribution lists
# - Parameters:
#		- variational_dist_list (list of PyTorch distributions): variational distribution list
#		- prior_dist_list (list of PyTorch distributions): prior distribution list
# - Outputs:
#		- divergence ([num_cells] or [num_genes] PyTorch Tensor): sum of KL divergences for each gene / cell distribution


# assumes variables are independent in the distribution lists
# returns: [num_cells] or [num_genes] length vector describing each cell/gene's divergence
def compute_divergence(variational_dist_list, prior_dist_list):
	
	# ** check that the lists are of the same length **
	if len(variational_dist_list) != len(prior_dist_list):
		raise Exception("Error in compute_divergence fn: variational and prior distribution lists are not of same length.")
	
	# ** compute the divergences between each pair of distributions **
	pair_divergence_list = []
	for variational_dist, prior_dist in zip(variational_dist_list, prior_dist_list):

		# compute divergence
		pair_divergence_vec = torch.distributions.kl_divergence(variational_dist, prior_dist)

		# add
		pair_divergence_list.append(pair_divergence_vec)
		
	# ** stack: [num_distributions x num_cells] **
	pair_divergence = torch.stack(pair_divergence_list,dim=0)
		
	# ** compute the divergence for each cell or gene by summing over the individual divergences **
	divergence = torch.sum(pair_divergence, dim = 0)

	return divergence









# - Description: computes the KL divergence between PyTorch PowerSpherical and Von Mises Fisher
#	distributions according to https://arxiv.org/abs/2006.04437
from torch.distributions.kl import register_kl
@register_kl(power_spherical.PowerSpherical, hyperspherical_vae.distributions.von_mises_fisher.VonMisesFisher)
def compute_kl_between_powerspherical_and_vmf(powerspherical_dist, vmf_dist):
	
	# term 1
	term_1 = -1 * powerspherical_dist.entropy()

	# term 2
	term_2 = vmf_dist._log_normalization()

	# term 3
	term_3 = vmf_dist.scale.flatten() * torch.sum((powerspherical_dist.loc * vmf_dist.loc),dim=1)

	# term 4
	d = powerspherical_dist.loc.shape[1]
	alpha = (d - 1) / 2 + powerspherical_dist.scale
	beta = (d - 1) / 2
	term_4 = (alpha - beta) / (alpha + beta)

	# put it all together
	kl = term_1 + term_2 - (term_3 * term_4)
	
	return kl



# - Description: computes variational inference objective loss
# - Parameters:
#		- gene_param_dict (dictionary of PyTorch Tensors): dictionary holding the current values of the gene parameter variational distributions
#		- cell_param_dict (dictionary of PyTorch Tensors): dictionary holding the current values of the cell parameter variational distributions
#		- gene_prior_dict (dictionary of PyTorch Tensors): dictionary holding the values of the gene parameter prior distributions
#		- cell_prior_dict (dictionary of PyTorch Tensors): dictionary holding the values of the cell parameter prior distributions
#		- gene_X ([num_cells x num_genes] Tensor of ints): gene count matrix 
#		- log_L ([num_cells] Tensor floats): natural log of cell library sizes
#		- num_cell_samples (int): number of samples to use for cell distributions for monte carlo estimate of log likelihood
#		- num_gene_samples (int): number of samples to use for gene distributions for monte carlo estimate of log likelihood
#		- max_amp (float): maximum possible amplitude for genes
#		- min_amp (float): minimum possible amplitude for genes
#		- use_cell_loss (bool): if True, computes the monte carlo estimate of the LL over cells, otherwise genes
#		- use_nb (bool): if true, uses Negative Binomial likelihood model, otherwise uses Poisson
#		- mean_disp_coef (numpy array of floats): if use_nb = True, describes the log mean - log dispersion relationship shared by all genes
# - Outputs:
#		- variational_and_prior_kl ([num_genes] or [num_cells] PyTorch Tensor of floats): sum of KL divergence for each gene / cell's variational distributions w/ priors
#		- expectation_log_likelihood ([num_genes] or [num_cells] PyTorch Tensor of floats): monte carlo estimates of the gene / cell log likelihoods
def compute_loss(gene_X,
				log_L,
				gene_param_dict,
				theta_sampled,
				gene_prior_dict,
				max_amp,
				min_amp,
				num_gene_samples,
				exp_over_cells = True,
				use_flat_model = False,
				use_nb = False,
				log_mean_log_disp_coef = None,
				batch_indicator_mat = None):
	



	# ** get distribution dict **
	distrib_dict = utils.init_distributions_from_param_dicts(gene_param_dict = gene_param_dict, gene_prior_dict = gene_prior_dict, max_amp = max_amp, min_amp = min_amp)


	# ** compute the expectation of the gene log likelihood **
	gene_param_loc_scale_dict = utils.get_distribution_loc_and_scale(gene_param_dict=gene_param_dict, min_amp = min_amp, max_amp = max_amp, prep = True)
	# expectation_log_likelihood = compute_expectation_log_likelihood(gene_X, log_L,
	# 	theta_sampled = theta_sampled, mu_loc = gene_param_loc_scale_dict['mu_loc'], A_loc = gene_param_loc_scale_dict['A_loc'],
	# 	phi_euclid_loc = gene_param_loc_scale_dict['phi_euclid_loc'], Q_prob_loc = gene_param_loc_scale_dict['Q_prob_loc'],
	# 	use_is_cycler_indicators = distrib_dict['Q_prob'] is not None, exp_over_cells = exp_over_cells, use_flat_model = use_flat_model,
	# 	use_nb = use_nb, log_mean_log_disp_coef = log_mean_log_disp_coef, batch_indicator_mat = batch_indicator_mat, B_loc = None, rsample = True)
	expectation_log_likelihood = compute_mc_expectation_log_likelihood(gene_X, log_L,
		theta_sampled = theta_sampled,
		mu_dist = distrib_dict['mu'], A_dist = distrib_dict['A'], phi_euclid_dist = distrib_dict['phi_euclid'], Q_prob_dist = distrib_dict['Q_prob'],
		num_gene_samples = num_gene_samples, exp_over_cells = exp_over_cells, use_flat_model = use_flat_model,
		use_nb = use_nb, log_mean_log_disp_coef = log_mean_log_disp_coef, rsample = True, batch_indicator_mat = batch_indicator_mat, batch_indicator_effect_mat = None, use_is_cycler_indicators = distrib_dict['Q_prob'] is not None)








	# -- compute the kl between the variational posterior and the prior --

	# ** get variational and prior dist lists **
	variational_dist_list = [distrib_dict['mu']]
	prior_dist_list = [distrib_dict['prior_mu']]
	if not use_flat_model:
		variational_dist_list += [distrib_dict['A'],distrib_dict['phi_euclid']]
		prior_dist_list += [distrib_dict['prior_A'],distrib_dict['prior_phi_euclid']]
	if 'Q_prob' in distrib_dict and 'prior_Q_prob' in distrib_dict:
		variational_dist_list += [distrib_dict['Q_prob']]
		prior_dist_list += [distrib_dict['prior_Q_prob']]
	else:
		raise Exception("Error: use_is_cycler_indicators = True, but Q_prob or prior_Q_prob not found in distrib_dict")




	# ** compute the divegence **
	variational_and_prior_kl = compute_divergence(variational_dist_list = variational_dist_list,
		prior_dist_list = prior_dist_list)




	return variational_and_prior_kl, expectation_log_likelihood







