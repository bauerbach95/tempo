
import torch


# tempo imports
from . import compute_cell_posterior
from . import utils
from . import cell_posterior
from . import objective_functions





class ClockGenePosterior(torch.nn.Module):

	def __init__(self,gene_param_dict,gene_prior_dict,num_grid_points,clock_indices,use_nb=False,log_mean_log_disp_coef=None,min_amp=0,max_amp=2.5,use_clock_output_only=False):
		super(ClockGenePosterior, self).__init__()

		self.clock_indices = clock_indices
		self.gene_param_dict = gene_param_dict
		self.gene_prior_dict = gene_prior_dict
		self.num_grid_points = num_grid_points
		self.use_nb = use_nb
		self.log_mean_log_disp_coef = log_mean_log_disp_coef
		self.min_amp = min_amp
		self.max_amp = max_amp
		self.num_genes = self.gene_param_dict['mu_loc'].shape[0]
		self.use_clock_output_only = use_clock_output_only





	def compute_cell_phase_posterior_likelihood(self,gene_X,log_L,prior_theta_euclid_dist,num_gene_samples=5):


		# --- SAMPLE THE GENE PARAMETERS ---


		# ** get distribution dict **
		distrib_dict = utils.init_distributions_from_param_dicts(gene_param_dict = self.gene_param_dict, max_amp = self.max_amp, min_amp = self.min_amp, prep = True)



		# ** sample **
		mu_sampled = distrib_dict['mu'].rsample((num_gene_samples,)) # [num_gene_samples x num_genes]
		A_sampled = distrib_dict['A'].rsample((num_gene_samples,)) # [num_gene_samples x num_genes]
		phi_euclid_sampled = distrib_dict['phi_euclid'].rsample((num_gene_samples,)) # [num_gene_samples x num_genes x 2]
		phi_sampled = torch.atan2(phi_euclid_sampled[:,:,1],phi_euclid_sampled[:,:,0]) # [num_gene_samples x num_genes x 2]
		Q_sampled = utils.get_is_cycler_samples_from_dist(distrib_dict['Q_prob'],num_gene_samples=num_gene_samples,rsample=True)


		# --- COMPUTE CELL POSTERIOR ---
		theta_posterior_likelihood = compute_cell_posterior.compute_cell_posterior(gene_X = gene_X,
							   log_L = log_L,
							   num_grid_points = self.num_grid_points,
							   prior_theta_euclid_dist = prior_theta_euclid_dist, # self.prior_theta_euclid_dist
							   mu_sampled = mu_sampled,
							   A_sampled = A_sampled,
							   phi_sampled = phi_sampled,
							   Q_sampled = Q_sampled,
							   B_sampled = None,
							   use_nb = self.use_nb,
							   log_mean_log_disp_coef = self.log_mean_log_disp_coef)



		return theta_posterior_likelihood


	def get_clock_gene_param_dict(self):
		clock_gene_param_dict = {}
		for key in self.gene_param_dict:
			if key == 'phi_euclid_loc':
				clock_gene_param_dict[key] = self.gene_param_dict['phi_euclid_loc'][self.clock_indices,:]
			else:
				clock_gene_param_dict[key] = self.gene_param_dict[key][self.clock_indices]
		return clock_gene_param_dict



	def compute_loss(self,gene_X,log_L,prior_theta_euclid_dist,num_cell_samples,num_gene_samples):


		# --- COMPUTE THE CELL POSTERIOR DISTRIBUTION ---
		theta_posterior_likelihood = self.compute_cell_phase_posterior_likelihood(gene_X,log_L,prior_theta_euclid_dist,num_gene_samples)


		# --- SAMPLE THE CELL PHASE POSTERIOR ---
		theta_dist = cell_posterior.ThetaPosteriorDist(theta_posterior_likelihood)
		theta_sampled = theta_dist.rsample(num_cell_samples)




		# --- GET THE DISTRIB DICT AND CLOCK LOC SCALE DICT ---
		

		# ** get distribution dict **

		# input gene distrib dict
		input_distrib_dict = utils.init_distributions_from_param_dicts(gene_param_dict = self.gene_param_dict, gene_prior_dict = self.gene_prior_dict, max_amp = self.max_amp, min_amp = self.min_amp)
		
		# output gene distrib dict
		if self.use_clock_output_only:
			output_distrib_dict = utils.init_distributions_from_param_dicts(gene_param_dict = self.get_clock_gene_param_dict(), gene_prior_dict = self.gene_prior_dict, max_amp = self.max_amp, min_amp = self.min_amp)			
		else:
			output_distrib_dict = input_distrib_dict

		# --- COMPUTE THE EXPECTATION LOG LIKELIHOOD OF THE CYCLING GENES ---


		# subset gene_X and distrib_dict to core clock genes only if need to
		if self.use_clock_output_only:
			gene_X = gene_X[:,self.clock_indices]


		# # compute the expectation of the LL
		# cycler_gene_expectation_log_likelihood = objective_functions.compute_expectation_log_likelihood(gene_X = gene_X, log_L = log_L, theta_sampled = theta_sampled,
		# 	mu_dist = output_distrib_dict['mu'], A_dist = output_distrib_dict['A'], phi_euclid_dist = output_distrib_dict['phi_euclid'], Q_prob_dist = output_distrib_dict['Q_prob'],
		# 	num_gene_samples = num_gene_samples, exp_over_cells = False, use_flat_model = False,
		# 	use_nb = self.use_nb, log_mean_log_disp_coef = self.log_mean_log_disp_coef, rsample = True, use_is_cycler_indicators = output_distrib_dict['Q_prob'] is not None)
		# clock_gene_expectation_log_likelihood = cycler_gene_expectation_log_likelihood[self.clock_indices]




		# ** compute gene LL in each cell over all samples **
		cycler_log_likelihood_sampled = objective_functions.compute_sample_log_likelihood(gene_X, log_L,
			theta_sampled = theta_sampled,
			mu_dist = output_distrib_dict['mu'], A_dist = output_distrib_dict['A'], phi_euclid_dist = output_distrib_dict['phi_euclid'], Q_prob_dist = output_distrib_dict['Q_prob'],
			num_gene_samples = num_gene_samples, use_flat_model = False,
			use_nb = self.use_nb, log_mean_log_disp_coef = self.log_mean_log_disp_coef, rsample = True,
			use_is_cycler_indicators = output_distrib_dict['Q_prob'] is not None)




		# ** compute the MC expectations **

		# cycler
		cycler_mc_lls = torch.sum(torch.sum(cycler_log_likelihood_sampled,dim=0),dim=0).flatten()
		cycler_gene_expectation_log_likelihood = torch.mean(cycler_mc_lls)



		# clock
		clock_mc_lls = torch.sum(torch.sum(cycler_log_likelihood_sampled[self.clock_indices,:,:,:],dim=0),dim=0).flatten()
		clock_gene_expectation_log_likelihood = torch.mean(clock_mc_lls)



		# --- COMPUTE THE KL OF THE CORE CLOCK GENES AND THE DE NOVO CYCLERS ---

		# ** get variational and prior dist lists **
		variational_dist_list = [input_distrib_dict['mu'],input_distrib_dict['A'],input_distrib_dict['phi_euclid']]
		prior_dist_list = [input_distrib_dict['prior_mu'],input_distrib_dict['prior_A'],input_distrib_dict['prior_phi_euclid']]
		if 'Q_prob' in input_distrib_dict and 'prior_Q_prob' in input_distrib_dict:
			variational_dist_list += [input_distrib_dict['Q_prob']]
			prior_dist_list += [input_distrib_dict['prior_Q_prob']]



		# ** compute the divegence **
		clock_and_de_novo_cycler_kl = objective_functions.compute_divergence(variational_dist_list = variational_dist_list,
			prior_dist_list = prior_dist_list)



		# # --- COMPUTE ELBO ---
		# kl_loss = torch.mean(clock_and_de_novo_cycler_kl)
		# if self.use_clock_output_only:
		# 	ll_loss = torch.mean(clock_gene_expectation_log_likelihood)
		# else:
		# 	ll_loss = torch.mean(cycler_gene_expectation_log_likelihood)
		# elbo_loss = kl_loss - ll_loss

		kl_loss = torch.sum(clock_and_de_novo_cycler_kl)
		if self.use_clock_output_only:
			ll_loss = clock_gene_expectation_log_likelihood
		else:
			ll_loss = cycler_gene_expectation_log_likelihood
		elbo_loss = kl_loss - ll_loss


		return elbo_loss, ll_loss, kl_loss






