import torch

import power_spherical
import os
import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader


# tempo imports
from . import data_loader
from . import cell_posterior
from . import objective_functions
from . import utils




# --- TRAIN UTIL FUNCTIONIS ---

# - general description:
#	- given a training run folder:
#		- for each run, identifies the epoch with the smallest lost and associated parameters
#		- of all the runs (and their associated optimal parameters), identifies the run with the lowest loss and returns associated parameters
# - parameters:
#	- run_folder (str): path to the run folder
#	- use_cell_loss (bool): whether or not to use cell or gene loss to identify the best run
# - returns:
# 	- optimal_cell_param_dict (dictionary of Tensors): dictionary holding optimal cell parameters
# 	- optimal_gene_param_dict (dictionary of Tensors): dictionary holding optimal gene parameters
#	- optimal_subrun_num (int): index of the subrun with the minimum loss
#	- optimal_loss (float): minimum loss value
#	- subrun_best_losses (list of floats): best losses for each subrun
#	- subrun_loss_list (list of list of floats): the full loss list

def get_best_gene_params_across_all_em_steps(em_res_folder):

	# ** get subrun folders **
	em_step_gene_folders = os.listdir("%s/gene_subruns" % em_res_folder)
	em_step_gene_folders = list(filter(lambda x: "." not in x, em_step_gene_folders))
	em_step_gene_folders = list(map(lambda x: "%s/gene_subruns/%s" % (em_res_folder, x), em_step_gene_folders))

	
	# ** sort **
	em_step_gene_folders = sorted(em_step_gene_folders)


	# ** for each, get their best parameter and corresponding loss **
	em_step_gene_param_dicts, em_step_best_losses, em_step_loss_list = [], [], [], []
	for em_step_gene_folder in em_step_gene_folders:

		# get best params from the subrun
		gene_param_dict, min_loss_epoch, min_loss, losses = get_best_gene_params_from_em_step(em_step_gene_folder)

		# add
		em_step_gene_param_dicts.append(gene_param_dict)
		em_step_best_losses.append(min_subrun_loss)
		em_step_loss_list.append(subrun_losses)

	# ** choose the subrun with the lowest loss **
	em_step_best_losses = np.array(em_step_best_losses)
	optimal_em_step_num = np.argmin(em_step_best_losses)
	optimal_loss = em_step_best_losses[optimal_em_step_num]

	# ** get corresponding parameters **
	optimal_gene_param_dict = subrun_gene_param_dicts[optimal_em_step_num]


	return optimal_gene_param_dict, optimal_em_step_num, optimal_loss, em_step_best_losses, em_step_loss_list




def get_best_gene_params_from_em_step(em_step_gene_folder):

	# ** load the loss **
	with open("%s/loss.txt" % em_step_gene_folder) as file_obj:
		losses = list(map(lambda x: float(x.replace("\n", "")), file_obj.readlines()))
		losses = np.array(losses)

	# ** choose the epoch w/ the lowest loss **
	min_loss_epoch = np.argmin(losses)
	min_loss = losses[min_loss_epoch]

	# ** load parameters at min_loss_epoch **
	param_dict = torch.load("%s/params/%s.pt" % (subrun_folder, min_loss_epoch))

	# ** make the gene param dict **

	# init
	gene_param_dict = {}

	# fill gene param dict
	for gene_param in ['mu_loc', 'mu_log_scale', 'A_log_alpha', 'A_log_beta', 'phi_euclid_loc', 'phi_log_scale', 'Q_prob_log_alpha', 'Q_prob_log_beta']:
		if gene_param in param_dict:
			gene_param_dict[gene_param] = param_dict[gene_param]


	return gene_param_dict, min_loss_epoch, min_loss, losses



def gene_fit(gene_X, 
	log_L, 
	gene_param_dict, 
	gene_prior_dict,
	folder_out,
	learning_rate_dict,
	theta_posterior_likelihood = None,
	theta = None,
	theta_euclid_loc = None,
	theta_scale = None,
	gene_param_grad_dict = None,
	max_iters = 200, 
	num_cell_samples = 5,
	num_gene_samples = 5,
	max_amp = 1.0 / np.log10(np.e),
	min_amp = 0.2 / np.log10(np.e),
	print_epoch_loss = False,
	improvement_window = 10,
	convergence_criterion = 1e-3,
	lr_scheduler_patience = 3,
	lr_scheduler_factor = 0.1,
	use_flat_model = False,
	batch_size = 3000,
	num_workers = 0,
	pin_memory = True,
	use_nb = False,
	log_mean_log_disp_coef = None,
	batch_indicator_mat = None,
	detect_anomaly = False,
	expectation_point_est_only = False,
	**kwargs):


	

	# ** turn on anomaly detection if specified **
	torch.autograd.set_detect_anomaly(detect_anomaly)

	# ** make folder out and subfolder **
	if not os.path.exists(folder_out):
		os.makedirs(folder_out)
	if not os.path.exists("%s/params" % folder_out):
		os.makedirs("%s/params" % (folder_out))


	# ** turn on/off relevant gradient computation for the gene parameters **
	if gene_param_grad_dict is not None:
		for gene_param_key, gene_param in gene_param_dict.items():
			gene_param.requires_grad = gene_param_grad_dict[gene_param_key]


	# ** create dataset for the cell parameters / counts etc. **
	dataset = data_loader.TempoDataset(X = gene_X,
							   log_L = log_L,
							   theta_posterior_likelihood = theta_posterior_likelihood,
							   theta_euclid_loc = theta_euclid_loc,
							   theta_scale = theta_scale,
							   theta = theta,
							   batch_indicator_mat = batch_indicator_mat)




	# ** create data loader **
	training_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory = pin_memory, collate_fn = data_loader.tempo_collate)


	# ** make the learning rate dict list **
	learning_rate_dict_list = []
	for key, param in gene_param_dict.items():
			learning_rate_dict_list.append({"params" : [param], "lr" : learning_rate_dict[key]})



	# ** init optimizer **
	optimizer = torch.optim.Adam(learning_rate_dict_list)


	# ** init scheduler **
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = lr_scheduler_factor, patience = lr_scheduler_patience)
		

		
	# ** run **
	losses, kl_losses, ll_losses = [], [], []
	loss_percent_differences = []
	for epoch in range(0,max_iters):

		
		
		# init total loss, kl_loss, and ll_loss
		total_loss, kl_loss, ll_loss = 0, 0, 0
		num_batches = 0
		for batch_data in training_dataloader:


			# zero gradient
			optimizer.zero_grad()

			# sample theta: [num_cells x num_samples]
			if batch_data['theta'] is not None:
				theta_sampled = batch_data['theta'].reshape(-1,1)
			else:
				if batch_data['theta_posterior_likelihood'] is not None:
					theta_dist = cell_posterior.ThetaPosteriorDist(theta_posterior_likelihood = batch_data['theta_posterior_likelihood'])
					theta_sampled = theta_dist.sample(num_cell_samples)
				elif batch_data['theta_euclid_loc'] is not None and batch_data['theta_scale'] is not None:
					theta_dist = power_spherical.PowerSpherical(batch_data['theta_euclid_loc'],batch_data['theta_scale'])
					theta_euclid_sampled = theta_dist.sample(num_cell_samples)
					theta_sampled = torch.atan2(theta_euclid_sampled[:,1],theta_euclid_sampled[:,0])


			# compute individual cell or gene losses
			variational_and_prior_kl, expectation_log_likelihood = objective_functions.compute_loss(gene_param_dict = gene_param_dict,
							theta_sampled = theta_sampled,
							gene_prior_dict = gene_prior_dict,
							gene_X = batch_data['X'],
							log_L = batch_data['log_L'],
							num_gene_samples = num_gene_samples,
							max_amp = max_amp,
							min_amp = min_amp,
							exp_over_cells = False,
							use_flat_model = use_flat_model,
							use_nb = use_nb,
							log_mean_log_disp_coef = log_mean_log_disp_coef,
							batch_indicator_mat = batch_data['batch_indicator_mat'],
							expectation_point_est_only = expectation_point_est_only)



			# compute the batch total cell or gene loss
			batch_total_loss = torch.sum(variational_and_prior_kl - expectation_log_likelihood)
			batch_kl_loss = torch.sum(variational_and_prior_kl)
			batch_ll_prop_loss = torch.sum(expectation_log_likelihood)
			num_batches += 1

			# update the totals
			total_loss += batch_total_loss.item()
			kl_loss += batch_kl_loss.item()
			ll_loss += batch_ll_prop_loss.item()

			# pass gradients backward
			batch_total_loss.backward(retain_graph=False)

			# step
			optimizer.step()



		# compute estimate of losses (mean of batch estimates)
		total_loss = total_loss / num_batches
		kl_loss = kl_loss / num_batches
		ll_loss = ll_loss / num_batches

		# print loss at epoch if specified
		if print_epoch_loss:
			print("iter: %s; ELBO: %s; E[LL]: %s; KL: %s" % (epoch, str(total_loss), str(ll_loss), str(kl_loss)))


		# ** add epoch loss to list **
		losses.append(total_loss)
		kl_losses.append(kl_loss)
		ll_losses.append(ll_loss)

		# ** update the LR schedule accordingly **
		scheduler.step(total_loss)

		# ** write parameters at epoch to file **
		fileout = "%s/params/%s.pt" % (folder_out,epoch)
		param_dict = gene_param_dict
		torch.save(param_dict, fileout)
		

		# ** calculate loss percentage improvement **
		if epoch >= 2 * improvement_window:
			loss_percent_improvement =  1 - (np.mean(losses[epoch-improvement_window:]) / np.mean(losses[epoch - (2 * improvement_window):epoch-improvement_window]))
			loss_percent_differences.append(loss_percent_improvement)
			if print_epoch_loss:
				print("Improvement: %s" % loss_percent_improvement)
			
		# ** check if we should stop training **
		if (epoch >= 2 * improvement_window) and (loss_percent_improvement <= convergence_criterion):
			print("Loss converged and stopping")
			break


	# ** write the loss list out **

	# total loss
	fileout = '%s/loss.txt' % (folder_out)
	losses = list(map(lambda x: str(x), losses))
	with open(fileout, "wb") as file_obj:
		file_obj.write("\n".join(losses).encode())

	# KL loss
	fileout = '%s/kl_loss.txt' % (folder_out)
	kl_losses = list(map(lambda x: str(x), kl_losses))
	with open(fileout, "wb") as file_obj:
		file_obj.write("\n".join(kl_losses).encode())

	# log prop loss
	fileout = '%s/ll_loss.txt' % (folder_out)
	ll_losses = list(map(lambda x: str(x), ll_losses))
	with open(fileout, "wb") as file_obj:
		file_obj.write("\n".join(ll_losses).encode())
		



	# ** get optimal gene parameters and write out **
	opt_gene_param_dict_unprepped = torch.load('%s/params/%s.pt' % (folder_out, np.argmin(losses)))
	opt_gene_param_dict_prepped = utils.prep_gene_params(opt_gene_param_dict_unprepped)
	torch.save(opt_gene_param_dict_unprepped, '%s/optimal_gene_params_unprepped.pt' % (folder_out))
	torch.save(opt_gene_param_dict_prepped, '%s/optimal_gene_params_prepped.pt' % (folder_out))



	return optimizer, opt_gene_param_dict_unprepped



























