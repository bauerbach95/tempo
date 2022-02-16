import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import power_spherical



# tempo imports
from . import data_loader
from . import clock_gene_posterior
from . import cell_posterior
from . import utils
from . import cell_posterior
from . import objective_functions
from . import compute_cell_posterior
from . import gene_fit



def run(gene_X,
	clock_indices,
	log_L,
	gene_param_dict,
	gene_prior_dict,
	min_amp,
	max_amp,
	prior_theta_euclid_dist,
	folder_out,
	learning_rate_dict,
	gene_param_grad_dict = None,
	use_nb = True,
	log_mean_log_disp_coef = torch.Tensor(np.array([-4,-0.5])),
	num_grid_points = 24,
	num_cell_samples = 5,
	num_gene_samples = 5,
	vi_max_epochs = 300,
	vi_print_epoch_loss = True,
	vi_improvement_window = 10,
	vi_convergence_criterion = 1e-3,
	vi_lr_scheduler_patience = 40,
	vi_lr_scheduler_factor = 0.1,
	vi_batch_size = 3000,
	vi_num_workers = 0,
	vi_pin_memory = False,
	batch_indicator_mat = None,
	detect_anomaly = False,
	use_clock_output_only = False):
	


	

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
	if prior_theta_euclid_dist.__class__ == cell_posterior.ThetaPosteriorDist:
		dataset = data_loader.TempoDataset(X = gene_X,
			   log_L = log_L,
			   theta_posterior_likelihood = prior_theta_euclid_dist.theta_posterior_likelihood)
	elif prior_theta_euclid_dist.__class__ == power_spherical.distributions.PowerSpherical:
		dataset = data_loader.TempoDataset(X = gene_X,
					   log_L = log_L,
					   theta_euclid_loc = prior_theta_euclid_dist.loc,
					   theta_scale = prior_theta_euclid_dist.scale)
	elif prior_theta_euclid_dist.__class__ == power_spherical.distributions.HypersphericalUniform:
		dataset = data_loader.TempoDataset(X = gene_X,
					   log_L = log_L)
	else:
		raise Exception("Error: invalid prior cell theta distribution")


	# ** create data loader **
	training_dataloader = DataLoader(dataset, batch_size=vi_batch_size, shuffle=True, num_workers=vi_num_workers, pin_memory = vi_pin_memory, collate_fn = data_loader.tempo_collate)

	# ** make the learning rate dict list **
	learning_rate_dict_list = []
	for key, param in gene_param_dict.items():
			learning_rate_dict_list.append({"params" : [param], "lr" : learning_rate_dict[key]})



	# ** init optimizer **
	optimizer = torch.optim.Adam(learning_rate_dict_list)

	# ** make the ClockGenePosterior object **
	clock_gene_posterior_obj = clock_gene_posterior.ClockGenePosterior(gene_param_dict = gene_param_dict,
		gene_prior_dict = gene_prior_dict,
		num_grid_points = num_grid_points, # prior_theta_euclid_dist = prior_theta_euclid_dist,
		clock_indices = clock_indices,
		use_nb=use_nb,
		log_mean_log_disp_coef=log_mean_log_disp_coef,
		min_amp=min_amp,
		max_amp=max_amp,
		use_clock_output_only = use_clock_output_only)
		


	# ** init scheduler **
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor = vi_lr_scheduler_factor, patience = vi_lr_scheduler_patience)
		


	# ** run **
	losses, kl_losses, ll_losses = [], [], []
	loss_percent_differences = []
	for epoch in range(0,vi_max_epochs):


		# init total loss, kl_loss, and ll_loss
		total_loss, kl_loss, ll_loss = 0, 0, 0
		num_batches = 0
		for batch_data in training_dataloader:


			# zero gradient
			# optimizer.zero_grad()
			for key, param in gene_param_dict.items():
				gene_param_dict[key].grad = None


			# get the batch_prior_theta_euclid_dist
			if prior_theta_euclid_dist.__class__ == cell_posterior.ThetaPosteriorDist:
				batch_prior_theta_euclid_dist = cell_posterior.ThetaPosteriorDist(batch_data['theta_posterior_likelihood'])
			elif prior_theta_euclid_dist.__class__ == power_spherical.distributions.PowerSpherical:
				batch_prior_theta_euclid_dist = power_spherical.PowerSpherical(batch_data['theta_euclid_loc'],batch_data['theta_scale'])
			elif prior_theta_euclid_dist.__class__ == power_spherical.distributions.HypersphericalUniform:
				batch_prior_theta_euclid_dist = prior_theta_euclid_dist


			# compute the batch loss loss
			batch_total_loss, batch_ll_prop_loss, batch_kl_loss = clock_gene_posterior_obj.compute_loss(batch_data['X'],batch_data['log_L'],batch_prior_theta_euclid_dist,num_cell_samples,num_gene_samples)


			# update the batch count
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
		if vi_print_epoch_loss:
			print("iter: %s; ELBO: %s; E[LL]: %s; KL: %s" % (epoch, str(total_loss), str(ll_loss), str(kl_loss)))


		# ** add epoch loss to list **
		losses.append(total_loss)
		kl_losses.append(kl_loss)
		ll_losses.append(ll_loss)

		# ** update the LR schedule accordingly **
		scheduler.step(total_loss)

		# ** write parameters at epoch to file **
		fileout = "%s/params/%s.pt" % (folder_out,epoch)
		# param_dict = {**gene_param_dict, **cell_param_dict, }
		# param_dict = {**gene_param_dict, }
		param_dict = gene_param_dict
		torch.save(param_dict, fileout)

		

		# ** calculate loss percentage improvement **
		if epoch >= 2 * vi_improvement_window:
			loss_percent_improvement =  1 - (np.mean(losses[epoch-vi_improvement_window:]) / np.mean(losses[epoch - (2 * vi_improvement_window):epoch-vi_improvement_window]))
			loss_percent_differences.append(loss_percent_improvement)
			if vi_print_epoch_loss:
				print("Improvement: %s" % loss_percent_improvement)
			
		# ** check if we should stop training **
		if (epoch >= 2 * vi_improvement_window) and (loss_percent_improvement <= vi_convergence_criterion):
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


	# ** get optimal theta posterior **
	optimal_clock_gene_posterior_obj = clock_gene_posterior.ClockGenePosterior(gene_param_dict = opt_gene_param_dict_unprepped,
		gene_prior_dict = gene_prior_dict,
		num_grid_points = num_grid_points, # prior_theta_euclid_dist = prior_theta_euclid_dist,
		clock_indices = clock_indices,
		use_nb=use_nb,
		log_mean_log_disp_coef=log_mean_log_disp_coef,
		min_amp=min_amp,
		max_amp=max_amp,
		use_clock_output_only=use_clock_output_only)
		
	optimal_theta_posterior_likelihood = optimal_clock_gene_posterior_obj.compute_cell_phase_posterior_likelihood(gene_X,log_L,prior_theta_euclid_dist,num_gene_samples=num_gene_samples)



	return optimal_theta_posterior_likelihood, opt_gene_param_dict_unprepped











