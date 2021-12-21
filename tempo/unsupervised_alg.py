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




def run(adata,
	folder_out,
	bulk_cycler_info_path,
	core_clock_gene_path,
	reference_gene = 'Arntl',
	min_gene_prop = 1e-5,
	min_amp = 0.0, # ** prep parameters **
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
	use_noninformative_phase_prior = True,
	use_nb = True,
	mean_disp_init_coef = [-4,-0.2], # ** mean / disp relationship learning parameters **
	mean_disp_log10_prop_bin_marks = list(np.linspace(-5,-1,20)), 
	mean_disp_max_num_genes_per_bin = 50,
	hv_std_residual_threshold = 0.5, # ** HVG selection parameters **
	mu_loc_lr = 1e-1, # ** conditional posterior opt parameters **
	mu_log_scale_lr = 1e-1,
	A_log_alpha_lr = 1e-1,
	A_log_beta_lr = 1e-1,
	phi_euclid_loc_lr = 5.0,
	phi_log_scale_lr = 5.0,
	Q_prob_log_alpha_lr = 1e-1,
	Q_prob_log_beta_lr = 1e-1,
	num_phase_grid_points = 24,
	num_cell_samples = 5,
	num_gene_samples = 5,
	vi_max_epochs = 300,
	vi_print_epoch_loss = True,
	vi_improvement_window = 10,
	vi_convergence_criterion = 1e-3,
	vi_lr_scheduler_patience = 10,
	vi_lr_scheduler_factor = 0.1,
	vi_batch_size = 3000,
	vi_num_workers = 0,
	vi_pin_memory = False,
	batch_indicator_mat = None,
	test_mode = False,
	null_percentile_threshold = 90.0,
	num_null_shuffles = 5,
	use_clock_input_only = False,
	use_clock_output_only = True,
	frac_pos_cycler_samples_threshold = 0.1,
	A_loc_pearson_residual_threshold = 0.5,
	confident_cell_interval_size_threshold = 12.0,
	**kwargs):



	# --- MAKE FOLDER OUTS ---

	# ** main folder **
	if not os.path.exists(folder_out):
		os.makedirs(folder_out)
		

	# ** mean_disp_param_folder_out folder **
	mean_disp_param_folder_out = "%s/mean_disp_param" % folder_out
	if not os.path.exists(mean_disp_param_folder_out):
		os.makedirs(mean_disp_param_folder_out)


	# # ** null_fits_folder_out **
	null_fits_folder_out = '%s/null_dist' % folder_out
	if not os.path.exists(null_fits_folder_out):
		os.makedirs(null_fits_folder_out)



	# --- GET THE CONFIG DICT AND WRITE ---

	# ** get **
	config_dict = locals()
	del config_dict['adata']

	# ** write
	config_path = "%s/config.txt" % folder_out
	with open(config_path, "wb") as file_obj:
		file_obj.write(str(config_dict).encode())



	# --- MAKE THE LR DICT ---
	vi_gene_param_lr_dict = {
		"mu_loc" : mu_loc_lr,
		"mu_log_scale" : mu_log_scale_lr,
		"A_log_alpha" : A_log_alpha_lr,
		"A_log_beta" : A_log_beta_lr,
		"phi_euclid_loc" : phi_euclid_loc_lr,
		"phi_log_scale" : phi_log_scale_lr,
		'Q_prob_log_alpha' : Q_prob_log_alpha_lr,
		'Q_prob_log_beta' : Q_prob_log_beta_lr
	}


	# --- TURN ON / OFF TEST MODE ---
	if test_mode:
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(0)   
		random.seed(0)
		detect_anomaly = True
		config_dict['detect_anomaly'] = detect_anomaly
	else:
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True
		detect_anomaly = False
		config_dict['detect_anomaly'] = detect_anomaly




	


	# --- GET CORE CLOCK GENES ---
	core_clock_genes = list(pd.read_table(core_clock_gene_path,header=None).iloc[:,0])
	core_clock_genes = list(filter(lambda x: x in adata.var_names, core_clock_genes))
	core_clock_genes = np.array(list(sorted(core_clock_genes)))


	# --- INITIALIZE is_clock and is_cycler FOR ALL GENES TO FALSE OTHER THAN THE CORE CLOCK GENES ---
	adata.var['is_clock'] = False
	adata.var.loc[core_clock_genes,'is_clock'] = True
	adata.var['is_cycler'] = False
	adata.var.loc[core_clock_genes,'is_cycler'] = True


	# --- MAKE AN INITIAL GUESS ABOUT PARAMS FOR MEAN / DISPERSION RELATIONSHIP ---

	print("--- ESTIMATING GLOBAL MEAN-DISPERSION RELATIONSHIP FOR NEGATIVE BINOMIAL ---")

	# ** get the log mean - log disp polynomial coefficients **
	if use_nb:
		log_mean_log_disp_coef = estimate_mean_disp_relationship.estimate_mean_disp_relationship(adata, mean_disp_init_coef, mean_disp_log10_prop_bin_marks, mean_disp_max_num_genes_per_bin,min_log_disp=-10,max_log_disp=10)
	else:
		log_mean_log_disp_coef = np.array([1e-1000]) # i.e. use Poisson (mean equals the variance; if log_disp = 1e-1000, np.exp(log_disp) ~ 0)
	log_mean_log_disp_coef = torch.Tensor(log_mean_log_disp_coef)

	# ** write out **
	mean_disp_fit_coef_fileout = '%s/log_mean_log_disp_poly_coef_0.txt' % (mean_disp_param_folder_out)
	np.savetxt(mean_disp_fit_coef_fileout,log_mean_log_disp_coef.detach().numpy())



	# --- RESTRICT ADATA TO ONLY GENES WITH PROP >= MIN_GENE_PROP / CORE CLOCK ---

	print("--- RESTRICTING CANDIDATE HIGHLY VARIABLE GENES BASED ON MINIMUM PROPORTION IN PSEUDOBULK ---")

	adata = adata[:,(adata.var['prop'] >= min_gene_prop) | (np.isin(adata.var_names, core_clock_genes))]

	print("Adata shape after thresholding minimum proportion")
	print(str(adata.shape))


	# --- CALL CANDIDATE HV GENES ---

	print("--- IDENTIFYING HIGHLY VARIABLE GENES BASED ON MEAN / VARIANCE RELATIONSHIP ---")

	hv_genes, pearson_residuals, log1p_prop_mean, log1p_prop_var, hv_gene_indices = hvg_selection.get_hv_genes_kernel(adata,std_residual_threshold=hv_std_residual_threshold,viz=False,bw=0.1,pseudocount=1)
	hv_genes = np.setdiff1d(np.array(hv_genes),np.array(core_clock_genes)) # make sure the core clock genes are not in there
	hv_genes = np.array(sorted(list(hv_genes)))
	adata.var['is_hv'] = False
	adata.var.loc[hv_genes,'is_hv'] = True



	# --- RESTRICT ADATA TO THOSE THAT ARE CORE CLOCK OR CANDIDATE HV GENES ---
	adata = adata[:,(np.isin(adata.var_names, hv_genes)) | (np.isin(adata.var_names, core_clock_genes))]



	# --- ORDER ADATA S.T. CORE CLOCK GENES FIRST AND THEN HV GENES SECOND ---
	adata = adata[:,list(core_clock_genes) + list(hv_genes)]



	print("Num HV genes: %s" % str(len(hv_genes)))
	print("Adata shape before starting algorithm: %s" % str(adata.shape))




	# --- GENERATE THE NULL DISTRIBUTION ---

	print("--- GENERATING NULL DISTRIBUTION OF DATA EVIDENCE UNDER RANDOM ORDERINGS ---")

	# ** get the clock adata **
	clock_adata = adata[:,adata.var['is_clock']]

	# ** get clock_X **
	clock_X, _, _, _, _ = prep.unsupervised_prep(clock_adata,**config_dict)

	# ** run **
	null_ll_vec = generate_null_dist.generate(adata = clock_adata, null_head_folder_out=null_fits_folder_out,learning_rate_dict=vi_gene_param_lr_dict,
		log_mean_log_disp_coef=log_mean_log_disp_coef,min_amp=min_amp,max_amp=max_amp,num_gene_samples=num_gene_samples,use_nb=use_nb,num_shuffles=num_null_shuffles,config_dict=copy.deepcopy(config_dict))










	# --- START ALGORTHM ---

	algorithm_step = 0
	prev_evidence = None
	alg_result_head_folder = "%s/alg_steps" % folder_out
	if not os.path.exists(alg_result_head_folder):
		os.makedirs(alg_result_head_folder)
	while True:


		print("--- STARTING ALGORITHM ITERATION %s --- " % algorithm_step)

		# --- MAKE THE SUBFOLDER FOR THE DE NOVO STEP ---
		alg_step_subfolder = "%s/%s" % (alg_result_head_folder, algorithm_step)
		if not os.path.exists(alg_step_subfolder):
			os.makedirs(alg_step_subfolder)




		# --- ESTIMATE THE CELL PHASE GIVEN CURRENT KNOWN CYCLING GENES ---



		print("--- ESTIMATING CELL PHASE POSTERIOR USING CURRENT CYCLERS ---")


		# ** get cycler adata **
		cycler_adata = adata[:,adata.var['is_cycler']]

		# ** initialize variational parameters to those from previous runs for clock and confident HVG **
		if algorithm_step >= 1:

			# previous alg step subfolder
			previous_alg_step_subfolder = "%s/%s" % (alg_result_head_folder, algorithm_step - 1)

			# load gene param df for previous round's cycler genes
			previous_cycler_gene_param_df_fileout = '%s/cell_phase_estimation/gene_prior_and_posterior.tsv' % previous_alg_step_subfolder
			previous_cycler_gene_param_df = pd.read_table(previous_cycler_gene_param_df_fileout,sep='\t',index_col='gene')

			# load the gene param df for the previous round's de novo cyclers
			previous_de_novo_cycler_gene_param_df_fileout = '%s/de_novo_cycler_id/gene_prior_and_posterior.tsv' % previous_alg_step_subfolder
			previous_de_novo_cycler_gene_param_df = pd.read_table(previous_de_novo_cycler_gene_param_df_fileout,sep='\t',index_col='gene')
			previous_de_novo_cyclers = np.intersect1d(np.array(cycler_adata.var_names), np.array(previous_de_novo_cycler_gene_param_df.index))
			previous_de_novo_cycler_gene_param_df = previous_de_novo_cycler_gene_param_df.loc[previous_de_novo_cyclers]

			# concat previous cycler and de novo cycler param df's
			current_cycler_gene_param_df =  pd.concat((previous_cycler_gene_param_df, previous_de_novo_cycler_gene_param_df))

			# filter cols relevant for initializing variational parameters
			cols_to_keep = list(filter(lambda x: "prior" not in x, current_cycler_gene_param_df.columns)) # drop prior columns
			current_cycler_gene_param_df = current_cycler_gene_param_df[cols_to_keep]

			# make sure current_cycler_gene_param_df is in the same order as cycler_adata
			current_cycler_gene_param_df = current_cycler_gene_param_df.loc[np.array(cycler_adata.var_names)] 

			# add parameters to adata to initialize cycler genes 
			for col in cols_to_keep:
				cycler_adata.var[col] = np.array(current_cycler_gene_param_df[col])




		# ** do the prep **
		cycler_gene_X, log_L, cycler_gene_param_dict, cell_prior_dict, cycler_gene_prior_dict = prep.unsupervised_prep(cycler_adata,**config_dict)
		if use_clock_output_only:
			clock_indices = np.where(cycler_adata.var['is_clock'])[0]
		else:
			clock_indices = np.arange(0,cycler_adata.shape[1])
		# non_clock_indices = np.setdiff1d(np.arange(0,cycler_adata.shape[1]), clock_indices)
		non_clock_indices = np.setdiff1d(np.arange(0,cycler_adata.shape[1]), np.where(cycler_adata.var['is_clock'])[0])
		cycler_gene_prior_dict['prior_Q_prob_alpha'][non_clock_indices] = torch.ones((non_clock_indices.shape[0])) # set the non-clock cycler Q priors to flat
		# cycler_gene_prior_dict['prior_Q_prob_alpha'][non_clock_indices] = 3 * torch.ones((non_clock_indices.shape[0])) # set the non-clock cycler Q priors to flat
		cycler_gene_prior_dict['prior_Q_prob_beta'][non_clock_indices] = torch.ones((non_clock_indices.shape[0]))


		# ** get prior theta euclid dist **
		prior_theta_euclid_dist = utils.init_distributions_from_param_dicts(cell_prior_dict = cell_prior_dict)['prior_theta_euclid']


		print("BEFORE")
		print("printing gene param df Q prob A")
		print(cycler_gene_prior_dict['prior_Q_prob_alpha'])
		print("printing gene param df Q prob B")
		print(cycler_gene_prior_dict['prior_Q_prob_beta'])



		# ** run **
		opt_cycler_theta_posterior_likelihood, opt_cycler_gene_param_dict_unprepped = clock_posterior_opt.run(gene_X = cycler_gene_X,
			clock_indices = clock_indices,
			log_L = log_L,
			gene_param_dict = cycler_gene_param_dict,
			gene_prior_dict = cycler_gene_prior_dict,
			min_amp = min_amp,
			max_amp = max_amp,
			prior_theta_euclid_dist = prior_theta_euclid_dist, # clock_posterior_dist
			folder_out = '%s/cell_phase_estimation' % alg_step_subfolder, # '%s/clock_and_confident_hv_inference' % folder_out,
			learning_rate_dict = vi_gene_param_lr_dict,
			gene_param_grad_dict = None,
			use_nb = use_nb,
			log_mean_log_disp_coef = log_mean_log_disp_coef,
			num_grid_points = num_phase_grid_points,
			num_cell_samples = num_cell_samples,
			num_gene_samples = num_gene_samples,
			vi_max_epochs = vi_max_epochs,
			vi_print_epoch_loss = vi_print_epoch_loss,
			vi_improvement_window = vi_improvement_window,
			vi_convergence_criterion = vi_convergence_criterion,
			vi_lr_scheduler_patience = vi_lr_scheduler_patience,
			vi_lr_scheduler_factor = vi_lr_scheduler_factor,
			vi_batch_size = vi_batch_size,
			vi_num_workers = vi_num_workers,
			vi_pin_memory = vi_pin_memory,
			batch_indicator_mat = batch_indicator_mat,
			detect_anomaly = detect_anomaly)



		print("AFTER")
		print("printing gene param df Q prob A")
		print(cycler_gene_prior_dict['prior_Q_prob_alpha'])
		print("printing gene param df Q prob B")
		print(cycler_gene_prior_dict['prior_Q_prob_beta'])




		# ** write out the params / priors as a DataFrame **

		# get df's
		cell_posterior_df = params_to_df.cell_multinomial_params_to_param_df(np.array(cycler_adata.obs.index), opt_cycler_theta_posterior_likelihood)
		cell_prior_df = params_to_df.cell_powerspherical_params_dict_to_param_df(np.array(cycler_adata.obs.index), cell_prior_dict)
		gene_param_df = params_to_df.gene_param_dicts_to_param_df(list(cycler_adata.var_names), utils.prep_gene_params(opt_cycler_gene_param_dict_unprepped), cycler_gene_prior_dict)




		# write
		cell_posterior_fileout = '%s/cell_phase_estimation/cell_posterior.tsv' % alg_step_subfolder # % (folder_out)
		cell_posterior_df.to_csv(cell_posterior_fileout,sep='\t')
		cell_prior_fileout = '%s/cell_phase_estimation/cell_prior.tsv' % alg_step_subfolder #  % (folder_out)
		cell_prior_df.to_csv(cell_prior_fileout,sep='\t')
		gene_param_df_fileout = '%s/cell_phase_estimation/gene_prior_and_posterior.tsv' % alg_step_subfolder #  (folder_out)
		gene_param_df.to_csv(gene_param_df_fileout,sep='\t')

		print("GENE PARAM DF")
		print("printing gene param df Q prob A")
		print(gene_param_df['prior_Q_prob_alpha'])
		print("printing gene param df Q prob B")
		print(gene_param_df['prior_Q_prob_beta'])




		# --- GET THE EVIDENCE FOR THE CELL PHASES ---

		print("--- ESTIMATING EVIDENCE FOR THE CELL PHASE POSTERIOR ---")


		# ** get the loc scale dict **
		gene_param_loc_scale_dict = utils.get_distribution_loc_and_scale(gene_param_dict = opt_cycler_gene_param_dict_unprepped, min_amp = min_amp, max_amp = max_amp, prep = True)


		# ** get theta sampled **
		theta_sampled = cell_posterior.ThetaPosteriorDist(opt_cycler_theta_posterior_likelihood).sample(num_samples=num_cell_samples)


		# ** compute expectation of the LL for each cell **
		cycler_cell_evidence = objective_functions.compute_expectation_log_likelihood(clock_X if use_clock_output_only else cycler_gene_X, log_L,
			theta_sampled = theta_sampled, mu_loc = gene_param_loc_scale_dict['mu_loc'][clock_indices], A_loc = gene_param_loc_scale_dict['A_loc'][clock_indices],
			phi_euclid_loc = gene_param_loc_scale_dict['phi_euclid_loc'][clock_indices], Q_prob_loc = gene_param_loc_scale_dict['Q_prob_loc'][clock_indices],
			use_is_cycler_indicators = gene_param_loc_scale_dict['Q_prob_loc'] is not None, exp_over_cells = True, use_flat_model = False,
			use_nb = use_nb, log_mean_log_disp_coef = log_mean_log_disp_coef, rsample = False)


		# ** compute expectation over cells **
		cycler_evidence = torch.mean(cycler_cell_evidence).item()


		# ** compute the percentile **
		percentile_in_null = scipy.stats.percentileofscore(null_ll_vec,cycler_evidence)


		# ** write **
		fileout = '%s/step_%s_cycler_percentile_in_null.txt' % (null_fits_folder_out, algorithm_step)
		with open(fileout,"wb") as file_obj:
			file_obj.write(str(percentile_in_null).encode())
		fileout = '%s/step_%s_cycler_evidence.txt' % (null_fits_folder_out, algorithm_step)
		with open(fileout,"wb") as file_obj:
			file_obj.write(str(cycler_evidence).encode())



		print("Cycler evidence percentile in null distribution: %s" % percentile_in_null)











		# --- GET HIGHLY CONFIDENT CELLS ---

		print("--- IDENTIFYING HIGHLY CONFIDENT CELLS BASED ON CELL PHASE POSTERIOR ---")

		if confident_cell_interval_size_threshold is None:
			confident_cell_indices = np.arange(0,adata.shape[0])
		else:
			# ** make the dist **
			cell_posterior_dist = cell_posterior.ThetaPosteriorDist(opt_cycler_theta_posterior_likelihood.detach())

			# ** compute the confidence intervals **
			cell_confidence_intervals = cell_posterior_dist.compute_confidence_interval(confidence=0.90)
			cell_confidence_intervals = (np.sum(cell_confidence_intervals,axis=1) / cell_posterior_dist.num_grid_points) * 24.0

			# ** get confident cell indices **
			confident_cell_indices = np.where(cell_confidence_intervals <= confident_cell_interval_size_threshold)[0]




		# --- FIT HVG PARAMETERS WHEN Q FIXED TO 1 ---

		print("--- FITTING HARMONIC PARAMETERS FOR HIGHLY VARIABLE GENES THAT ARE NON-CYCLERS ---")
		# print("WARNING: FOR ALG STEP >= 1, WE COULD INITIALIZE HV GENE PARAMETERS TO THOSE FROM PREVIOUS RUNS TO HELP W/ SPEED")


		# ** get hv adata **
		hv_adata = adata[:,(~(adata.var['is_cycler'])) & (adata.var['is_hv'])]

		# ** restrict to confident cells **
		hv_adata = hv_adata[confident_cell_indices,:]



		# ** initialize variational parameters to final converged params from previous runs for current non-cycling genes **
		if algorithm_step >= 1:

			# previous alg step subfolder
			previous_alg_step_subfolder = "%s/%s" % (alg_result_head_folder, algorithm_step - 1)

			# load the gene param df for the previous round's de novo cyclers
			previous_de_novo_cycler_gene_param_df_fileout = '%s/de_novo_cycler_id/gene_prior_and_posterior.tsv' % previous_alg_step_subfolder
			previous_de_novo_cycler_gene_param_df = pd.read_table(previous_de_novo_cycler_gene_param_df_fileout,sep='\t',index_col='gene')
			
			# get the current_non_cycler_gene_param_df
			current_non_cycler_gene_param_df = previous_de_novo_cycler_gene_param_df.loc[np.array(hv_adata.var_names)]

			# filter cols relevant for initializing variational parameters
			cols_to_keep = list(filter(lambda x: "prior" not in x, current_non_cycler_gene_param_df.columns)) # drop prior columns
			current_non_cycler_gene_param_df = current_non_cycler_gene_param_df[cols_to_keep]


			# add parameters to adata to initialize cycler genes 
			for col in cols_to_keep:
				hv_adata.var[col] = np.array(current_non_cycler_gene_param_df[col])



		# ** prep **
		hv_gene_X, log_L, hv_gene_param_dict, cell_prior_dict, hv_gene_prior_dict = prep.unsupervised_prep(hv_adata,**config_dict)
		hv_gene_prior_dict['prior_Q_prob_alpha'] = 999.0 * torch.ones(hv_adata.shape[1]) # just for this, set the HV Q prob prior to non-informative
		hv_gene_prior_dict['prior_Q_prob_beta'] = torch.ones(hv_adata.shape[1])
		hv_gene_param_dict['Q_prob_log_alpha'] = torch.nn.Parameter(torch.log(999.0 * torch.ones(hv_adata.shape[1])).detach(),requires_grad=True) # fit Q params to values that yield Q =~ 1
		hv_gene_param_dict['Q_prob_log_beta'] = torch.nn.Parameter(torch.log(torch.ones(hv_adata.shape[1])).detach(),requires_grad=True)
		gene_param_grad_dict = {
			"mu_loc" : True, "mu_log_scale" : True,
			"phi_euclid_loc" : True, "phi_log_scale" : True,
			"A_log_alpha" : True, "A_log_beta" : True,
			"Q_prob_log_alpha" : False, "Q_prob_log_beta" : False,
		}




		# ** run **
		_, opt_hv_gene_param_dict_unprepped = gene_fit.gene_fit(gene_X = hv_gene_X, 
			log_L = log_L, 
			gene_param_dict = hv_gene_param_dict, 
			gene_prior_dict = hv_gene_prior_dict,
			folder_out = "%s/de_novo_cycler_id_preinference" % (alg_step_subfolder),  # '%s/hv_preinference' % folder_out,
			learning_rate_dict = vi_gene_param_lr_dict,
			theta_posterior_likelihood = opt_cycler_theta_posterior_likelihood[confident_cell_indices,:], # opt_clock_theta_posterior_likelihood
			gene_param_grad_dict = gene_param_grad_dict,
			max_iters = vi_max_epochs, 
			num_cell_samples = num_cell_samples,
			num_gene_samples = num_gene_samples,
			max_amp = max_amp,
			min_amp = min_amp,
			print_epoch_loss = vi_print_epoch_loss,
			improvement_window = vi_improvement_window,
			convergence_criterion = vi_convergence_criterion,
			lr_scheduler_patience = vi_lr_scheduler_patience,
			lr_scheduler_factor = vi_lr_scheduler_factor,
			use_flat_model = False,
			batch_size = vi_batch_size,
			num_workers = vi_num_workers,
			pin_memory = vi_pin_memory,
			use_nb = use_nb,
			log_mean_log_disp_coef = log_mean_log_disp_coef,
			batch_indicator_mat = batch_indicator_mat,
			detect_anomaly = detect_anomaly)






		# --- FIT Q FOR HVG ---

		print("--- FITTING CYCLING INDICATOR PARAMETER FOR HIGHLY VARIABLE GENES THAT ARE NON-CYCLERS ---")


		# ** get hv adata **
		hv_adata = adata[:,(~(adata.var['is_cycler'])) & (adata.var['is_hv'])]

		# ** restrict to confident cells **
		hv_adata = hv_adata[confident_cell_indices,:]

		# ** prep **
		hv_gene_X, log_L, _, _, hv_gene_prior_dict = prep.unsupervised_prep(hv_adata,**config_dict)
		gene_param_grad_dict = {
			"mu_loc" : False, "mu_log_scale" : False,
			"phi_euclid_loc" : False, "phi_log_scale" : False,
			"A_log_alpha" : False, "A_log_beta" : False,
			"Q_prob_log_alpha" : True, "Q_prob_log_beta" : True,
		}



		# ** run **
		_, opt_hv_gene_param_dict_unprepped = gene_fit.gene_fit(gene_X = hv_gene_X, 
			log_L = log_L, 
			gene_param_dict = opt_hv_gene_param_dict_unprepped, 
			gene_prior_dict = hv_gene_prior_dict,
			folder_out = "%s/de_novo_cycler_id" % (alg_step_subfolder), # '%s/hv_preinference_Q_fit' % folder_out,
			learning_rate_dict = vi_gene_param_lr_dict,
			theta_posterior_likelihood = opt_cycler_theta_posterior_likelihood[confident_cell_indices,:], # opt_clock_theta_posterior_likelihood
			gene_param_grad_dict = gene_param_grad_dict,
			max_iters = vi_max_epochs, 
			num_cell_samples = num_cell_samples,
			num_gene_samples = num_gene_samples,
			max_amp = max_amp,
			min_amp = min_amp,
			print_epoch_loss = vi_print_epoch_loss,
			improvement_window = vi_improvement_window,
			convergence_criterion = vi_convergence_criterion,
			lr_scheduler_patience = vi_lr_scheduler_patience,
			lr_scheduler_factor = vi_lr_scheduler_factor,
			use_flat_model = False,
			batch_size = vi_batch_size,
			num_workers = vi_num_workers,
			pin_memory = vi_pin_memory,
			use_nb = use_nb,
			log_mean_log_disp_coef = log_mean_log_disp_coef,
			batch_indicator_mat = batch_indicator_mat,
			detect_anomaly = detect_anomaly)






		# --- IDENTIFY THE HVG THAT ARE HIGHLY CONFIDENT CYCLERS ---

		print("--- IDENTIFYING DE NOVO CYCLERS ---")


		# ** compute the pearson residuals of the amplitude loc's (based on mesor - amplitude relationship across all genes) **
		opt_hv_gene_param_loc_scale_dict = utils.get_distribution_loc_and_scale(gene_param_dict=opt_hv_gene_param_dict_unprepped,min_amp=min_amp,max_amp=max_amp,prep=True)
		mu_loc = opt_hv_gene_param_loc_scale_dict['mu_loc'].detach().numpy()
		A_loc = opt_hv_gene_param_loc_scale_dict['A_loc'].detach().numpy()
		kernel_model = statsmodels.nonparametric.kernel_regression.KernelReg(A_loc, mu_loc.reshape(-1,1), var_type = ['c'], bw = [0.1 / np.log10(np.e)]) 
		pred_A,marginal_effects = kernel_model.fit()
		est_std = (np.mean((A_loc - pred_A)**2)) ** 0.5
		A_loc_pearson_residuals = (A_loc - pred_A) / est_std



		# ** compute the fraction of Q = 1 for cyclers **
		num_Q_samples = 100
		hvg_gene_param_dist_dict = utils.init_distributions_from_param_dicts(gene_param_dict = opt_hv_gene_param_dict_unprepped, prep = True)
		Q_samples = utils.get_is_cycler_samples_from_dist(hvg_gene_param_dist_dict['Q_prob'],num_gene_samples=num_Q_samples,rsample=False)
		num_pos_cycler_samples = torch.sum(Q_samples,dim=0).detach().numpy()
		frac_pos_cycler_samples = num_pos_cycler_samples / num_Q_samples



		# ** get HV gene param df **
		hv_gene_param_df = params_to_df.gene_param_dicts_to_param_df(list(hv_adata.var_names), utils.prep_gene_params(opt_hv_gene_param_dict_unprepped), hv_gene_prior_dict)


		# ** add A_loc_pearson_residuals and frac_pos_cycler_samples to HV adata and gene parameter DF **
		hv_gene_param_df['A_loc_pearson_residual'] = A_loc_pearson_residuals
		hv_gene_param_df['frac_pos_cycler_samples'] = frac_pos_cycler_samples
		hv_adata.var['A_loc_pearson_residual'] = A_loc_pearson_residuals
		hv_adata.var['frac_pos_cycler_samples'] = frac_pos_cycler_samples



		# ** write out gene param DF **
		hv_gene_param_df_fileout = '%s/de_novo_cycler_id/gene_prior_and_posterior.tsv' % alg_step_subfolder # % (folder_out)
		hv_gene_param_df.to_csv(hv_gene_param_df_fileout,sep='\t')



		# ** get cycler genes based on frac_pos_cycler_samples and A_loc_pearson_residual_threshold **
		confident_hv_gene_param_df = hv_gene_param_df[(hv_gene_param_df['frac_pos_cycler_samples'] >= frac_pos_cycler_samples_threshold) & (hv_gene_param_df['A_loc_pearson_residual'] >= A_loc_pearson_residual_threshold)]
		new_de_novo_cycler_genes = np.array(list(confident_hv_gene_param_df.index))
		adata.var.loc[new_de_novo_cycler_genes,'is_cycler'] = True
			



		# --- IF SPECIFIED JUST TO USE THE CORE CLOCK ONLY, STOP HERE ---


		if use_clock_input_only:
			print("--- SUCCESSFULLY FINISHED ---")
			return






		# --- CHECK FOR CONVERGENCE ----

		print("Temporarily going until 3rd iteration!!!!")
		if algorithm_step >= 3:
			break

		# # break if evidence for core clock expression has converged
		# if prev_evidence is None:
		# 	prev_evidence = cycler_evidence
		# else:
		# 	evidence_improvement = ((cycler_evidence - prev_evidence) / np.abs(prev_evidence))
		# 	print("Evidence improvement at algorithm step %s: %s" % (algorithm_step, evidence_improvement))
		# 	if evidence_improvement <= 0.01:
		# 		break



		# otherwise, keep going
		algorithm_step +=1





	print("--- SUCCESSFULLY FINISHED ---")






if __name__ == "__main__":
	main(sys.argv)

















