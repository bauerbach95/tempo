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
from . import clock_gene_posterior
from . import est_cell_phase_from_current_cyclers
from . import identify_de_novo_cyclers
from . import compute_clock_evidence
import random
import gc




# - Description: Runs Tempo algorithm for unsupervised phase inference
# - Parameters:
#		- adata ([num_cells x num_genes] AnnData object): gene count matrix 
#		- folder_out (str): path for results folder
#		- gene_acrophase_prior_path (str): path to gene acrophase prior file, which is a CSV
#		- core_clock_gene_path (str): path to file for core clock genes, which is a plain text file listing the names of each core clock gene on each line
#		- cell_phase_prior_path (str): path to cell phase prior file, which is a CSV
#		- reference_gene (str): name of reference core clock gene
#		- min_gene_prop (float; 0 to 1): minimum proportion of transcripts in pseudobulk that the gene must have
#		- min_amp (float, positive real): minimum amplitude of genes' amplitudes
#		- max_amp (float, positive real): maximimum amplitude of genes' amplitudes
# 		- init_mesor_scale_val (float, positive real): value to initialize variational mesor scale for all genes
# 		- prior_mesor_scale_val (float, positive real): value to set prior mesor scale for all genes
# 		- init_amp_loc_val (float, positive real lying in [min_amp,max_amp]): value to initialize the location of the variational amplitude for all genes
# 		- init_amp_scale_val (float, positive real): number of pseudotrials of the Beta distributoin to initialize the variational amplitude to. Larger values indicate more certainty.
# 		- prior_amp_alpha_val (float, positive real): alpha values of Beta distribution for prior amplitude
# 		- prior_amp_beta_val (float, positive real): beta values of Beta distribution for prior amplitude
# 		- known_cycler_init_shift_95_interval (float in [0,pi]): 95% interval to initialize the acrophase distribution scale, specifically for genes that are known cycling genes
# 		- unknown_cycler_init_shift_95_interval (float in [0,pi]): 95% interval to initialize the acrophase distribution scale, specifically for genes that are not known cycling genes
# 		- known_cycler_prior_shift_95_interval: 95% interval to set acrophase prior distribution scale, specifically for genes that are known cycling genes; note: values in gene_acrophase_prior_path take precedence
# 		- init_clock_Q_prob_alpha (float, positive real): alpha value of Beta distribution for variational gamma (probability gene has non-zero amplitude), specifically for user-supplied clock genes
# 		- init_clock_Q_prob_beta (float, positive real): beta value of Beta distribution for variational gamma (probability gene has non-zero amplitude), specifically for user-supplied clock genes
# 		- init_non_clock_Q_prob_alpha (float, positive real): alpha value of Beta distribution for variational gamma (probability gene has non-zero amplitude), specifically for non-clock genes
# 		- init_non_clock_Q_prob_beta (float, positive real): beta value of Beta distribution for variational gamma (probability gene has non-zero amplitude), specifically for non-clock genes
# 		- prior_clock_Q_prob_alpha (float, positive real): alpha value of Beta distribution for prior gamma (probability gene has non-zero amplitude), specifically for user-supplied clock genes
# 		- prior_clock_Q_prob_beta (float, positive real): beta value of Beta distribution for prior gamma (probability gene has non-zero amplitude), specifically for user-supplied clock genes
# 		- prior_non_clock_Q_prob_alpha (float, positive real): alpha value of Beta distribution for prior gamma (probability gene has non-zero amplitude), specifically for non-clock genes
# 		- prior_non_clock_Q_prob_beta (float, positive real): beta value of Beta distribution for prior gamma (probability gene has non-zero amplitude), specifically for non-clock genes
# 		- use_noninformative_phase_prior (boolean): if true, all cell phase priors are set to noninformative priors. however, if true, note cell phase priors specified by cell_phase_prior_path take precedent. if false, user must supply cell_phase_prior_path.
# 		- use_nb (boolean): if true, uses negative binomial likelihood model for gene transcript counts. if false, uses poisson.
# 		- mean_disp_init_coef (list of floats): initial values to set the log transcript proportion - log dispersion coefficients to (zeta parameterizing function g in the paper supplement) when use_nb is true, since the coefficients are fit using a gradient optimizer
# 		- est_mean_disp_relationship (boolean): if true, optimizes the log transcript proportion - log dispersion coefficients; if false, directly treats the user-supplied coefficients in mean_disp_init_coef as zeta
# 		- mean_disp_log10_prop_bin_marks (list of log10 transformed fractions / proportions): where to set bins of genes' log10 proportions to sample genes to estimate zeta (parameters of the globa log proportion - log dispersion relationship )
# 		- mean_disp_max_num_genes_per_bin (int, positive): maximum number of genes to sample per bin when estimating zeta (parameters of the globa log proportion - log dispersion relationship )
# 		- hv_std_residual_threshold (float): threshold of pearson residuals of genes' variances vs. expected variances (given their means) to restrict highly variable genes to consider as postential cycling genes by the algorithm. 
# 		- mu_loc_lr (positive float):  learning rate for variational mesor loc parameter
# 		- mu_log_scale_lr (positive float):  learning rate for variational mesor scale parameter
# 		- A_log_alpha_lr (positive float):  learning rate for variational amplitude alpha parameter
# 		- A_log_beta_lr (positive float):  learning rate for variational amplitude beta parameter
# 		- phi_euclid_loc_lr (positive float):  learning rate for variational acrophase loc parameter
# 		- phi_log_scale_lr (positive float):  learning rate for variational acrophase scale parameter
# 		- Q_prob_log_alpha_lr (positive float):  learning rate for variational non-zero amplitude probability loc parameter
# 		- Q_prob_log_beta_lr (positive float):  learning rate for variational non-zero amplitude probability scale parameter
# 		- num_phase_grid_points (positive int): number of grid points to use to approximate the conditional posterior cell phase distribution
# 		- num_phase_est_cell_samples (positive int): number of monte carlo samples of the cell phases to use to compute the ELBO expectation term when estimates cell phase (step 1 of the algorithm) 
# 		- num_phase_est_gene_samples (positive int): number of monte carlo samples of the gene parameters to use to compute the ELBO expectation term when estimates cell phase (step 1 of the algorithm) 
# 		- num_harmonic_est_cell_samples (positive int): number of monte carlo samples of the cell phases to use when fitting gene parameters of non-cycling genes in step 2 of the algorithm
# 		- num_harmonic_est_gene_samples (positive int): number of monte carlo samples of the gene parameters to use to compute the ELBO expectation term when fitting gene parameters of non-cycling genes in step 2 of the algorithm 
# 		- vi_max_epochs (positive int): maximum number of epochs used to optimize parameters (for either step 1 or 2 of the algorithm)
# 		- vi_print_epoch_loss (boolean): if true, prints the ELBO at each epoch for each step of the algorithm
# 		- vi_improvement_window (int): size of the window of epochs to compare ELBO progress to (i.e. for vi_improvement_window = 10, the mean ELBO in the last 10 epochs is compared to the previous non-overlapping 10 epoch window)
# 		- vi_convergence_criterion (positive float): threshold improvement of current epoch window's mean ELBO to previous epoch window's mean ELBO at which to say the algorithm has converged
# 		- vi_lr_scheduler_patience (positive int): number of epochs of the ELBO getting worse before the scheduler decreases the learning rate
# 		- vi_lr_scheduler_factor (positive float): the multiplicative factor to apply to the current learning rate if the scheduler has "run out of patience"; values < 1 will lead to a decreased learning rate
# 		- vi_batch_size (positive int): cell batch size to complete the objective function
# 		- test_mode (boolean): if true, uses pytorch profilers which can slow down computation
# 		- use_clock_input_only (boolean): if true, algorithm only uses the core clock genes to estimate cell phase (i.e. it only runs Step 1 of the algorithm using the core clock genes, and then the algorithm halts.)
# 		- use_clock_output_only (boolean): if true, the algorithm only uses the core clock genes to compute the expectation of the ELBO in Step 1
# 		- frac_pos_cycler_samples_threshold (float, [0,1]): threshold for the MAP of a gene's non-zero amplitude probability to call them a de novo cycler
# 		- A_loc_pearson_residual_threshold (float): threshold for the difference between a gene's MAP amplitude and expected amplitude (given its mesor) reported in terms of a pearson residual in order to call the gene a de novo cyceler; larger values mean a stricter threshold
# 		- confident_cell_interval_size_threshold (float in [0 to 24]): threshold for a cells' 95% posterior interval for the cell to be considered when computing the expectation of the ELBO in Step 2 (to identify de novo cyclers); this can be used to improve computational efficiency, since cells with high uncertainty will contribute little information to the estimation of gene parameters
# 		- max_num_alg_steps (int): maximum number of times to run Steps 1 and 2 of the algorithm
# 		- opt_phase_est_gene_params (boolean): if False, does not optimize the variational gene distributions in step 1 -- the variational gene distributions are set to the priors, and Step 1 uses these and the observed counts to compute the conditional posterior of the cell phases
# 		- init_variational_dist_to_prior (boolean): if True, always sets the gene variational distributions to prior distributions at initialization of Steps 1 and 2. 
# 		- log10_bf_tempo_vs_null_threshold (positive float): threshold of the log10 bayes factor comparing Tempo's core clock evidence (from step 1) to random core clock evidence; if the bayes factor does not exceed this threshold, the algorithm halts



def run(adata,
	folder_out,
	gene_acrophase_prior_path,
	core_clock_gene_path,
	cell_phase_prior_path = None,
	reference_gene = 'Arntl',
	min_gene_prop = 1e-5,
	min_amp = 0.0, # ** prep parameters **
	max_amp = 1.5 / np.log10(np.e),
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
	prior_non_clock_Q_prob_beta = 1,
	use_noninformative_phase_prior = True,
	use_nb = True,
	mean_disp_init_coef = [-4,-0.2], # ** mean / disp relationship learning parameters **
	est_mean_disp_relationship = True,
	mean_disp_log10_prop_bin_marks = list(np.linspace(-5,-1,20)), 
	mean_disp_max_num_genes_per_bin = 50,
	hv_std_residual_threshold = 0.5, # ** HVG selection parameters **
	clock_std_residual_threshold = -5, # 1.0,
	mu_loc_lr = 1e-1, # ** conditional posterior opt parameters **
	mu_log_scale_lr = 1e-1,
	A_log_alpha_lr = 1e-1,
	A_log_beta_lr = 1e-1,
	phi_euclid_loc_lr = 1e-1,
	phi_log_scale_lr = 1e-1,
	Q_prob_log_alpha_lr = 1e-1,
	Q_prob_log_beta_lr = 1e-1,
	num_phase_grid_points = 24,
	num_phase_est_cell_samples = 3,
	num_phase_est_gene_samples = 3,
	num_harmonic_est_cell_samples = 3,
	num_harmonic_est_gene_samples = 3,
	vi_max_epochs = 300,
	vi_print_epoch_loss = True,
	vi_improvement_window = 10,
	vi_convergence_criterion = 1e-3,
	vi_lr_scheduler_patience = 10,
	vi_lr_scheduler_factor = 0.1,
	vi_batch_size = 3000,
	vi_num_workers = 0,
	vi_pin_memory = False,
	test_mode = False,
	use_clock_input_only = False,
	use_clock_output_only = True,
	frac_pos_cycler_samples_threshold = 0.95,
	A_loc_pearson_residual_threshold = 1.0,
	confident_cell_interval_size_threshold = 12.0,
	max_num_alg_steps=3,
	opt_phase_est_gene_params = True,
	init_variational_dist_to_prior = False,
	log10_bf_tempo_vs_null_threshold = np.log10(1.5),
	use_de_novo_cycler_detection = True,
	**kwargs):


	# --- SET NUM NULL SHUFFLES TO 1 ---
	num_null_shuffles = 1


	# --- MAKE FOLDER OUTS ---

	# ** main folder **
	if not os.path.exists(folder_out):
		os.makedirs(folder_out)
		

	# ** mean_disp_param_folder_out folder **
	mean_disp_param_folder_out = "%s/mean_disp_param" % folder_out
	if not os.path.exists(mean_disp_param_folder_out):
		os.makedirs(mean_disp_param_folder_out)


	# # ** evidence folder **
	evidence_folder_out = '%s/evidence' % folder_out
	if not os.path.exists(evidence_folder_out):
		os.makedirs(evidence_folder_out)



	# --- GET THE CONFIG DICT AND WRITE ---

	# ** get **
	config_dict = locals()
	del config_dict['adata']
	del config_dict['folder_out']

	# ** write **
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



	# --- MAKE SURE MAX NUM ALG STEPS IS AT LEAST 1 ---
	max_num_alg_steps = max(max_num_alg_steps,1)
	

	# --- COMPUTE THE LIBRARY SIZES AND PROP'S ---
	adata.obs['lib_size'] = np.array(np.sum(adata.X,axis=1))
	adata.obs['log_L'] = np.log(np.array(adata.obs['lib_size']))
	adata.var['prop'] = np.array(np.sum(adata.X,axis=0)).flatten() / np.sum(adata.X)
	

	# --- GET CORE CLOCK GENES ---
	core_clock_genes = list(pd.read_table(core_clock_gene_path,header=None).iloc[:,0])
	core_clock_genes = list(filter(lambda x: x in adata.var_names, core_clock_genes))
	core_clock_genes = list(filter(lambda x: adata.var.loc[x]['prop'] > 0, core_clock_genes))
	core_clock_genes = np.array(list(sorted(core_clock_genes)))


	# --- INITIALIZE is_clock and is_cycler FOR ALL GENES TO FALSE OTHER THAN THE CORE CLOCK GENES ---
	adata.var['is_clock'] = False
	adata.var.loc[core_clock_genes,'is_clock'] = True
	adata.var['is_cycler'] = False
	adata.var.loc[core_clock_genes,'is_cycler'] = True


	# --- MAKE AN INITIAL GUESS ABOUT PARAMS FOR MEAN / DISPERSION RELATIONSHIP ---

	

	# ** get the log mean - log disp polynomial coefficients **
	if use_nb:
		print("--- ESTIMATING GLOBAL MEAN-DISPERSION RELATIONSHIP FOR NEGATIVE BINOMIAL ---")
		if est_mean_disp_relationship:
			log_mean_log_disp_coef = estimate_mean_disp_relationship.estimate_mean_disp_relationship(adata, mean_disp_init_coef, mean_disp_log10_prop_bin_marks, mean_disp_max_num_genes_per_bin,min_log_disp=-10,max_log_disp=10)
		else:
			log_mean_log_disp_coef = np.array(mean_disp_init_coef)
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

	hv_genes, pearson_residuals, log1p_prop_mean, log1p_prop_var, hv_gene_indices = hvg_selection.get_hv_genes_kernel(adata,std_residual_threshold=hv_std_residual_threshold,viz=False,bw=0.05,pseudocount=1) # bw = 0.1
	hv_genes = np.setdiff1d(np.array(hv_genes),np.array(core_clock_genes)) # make sure the core clock genes are not in there
	hv_genes = np.array(sorted(list(hv_genes)))
	adata.var['is_hv'] = False
	adata.var.loc[hv_genes,'is_hv'] = True
	adata.var['var_pearson_residual'] = pearson_residuals



	# --- RESTRICT ADATA TO THOSE THAT ARE CORE CLOCK OR CANDIDATE HV GENES ---
	adata = adata[:,(np.isin(adata.var_names, hv_genes)) | (np.isin(adata.var_names, core_clock_genes))]


	# --- GET RID OF CLOCK GENES THAT DON'T HAVE SUFFICIENT VARIANCE ---
	clock_adata = adata[:,adata.var['is_clock']]
	core_clock_genes = list(clock_adata[:,clock_adata.var['var_pearson_residual'] >= clock_std_residual_threshold].var_names)


	# --- ORDER ADATA S.T. CORE CLOCK GENES FIRST AND THEN HV GENES SECOND ---
	adata = adata[:,list(core_clock_genes) + list(hv_genes)]


	print("Adata shape after thresholding based on gene variance")
	print(str(adata.shape))

	# --- GENERATE THE NULL DISTRIBUTION OF CORE CLOCK EVIDENCE ---

	print("--- GENERATING NULL DISTRIBUTION OF CORE CLOCK EVIDENCE UNDER RANDOM CELL PHASE ASSIGNMENTS ---")

	# ** get the clock adata **
	clock_adata = adata[:,adata.var['is_clock']]


	print("clock_adata shape after thresholding based on gene variance")
	print(str(clock_adata.shape))

	# ** run **
	null_log_evidence_vec = generate_null_dist.run(clock_adata.copy(), evidence_folder_out, log_mean_log_disp_coef, **copy.deepcopy(config_dict))


	# ** release memory **
	del clock_adata
	gc.collect()
	






	# --- START ALGORTHM ---

	algorithm_step = 0
	prev_clock_log_evidence = None
	alg_result_head_folder = "%s/tempo_results" % folder_out
	alg_step_to_return = None
	better_than_random = False
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

		# ** initialize variational / prior parameters to those from previous runs for clock and de novo cyclers **
		if algorithm_step >= 1:
			previous_alg_step_subfolder = "%s/%s" % (alg_result_head_folder, algorithm_step - 1)
			cycler_adata = utils.init_cycler_adata_variational_and_prior_dist_from_prev_round(cycler_adata, previous_alg_step_subfolder, enforce_de_novo_cycler_flat_Q_prior = False)


		# ** estimate cell phase from clock + de novo cyclers **
		phase_est_folder_out = alg_step_subfolder
		opt_cycler_theta_posterior_likelihood, opt_cycler_gene_param_dict_unprepped = est_cell_phase_from_current_cyclers.run(cycler_adata, phase_est_folder_out, log_mean_log_disp_coef, **config_dict)





		# --- COMPUTE CORE CLOCK EVIDENCE ---

		print("--- ESTIMATING CLOCK EVIDENCE FOR THE CELL PHASE POSTERIOR ---")

		# ** compute **
		clock_log_evidence = compute_clock_evidence.run(cycler_adata, opt_cycler_gene_param_dict_unprepped,
			opt_cycler_theta_posterior_likelihood, log_mean_log_disp_coef, **config_dict)

		# ** write out evidence **
		fileout = '%s/step_%s_clock_evidence.txt' % (evidence_folder_out, algorithm_step)
		with open(fileout,"wb") as file_obj:
			file_obj.write(str(clock_log_evidence).encode())


		# ** compare the clock evidence to random **
		log10_bf_tempo_vs_null = (clock_log_evidence - np.max(null_log_evidence_vec)) / np.log(10)


		# ** halt algorithm progression if not sufficiently better than random **
		print("Tempo vs. null clock evidence log10 bayes factor: %s" % log10_bf_tempo_vs_null)
		if log10_bf_tempo_vs_null < log10_bf_tempo_vs_null_threshold:
			print("Clock evidence not sufficiently better than random. Halting algorithm.")	
			break


		# ** flip indicator if we can do better than random **
		if algorithm_step == 0 and log10_bf_tempo_vs_null >= log10_bf_tempo_vs_null_threshold:
			better_than_random = True

		
		# ** halt algorithm progression if clock evidence worsens **
		if algorithm_step == 0:
			prev_clock_log_evidence = clock_log_evidence
		elif clock_log_evidence - prev_clock_log_evidence < 0:
			print("Clock log evidence decreased from previous Tempo step. Halting algorithm.")
			alg_step_to_return = algorithm_step - 1
			break



		# ** release memory **
		del cycler_adata
		gc.collect()


		# ** halt algorithm if not running w/ de novo cycler detection **
		if not use_de_novo_cycler_detection:
			alg_step_to_return = 0
			break


		# --- IDENTIFY DE NOVO CYCLERS ---

		# ** get hv adata **
		hv_adata = adata[:,(~(adata.var['is_cycler'])) & (adata.var['is_hv'])]

		# ** initialize variational / prior parameters to those from previous runs for clock and de novo cyclers **
		if algorithm_step >= 1:
			previous_alg_step_subfolder = "%s/%s" % (alg_result_head_folder, algorithm_step - 1)
			hv_adata = utils.init_hv_adata_variational_and_prior_dist_from_prev_round(hv_adata, previous_alg_step_subfolder)


		# ** id de novo cyclers **
		de_novo_cycler_detection_folder_out = alg_step_subfolder
		new_de_novo_cycler_genes = identify_de_novo_cyclers.run(hv_adata, opt_cycler_theta_posterior_likelihood, de_novo_cycler_detection_folder_out, log_mean_log_disp_coef, **config_dict)
		adata.var.loc[new_de_novo_cycler_genes,'is_cycler'] = True


		# ** release memory **
		del hv_adata
		gc.collect()



		# --- IF SPECIFIED JUST TO USE THE CORE CLOCK ONLY, STOP HERE ---
		if use_clock_input_only:
			print("--- SUCCESSFULLY FINISHED ---")
			return





		# --- INCREMENT ALGORITHM STEP / HALT ----


		# otherwise, keep going
		algorithm_step +=1

		# if reached max number of steps, halt
		if algorithm_step >= max_num_alg_steps:
			alg_step_to_return = algorithm_step - 1
			break





	print("--- SUCCESSFULLY FINISHED ---")


	# --- WRITE OPTIMAL RESULTS FROM alg_step_to_return ---

	if better_than_random:

		# paths out
		opt_folder_out = '%s/opt' % alg_result_head_folder
		if not os.path.exists(opt_folder_out):
			os.makedirs(opt_folder_out)
		opt_cell_posterior_df_path_out = '%s/cell_posterior.tsv' % opt_folder_out
		opt_cycler_gene_df_path_out = '%s/cycler_gene_prior_and_posterior.tsv' % opt_folder_out
		opt_flat_gene_df_path_out = '%s/flat_gene_prior_and_posterior.tsv' % opt_folder_out

		# read in cell phase data and write out
		opt_cell_posterior_df_path = '%s/%s/cell_phase_estimation/cell_posterior.tsv' % (alg_result_head_folder, alg_step_to_return)
		opt_cell_posterior_df = pd.read_table(opt_cell_posterior_df_path,sep='\t',index_col='barcode')
		opt_cell_posterior_df.to_csv(opt_cell_posterior_df_path_out,sep='\t')

		# read in cycler gene data and write
		opt_cycler_gene_df_path = '%s/%s/cell_phase_estimation/gene_prior_and_posterior.tsv' % (alg_result_head_folder, alg_step_to_return)
		opt_cycler_gene_df = pd.read_table(opt_cycler_gene_df_path,sep='\t',index_col='gene')
		opt_cycler_gene_df.to_csv(opt_cycler_gene_df_path_out,sep='\t')


		if use_de_novo_cycler_detection:

			# read in flat gene data and write
			opt_flat_gene_df_path = '%s/%s/de_novo_cycler_id/gene_prior_and_posterior.tsv' % (alg_result_head_folder, alg_step_to_return)
			try:
				opt_flat_gene_df = pd.read_table(opt_flat_gene_df_path,sep='\t',index_col='gene')
				opt_flat_gene_df.to_csv(opt_flat_gene_df_path_out,sep='\t')
			except Exception as e:
				print("Error: writing out optimal flat gene df: %s" % str(e))







if __name__ == "__main__":
	main(sys.argv)

















