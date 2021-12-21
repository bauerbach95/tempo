import numpy as np
import scipy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tqdm
import anndata
import pandas as pd



def estimate_mean_disp_relationship(adata, init_coef, log10_prop_bin_marks, max_num_genes_per_bin,min_log_disp=-10,max_log_disp=10):


	# ** assign genes to bins **
	num_bins = len(log10_prop_bin_marks) - 1
	bin_gene_dict = {} # keys: bin index, vals: numpy array of gene names
	adata.var['log10_prop'] = np.log10(adata.var['prop'])
	for bin_index in range(0,len(log10_prop_bin_marks) - 1):
		bin_start_log10_prop, bin_end_log10_prop = log10_prop_bin_marks[bin_index], log10_prop_bin_marks[bin_index+1]
		bin_gene_dict[bin_index] = np.array(adata.var[(adata.var['log10_prop'] >= bin_start_log10_prop) & (adata.var['log10_prop'] < bin_end_log10_prop)].index)


	# ** if a bin contains more genes than max_num_genes_per_bin, sample max_num_genes_per_bin genes per bin **
	for bin_index in range(0,len(log10_prop_bin_marks) - 1):
		if bin_gene_dict[bin_index].size > max_num_genes_per_bin:
			bin_gene_dict[bin_index] = np.random.choice(bin_gene_dict[bin_index],size=max_num_genes_per_bin,replace=False)
		print("Bin %s size: %s" % (bin_index,bin_gene_dict[bin_index].size))

	# ** get the final gene list to estimate the mean - disp relationship **
	subsampled_genes = []
	for bin_index in range(0,len(log10_prop_bin_marks) - 1):
		subsampled_genes += list(bin_gene_dict[bin_index])


	# ** get the subsampled adata **
	subsampled_adata = adata[:,subsampled_genes]


	# ** fit mean models for each of the genes **

	def prep_regression_df(adata,gene_index):


		# ** get expression as a dense matrix **
		try:
			gene_X = np.array(adata.X[:,gene_index].todense())
		except:
			gene_X = np.array(adata.X[:,gene_index])
		try:
			gene_X = gene_X.flatten()
		except:
			pass


		# ** create gene_df to run regressions **


		# add expression and covariates to df
		gene_df = pd.DataFrame()
		gene_df["X"] = gene_X

		# log lib size
		try:
			gene_df["log_L"] = np.array(adata.obs['log_L'])
		except:
			pass


		# intercept
		gene_df['inter'] = 1

		return gene_df

	# fit Poisson adjusted for library size
	subsampled_gene_mu = np.zeros((subsampled_adata.shape[1]))
	for gene_index in tqdm.tqdm(range(0,subsampled_adata.shape[1])):
		gene_df = prep_regression_df(subsampled_adata,gene_index = gene_index)
		mean_formula = "X ~ inter - 1"
		mean_model = smf.glm(formula = mean_formula, data = gene_df, family=sm.families.Poisson(), offset = np.log(np.array(subsampled_adata.obs['lib_size'])))
		mean_res = mean_model.fit()
		subsampled_gene_mu[gene_index] = mean_res.params['inter']


	# ** estimate dispersion for each gene **


	# fixed mean and library size coefficient
	def compute_gene_nb_nll_fixed_mean_and_lib_coef(params, *args):

		# ** offset for numerical stability in log **
		eps = 1e-8

		# ** get arg data **

		# lambda (log proportion)
		gene_lambda = args[2]

		# counts for the gene
		x = args[0]

		# log L
		log_L = args[1]

		# lob L coefficient
		log_L_coef = args[3]

		# ** get gene parameters **

		# dispersion parameters
		log_disp = params[0]
		disp = np.exp(log_disp)
		theta = 1.0 / disp

		# compute the proportions and means
		pred_log_prop = gene_lambda
		pred_prop = np.exp(pred_log_prop)
		pred_mean = np.exp(pred_log_prop + (log_L_coef * log_L))


		# ** compute mean NLL **

		# LL of each observation
		log_theta_mu_eps = np.log(theta + pred_mean + eps)
		obs_ll = (
			theta * (np.log(theta + eps) - log_theta_mu_eps)
			+ x * (np.log(pred_mean + eps) - log_theta_mu_eps)
			+ scipy.special.loggamma(x + theta)
			-  scipy.special.loggamma(theta)
			-  scipy.special.loggamma(x + 1)
		)

		# mean NLL
		nll = -1 * np.mean(obs_ll)

		return nll

	def estimate_gene_nb_coefs(gene_lambda, gene_X, log_L, log_L_coef, log_disp_guess = -1, fixed_mean_lib_size_coef = True):

		# method 
		method = 'Powell' # 'BFGS'

		# run optimization
		if fixed_mean_lib_size_coef: # BFGS
			res = scipy.optimize.minimize(compute_gene_nb_nll_fixed_mean_and_lib_coef, [log_disp_guess], args = (gene_X, log_L, gene_lambda, log_L_coef), method = method)
		else:
			res = scipy.optimize.minimize(compute_gene_nb_nll_learnable_mean_and_lib_coef, [log_disp_guess, gene_lambda, log_L_coef], args = (gene_X, log_L), method = method)


		# get the result
		try:
			hessian_inverse = res.hess_inv.item()
		except:
			hessian_inverse = None
		mean_nll = res.fun
		nll = mean_nll * gene_X.size

		# ** get the parameter estimate dict **
		param_est_dict = {"log_disp" : res.x[0]}
		if not fixed_mean_lib_size_coef:
			param_est_dict['mu'] = res.x[1]
			param_est_dict['log_L'] = res.x[2]


		return param_est_dict, hessian_inverse, nll


	def estimate_nb_coefs(adata, gene_lambda_vec, log_L_coef, fixed_mean_lib_size_coef):


		param_est_dicts = []
		try:
			X = np.array(adata.X.todense())
		except:
			X = np.array(adata.X)
		log_L = np.array(adata.obs['log_L'])
		nll_list = []
		for gene_index in tqdm.tqdm(range(0,adata.shape[1])):

			# get gene lambda, X, and log_L
			gene_lambda = gene_lambda_vec[gene_index]
			gene_X = X[:,gene_index]

			# run
			param_est_dict, hessian_inverse, nll = estimate_gene_nb_coefs(gene_lambda, gene_X, log_L, log_L_coef[gene_index], fixed_mean_lib_size_coef = fixed_mean_lib_size_coef)


			# add
			param_est_dicts.append(param_est_dict)
			nll_list.append(nll)


		# make the param DF
		param_df = pd.DataFrame()
		for key in param_est_dicts[0]:
			val_list = list(map(lambda x: x[key], param_est_dicts))
			param_df[key] = val_list
		param_df['gene'] = list(adata.var_names)
		param_df['nll'] = nll_list
		param_df = param_df.set_index("gene")


		return param_df


	# estimate NB parameters
	subsampled_gene_nb_param_est_df = estimate_nb_coefs(subsampled_adata, gene_lambda_vec = subsampled_gene_mu, log_L_coef = np.ones((adata.shape[1])), fixed_mean_lib_size_coef = True)
	subsampled_gene_nb_param_est_df['log_L'] = np.ones((subsampled_adata.shape[1]))
	subsampled_gene_nb_param_est_df['mu'] = subsampled_gene_mu


	# ** estimate the mean - disp relationship **
	def poly_mse(params, *args):

		# ** get arg data **

		# gene mu
		mu = args[0]

		# log dispersion
		log_disp = args[1]

		# polynomial power
		poly_power = len(params) - 1


		# ** compute the estimates of the log disp **
		log_disp_hat = np.matmul(np.power(mu.reshape(-1,1),np.arange(0,poly_power+1)),params) # [num_genes]

		# ** compute the MSE **
		mse = np.mean((log_disp - log_disp_hat) ** 2)


		return mse



	def estimate_poly(log_disp,mu,init_coef):

		# run optimization
		res = scipy.optimize.minimize(poly_mse, init_coef, args = (mu,log_disp), method = 'Powell')


		return res
	
	
	

	# get rid of the outlier dispersion estimates
	subsampled_gene_nb_param_est_df_no_outliers = subsampled_gene_nb_param_est_df[(subsampled_gene_nb_param_est_df['log_disp'] >= min_log_disp) & (subsampled_gene_nb_param_est_df['log_disp'] <= max_log_disp)]


	# est
	mean_disp_fit_res = estimate_poly(log_disp = np.array(subsampled_gene_nb_param_est_df_no_outliers['log_disp']), mu = np.array(subsampled_gene_nb_param_est_df_no_outliers['mu']), init_coef = init_coef)


	# return the coefficients
	mean_disp_fit_coef = mean_disp_fit_res.x
	
	
	return mean_disp_fit_coef











