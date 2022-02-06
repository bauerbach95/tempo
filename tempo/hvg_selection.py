import numpy as np
import anndata
import scipy
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf



# --- DEFINE FN FOR GETTING MOST HIGHLY VARIABLE GENES ---

def get_hv_genes_binned(adata,num_bins=20,min_num_genes_per_bin=20,std_residual_threshold=1,viz=False):

	# ** compute MV relationship **
	try:
		log1p_prop_data = np.log10((np.array(adata.X.todense()) + 1) / np.array(adata.obs['lib_size']).reshape(-1,1))
	except:
		log1p_prop_data = np.log10((np.array(adata.X) + 1) / np.array(adata.obs['lib_size']).reshape(-1,1))
	log1p_prop_mean = np.mean(log1p_prop_data,axis=0)
	log1p_prop_var = np.var(log1p_prop_data,axis=0)

	# ** get bins for log1p_prop_mean **
	log1p_prop_mean_min, log1p_prop_mean_max = np.min(log1p_prop_mean), np.max(log1p_prop_mean)
	bin_boundaries = np.linspace(log1p_prop_mean_min,log1p_prop_mean_max,num_bins+1)

	# ** get the HV gene indices **
	hv_gene_indices = []
	for bin_index in range(0,num_bins):

		# get bin boundaries
		bin_start,bin_end = bin_boundaries[bin_index],bin_boundaries[bin_index+1]

		# get the genes within the bin
		bin_gene_indices = np.where((log1p_prop_mean >= bin_start) & (log1p_prop_mean < bin_end))[0]

		# if the number of genes in the bin is not greater than min_num_genes_per_bin, then continue
		if bin_gene_indices.size < min_num_genes_per_bin:
			continue


		# compute the median variance
		bin_gene_variances = log1p_prop_var[bin_gene_indices]
		bin_median_variance = np.median(bin_gene_variances)
		bin_std_variance = np.std(bin_gene_variances)

		# compute residuals until normal
		bin_gene_residuals = (bin_gene_variances - bin_median_variance) / bin_std_variance
		hv_gene_indices += list(bin_gene_indices[bin_gene_residuals >= std_residual_threshold])



	# ** viz, if specified **
	if viz:
		import matplotlib.pyplot as plt
		plt.clf()
		plt.scatter(log1p_prop_mean,log1p_prop_var,s=1,c='b')
		plt.scatter(log1p_prop_mean[hv_gene_indices],log1p_prop_var[hv_gene_indices],s=1,c='r')
		plt.title("Log1p prop var vs. mean colored by HV indicator")
		plt.show()

	# ** get hv gene names **
	hv_genes = np.array(adata.var_names[hv_gene_indices])

	return hv_genes





def get_hv_genes_kernel(adata,std_residual_threshold=0.5,viz=False,bw=0.1,pseudocount=1):

	# ** compute MV relationship **
	try:
		log1p_prop_data = np.log10((np.array(adata.X.todense()) + pseudocount) / np.array(adata.obs['lib_size']).reshape(-1,1))
	except:
		log1p_prop_data = np.log10((np.array(adata.X) + pseudocount) / np.array(adata.obs['lib_size']).reshape(-1,1))
	log1p_prop_mean = np.mean(log1p_prop_data,axis=0)
	log1p_prop_var = np.var(log1p_prop_data,axis=0)

	


	# ** fit kernel regression at specified bandwidth **
	model = statsmodels.nonparametric.kernel_regression.KernelReg(log1p_prop_var, log1p_prop_mean.reshape(-1,1), var_type = ['c'], bw = [bw]) 

	# ** compute the pearson residuals **
	pred_var,marginal_effects = model.fit()
	est_std = (np.mean((log1p_prop_var - pred_var)**2)) ** 0.5
	pearson_residuals = (log1p_prop_var - pred_var) / est_std
	
	# ** get the hv gene indices **
	hv_gene_indices = np.where(pearson_residuals >= std_residual_threshold)[0]
	
	
	# ** viz, if specified **
	if viz:
		import matplotlib.pyplot as plt
		plt.clf()
		plt.scatter(log1p_prop_mean,log1p_prop_var,s=1,c='b')
		plt.scatter(log1p_prop_mean[hv_gene_indices],log1p_prop_var[hv_gene_indices],s=1,c='r')
		plt.title("Log10_1p prop var vs. mean colored by HV indicator")
		plt.xlabel("Log10_1p prop mean")
		plt.ylabel("Log10_1p prop var")
		plt.savefig("/users/benauerbach/desktop/hv_fig.png",dpi=200)
		# plt.show()


	# ** get hv gene names **
	hv_genes = np.array(adata.var_names[hv_gene_indices])

	return hv_genes, pearson_residuals, log1p_prop_mean, log1p_prop_var, hv_gene_indices





