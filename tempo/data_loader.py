import torch
from torch.utils.data import Dataset, DataLoader


class TempoDataset(Dataset):



	def __init__(self, X, log_L, theta_posterior_likelihood = None, theta_euclid_loc = None, theta_scale = None, theta = None, batch_indicator_mat = None):



		# --- COUNT DATA ---
		self.X = X # [num_cells x num_genes]
		self.log_L = log_L # [num_cells]



		# --- CELL PHASE PARAM ---
		self.theta_posterior_likelihood, self.theta_euclid_loc, self.theta_scale, self.theta = None, None, None, None
		if theta_posterior_likelihood is not None:
			self.theta_posterior_likelihood = theta_posterior_likelihood
		elif theta_euclid_loc is not None and theta_scale is not None:
			self.theta_euclid_loc = theta_euclid_loc
			self.theta_scale = theta_scale
		elif theta is not None:
			self.theta = theta



		# --- BATCH EFFECT PARAMS ---
		self.batch_indicator_mat = None
		if batch_indicator_mat is not None:
			self.batch_indicator_mat = batch_indicator_mat # [num_cells x num_batches]



	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		sample = {'X' : self.X[idx,:], 
		'log_L' : self.log_L[idx],
		'theta_euclid_loc' : None,
		'theta_scale' : None,
		'theta' : None,
		'theta_posterior_likelihood' : None,
		'batch_indicator_mat' : None}


		# ** add cell phase params **
		if self.theta_posterior_likelihood is not None:
			sample['theta_posterior_likelihood'] = self.theta_posterior_likelihood[idx,:]
		elif self.theta_euclid_loc is not None and self.theta_scale is not None:
			sample['theta_euclid_loc'] = self.theta_euclid_loc[idx,:]
			sample['theta_scale'] = self.theta_scale[idx]
		elif self.theta is not None:
			sample['theta'] = self.theta[idx]


		# ** add batch effect parameters **
		if self.batch_indicator_mat is not None:
			sample['batch_indicator_mat'] = self.batch_indicator_mat[idx,:]

		return sample





def tempo_collate(inp):


	# --- COMPUTE NUM CELLS AND GENES ---
	num_genes = inp[0]['X'].shape[0]
	num_cells = len(inp)



	# --- COLLATE THE CELL BASED TENSORS / PARAMETERS ---

	# ** X **
	X_list = [inp[z]['X'] for z in range(0, num_cells)]
	X = torch.stack(X_list,dim=0)


	# ** log_L **
	log_L_list = [inp[z]['log_L'] for z in range(0, num_cells)]
	log_L = torch.stack(log_L_list,dim=0)

	# ** theta_euclid_loc **
	theta_euclid_loc_list = [inp[z]['theta_euclid_loc'] for z in range(0, num_cells)]
	try:
		theta_euclid_loc = torch.stack(theta_euclid_loc_list,dim=0)
	except:
		theta_euclid_loc = None


	# ** theta_scale **
	theta_scale_list = [inp[z]['theta_scale'] for z in range(0, num_cells)]
	try:
		theta_scale = torch.stack(theta_scale_list,dim=0)
	except:
		theta_scale = None


	# ** theta_posterior_likelihood **
	theta_posterior_likelihood_list = [inp[z]['theta_posterior_likelihood'] for z in range(0, num_cells)]
	try:
		theta_posterior_likelihood = torch.stack(theta_posterior_likelihood_list,dim=0)
	except:
		theta_posterior_likelihood = None

	# ** theta **
	theta_list = [inp[z]['theta'] for z in range(0, num_cells)]
	try:
		theta = torch.stack(theta_list,dim=0)
	except:
		theta = None



	# ** batch_indicator_mat **
	batch_indicator_mat_list = [inp[z]['batch_indicator_mat'] for z in range(0, num_cells)]
	try:
		batch_indicator_mat = torch.stack(batch_indicator_mat_list,dim=0)
	except:
		batch_indicator_mat = None



	# --- MAKE THE OUTPUT DICT ---
	sample_dict = {'X' : X,
		'log_L' : log_L,
		'theta_euclid_loc' : theta_euclid_loc,
		'theta_scale' : theta_scale,
		'batch_indicator_mat' : batch_indicator_mat,
		'theta' : theta,
		'theta_posterior_likelihood' : theta_posterior_likelihood}





	return sample_dict














	