import torch
import numpy as np
import copy




# ** make the cell phase dist **
class ThetaPosteriorDist():
	def __init__(self,theta_posterior_likelihood):
		self.theta_posterior_likelihood = theta_posterior_likelihood
		self.num_grid_points = theta_posterior_likelihood.shape[1]
		max_comp = torch.max(self.theta_posterior_likelihood,dim=1)
		self.map_certainty = max_comp.values
		self.map_uncertainty = 1.0 - self.map_certainty
		self.map_indices = max_comp.indices
		self.map_phase = (self.map_indices / self.num_grid_points) * 2 * np.pi
		self.num_cells = theta_posterior_likelihood.shape[0]
	def sample(self,num_samples):
		return (torch.multinomial(self.theta_posterior_likelihood, num_samples=num_samples, replacement=True) / self.num_grid_points) * 2 * np.pi


	def rsample(self,num_samples):
		cell_phase_samples = torch.zeros((self.num_cells,num_samples))
		phase_grid_reshaped = torch.linspace(0,2*np.pi,self.num_grid_points).double().reshape(-1,1) # [num_grid_points x 1]
		logits = torch.log(torch.clamp(self.theta_posterior_likelihood,min=1e-300)) # clamping for numerical stability
		for sample_index in range(0,num_samples):
			is_cycler_sample = torch.nn.functional.gumbel_softmax(logits = logits,
											  tau = 1,
											  hard = True,
											  dim = -1)
			sampled_phase = torch.matmul(is_cycler_sample, phase_grid_reshaped)
			sampled_phase = sampled_phase.flatten()
			cell_phase_samples[:,sample_index] = sampled_phase
		return cell_phase_samples



	def get_map_shifted_posterior_likelihood(self):
		# ** get the shifted posterior likelihood mat (s.t. MAP is at 0) **
		shifted_indices = (torch.arange(0,self.num_grid_points).reshape(-1,1) - self.map_indices) % (self.num_grid_points) # [num_grid_points x num_cells]
		shifted_indices = shifted_indices.T # [num_cells x num_grid_points]
		shifted_posterior_likelihood = copy.deepcopy(self.theta_posterior_likelihood)
		for cell_index in range(0,self.num_cells):
			shifted_posterior_likelihood[cell_index,shifted_indices[cell_index,:]] = self.theta_posterior_likelihood[cell_index,:]
		return shifted_posterior_likelihood, shifted_indices

	def get_shifted_posterior_likelihood(self,radian_shift):
		radian_shift = (4 / 24) * 2 * np.pi
		radian_shift_discretized = np.round((radian_shift / (2 * np.pi)) * 24)
		print("printing radian shift discretized: %s" % str(radian_shift_discretized))
		shifted_posterior_likelihood = copy.deepcopy(self.theta_posterior_likelihood)
		shifted_posterior_likelihood = torch.roll(shifted_posterior_likelihood, shifts=-1 * int(radian_shift_discretized), dims=1)
		return shifted_posterior_likelihood


	# for a given cell, [num_grid_points] boolean matrix, where true if in the interval
	def compute_indiv_cell_confidence_interval(self,map_shifted_cell_posterior_likelihood,confidence):


		# ** compute the interval at given confidence level **
		cumsum = map_shifted_cell_posterior_likelihood[0]
		in_interval = torch.zeros(map_shifted_cell_posterior_likelihood.shape)
		in_interval[0] = 1
		increasing_index = 1
		decreasing_index = self.num_grid_points - 1
		while cumsum < confidence:
			if map_shifted_cell_posterior_likelihood[increasing_index] >= map_shifted_cell_posterior_likelihood[decreasing_index]:
				cumsum += map_shifted_cell_posterior_likelihood[increasing_index]
				in_interval[increasing_index] = 1
				increasing_index += 1
			else:
				cumsum += map_shifted_cell_posterior_likelihood[decreasing_index]
				in_interval[decreasing_index] = 1
				decreasing_index -= 1
			if decreasing_index < increasing_index:
				break
			
		return in_interval


	# return [num_cells x num_grid_points] boolean matrix, where true if in the interval
	def compute_confidence_interval(self,confidence,map_shifted_posterior_likelihood = None, map_shifted_indices = None):


		# ** get the map shifted posterior likelihood **
		if map_shifted_posterior_likelihood is None and map_shifted_indices is None:
			map_shifted_posterior_likelihood, map_shifted_indices = self.get_map_shifted_posterior_likelihood()


		# ** compute the confidence interval for each cell **
		cell_confidence_intervals = np.zeros(map_shifted_posterior_likelihood.shape)
		for cell_index in range(0,self.num_cells):    
			cell_confidence_intervals[cell_index,:] = self.compute_indiv_cell_confidence_interval(copy.deepcopy(map_shifted_posterior_likelihood[cell_index,:]),confidence=confidence).numpy()

		return cell_confidence_intervals




	# confidence: [0,1]
	# true_cell_phase: [num_cells] in range [0,2 Pi]
	def get_num_cells_in_interval_at_confidence(self, confidence, true_cell_phase, map_shifted_posterior_likelihood = None, map_shifted_indices = None):


		# ** get the map shifted posterior likelihood **
		if map_shifted_posterior_likelihood is None and map_shifted_indices is None:
			map_shifted_posterior_likelihood, map_shifted_indices = self.get_map_shifted_posterior_likelihood()


		# ** get the confidence interval for each cell: [num_cells x num_grid_points] **
		confidence_interval = self.compute_confidence_interval(confidence,map_shifted_posterior_likelihood,map_shifted_indices)


		# ** discretize the true cell phase **
		true_cell_phase_index = np.round((true_cell_phase / (2 * np.pi)) * self.num_grid_points)

		# ** shift the discretized true cell phase **
		shifted_true_cell_phase_index = (torch.Tensor(true_cell_phase_index) - self.map_indices) % (self.num_grid_points)
		shifted_true_cell_phase_index = shifted_true_cell_phase_index.int()


		# ** compute the number of cells that fell into interval **
		num_cells_in_interval = 0
		for cell_index in range(0,self.num_cells):    
			if confidence_interval[cell_index,shifted_true_cell_phase_index[cell_index]] == 1:
				num_cells_in_interval += 1
				
		return num_cells_in_interval   




	# phases: [num_samples x num_cells]
	def log_prob(self,phases):

		# ** assign the phases to bins **
		phases_discretized_indices = torch.round((phases / (2 * np.pi)) * self.num_grid_points) # convert [0,1] to [0,num_grid_points] (float), and then discretize
		phases_discretized_indices[torch.where(phases_discretized_indices == self.num_grid_points)] = 0 # since bins actually go from [0,num_grid_points - 1], let's wrap back around
		phases_discretized_indices = phases_discretized_indices.int()

		# ** compute log prob **
		return torch.distributions.categorical.Categorical(self.theta_posterior_likelihood).log_prob(phases_discretized_indices)



	











