

##########################################################################################################################################
###  GETTING RID OF THIS ERROR: https://stackoverflow.com/questions/71106940/cannot-import-name-centered-from-scipy-signal-signaltools ###
##########################################################################################################################################

import  scipy.signal.signaltools

def _centered(arr, newsize):
	# Return the center newsize portion of the array.
	newsize = np.asarray(newsize)
	currsize = np.array(arr.shape)
	startind = (currsize - newsize) // 2
	endind = startind + newsize
	myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
	return arr[tuple(myslice)]

scipy.signal.signaltools._centered = _centered