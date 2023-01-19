import numpy as np

# from calibrationMethods import BetaCalibration

def baseline(dataset, conf):
	scores, ground_truth = dataset.get_scores(include_gt=True)
	# cal = BetaCalibration(scores=scores,
	#                       ground_truth=ground_truth,
	# 					  score_min=-1,
	# 					  score_max=1)
	return {}