import numpy as np

from dataset import Dataset
from argparse import Namespace

from calibrationMethods import BetaCalibration
from approaches.utils import get_threshold

def baseline(dataset: Dataset, conf: Namespace):
	data = {'confidences': dict(),
	        'threshold': dict(),
			'fpr': dict()
		   }

	print("Calibrating global scores...")
	scores, ground_truth = dataset.get_scores(train=True, include_gt=True)

	calibrator = BetaCalibration(scores=scores,
	                             ground_truth=ground_truth,
					       	     score_min=-1,
						         score_max=1,
								)
	calibrated_scores = calibrator.predict(scores)
	data['confidences']['cal'] = calibrated_scores

	thr = get_threshold(calibrated_scores, ground_truth, conf.fpr_thr)
	data['threshold']['global'] = thr


	scores, ground_truth = dataset.get_scores(train=False, include_gt=True)

	calibrated_scores = calibrator.predict(scores)
	data['confidences']['test'] = calibrated_scores

	fpr = 0. # TODO compute FPR for test set
	data['fpr']['global'] = fpr

	
	print("Calibrating subgroup scores...")
	for subgroup in dataset.iterate_subgroups(use_attributes='ethnicity'):
		dataset.select_subgroup(**subgroup)

		scores, ground_truth = dataset.get_scores(train=True, include_gt=True)
		calibrated_scores = calibrator.predict(scores)
		thr = get_threshold(calibrated_scores, ground_truth, conf.fpr_thr)

		scores, ground_truth = dataset.get_scores(train=False, include_gt=True)
		calibrated_scores = calibrator.predict(scores)
		fpr = 0. # TODO compute FPR for test set of subgroup

		data['threshold'][subgroup['ethnicity']] = thr
		data['fpr'][subgroup['ethnicity']] = fpr

	return data