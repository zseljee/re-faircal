from dataset import Dataset
from argparse import Namespace

from approaches.utils import get_threshold

def uncalibrated(dataset: Dataset, conf: Namespace):
	data = {'confidences': dict(),
	        'threshold': dict(),
			'fpr': dict()
		   }

	scores, ground_truth = dataset.get_scores(train=True, include_gt=True)

	thr = get_threshold(scores, ground_truth, conf.fpr_thr)

	data['confidences']['cal'] = scores
	data['threshold']['global'] = thr

	scores, ground_truth = dataset.get_scores(train=False, include_gt=True)
	fpr = 0. # TODO compute FPR for test set

	data['confidences']['test'] = scores
	data['fpr']['global'] = fpr
	
	for subgroup in dataset.iterate_subgroups(use_attributes='ethnicity'):
		dataset.select_subgroup(**subgroup)

		scores, ground_truth = dataset.get_scores(train=True, include_gt=True)
		thr = get_threshold(scores, ground_truth, conf.fpr_thr)

		scores, ground_truth = dataset.get_scores(train=False, include_gt=True)
		fpr = 0. # TODO compute FPR for test set of subgroup

		data['threshold'][subgroup['ethnicity']] = thr
		data['fpr'][subgroup['ethnicity']] = fpr

	return data