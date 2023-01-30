from argparse import Namespace
import pickle
import numpy as np
import pandas as pd
import os

from approaches.utils import tpr_at_fpr
from utils import get_experiment_folder, iterate_configurations

def conf2idx(conf: Namespace):
    # approach determines row
    i = ['baseline', 'faircal', 'oracle'].index(conf.approach)

    # First 4 columns are RFW with FaceNet(VGGFace2) and FaceNet(WebFace)
    if conf.dataset == 'rfw':
        j = ['facenet', 'facenet-webface'].index(conf.feature) * 2 # There are 2 columns per feature

    # Starting from index 4, BFW is with FaceNet(WebFace) and ArcFace
    else:
        j = 4+['facenet-webface', 'arcface'].index(conf.feature) * 2 # 2 columns per feature

    # Even column are TPR@.1%FPR, odd columns are TPR@1%FPR
    j += [0.001, .01].index(conf.target_fpr)

    return i,j

def table_accuracy():
    teamName = 'FACT-AI' # What do we call ourselves?
    columnNames = [
        # TODO
        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 0.1\% FPR', 'Salvador'),
        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 0.1\% FPR', teamName),
        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 0.1\% FPR', 'diff.'),

        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 1.0\% FPR', 'Salvador'),
        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 1.0\% FPR', teamName),
        ('RFW', 'FaceNet (VGGFace2)', 'TPR @ 1.0\% FPR', 'diff.'),

        ('RFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', 'Salvador'),
        ('RFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', teamName),
        ('RFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', 'diff.'),

        ('RFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', 'Salvador'),
        ('RFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', teamName),
        ('RFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', 'diff.'),


        # ('BFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', 'Salvador'),
        # ('BFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', teamName),
        # ('BFW', 'FaceNet (WebFace)', 'TPR @ 0.1\% FPR', 'diff.'),

        # ('BFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', 'Salvador'),
        # ('BFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', teamName),
        # ('BFW', 'FaceNet (WebFace)', 'TPR @ 1.0\% FPR', 'diff.'),

        # ('BFW', 'ArcFace', 'TPR @ 0.1\% FPR', 'Salvador'),
        # ('BFW', 'ArcFace', 'TPR @ 0.1\% FPR', teamName),
        # ('BFW', 'ArcFace', 'TPR @ 0.1\% FPR', 'diff.'),

        # ('BFW', 'ArcFace', 'TPR @ 1.0\% FPR', 'Salvador'),
        # ('BFW', 'ArcFace', 'TPR @ 1.0\% FPR', teamName),
        # ('BFW', 'ArcFace', 'TPR @ 1.0\% FPR', 'diff.'),
    ]
    columnIndex = pd.MultiIndex.from_tuples(columnNames, names=["Dataset", "Feature", "Metric", "By"])

    # Data from Table 2 in Salvador (2022)
    data_salvador = np.array([
        [18.42, 34.88, 11.18, 26.04, 33.61, 58.87, 86.27, 90.11], # Baseline TODO
        [23.55, 41.88, 20.64, 33.13, 46.74, 69.21, 86.28, 90.14], # FairCal TODO
        [21.40, 41.83, 16.71, 31.60, 45.13, 67.56, 86.41, 90.40], # Oracle TODO
    ])

    data_factai = np.full((3,8), np.nan)

    # Iterate these configurations
    configurations = {
        'approach': ['baseline', 'faircal', 'oracle'],
        'dataset': ['rfw', 'bfw'],
        'feature': ['facenet', 'facenet-webface', 'arcface'],
        'target_fpr': [.1/100., 1./100.],
    }

    for conf in iterate_configurations(Namespace(**configurations), keys=configurations.keys()):
        # ArcFace has seen RFW images, FaceNet(VGGFace2) has seen BFW
        if (conf.dataset == 'rfw' and conf.feature == 'arcface')\
        or (conf.dataset == 'bfw' and conf.feature == 'facenet'):
            continue

        conf.calibration_method = "beta" # For exp folder, no other calMethod is used
        exp_folder = get_experiment_folder(conf, makedirs=False)

        fname = os.path.join(exp_folder, 'results.npy')
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                data = pickle.load(f)
            
            tprs = []
            for fold in data:
                tpr = data[fold]['metrics']['Global']['tpr']
                fpr = data[fold]['metrics']['Global']['fpr']

                tprs.append( tpr_at_fpr(tpr, fpr, conf.target_fpr) )

            i,j = conf2idx(conf)
            data_factai[i,j] = 100.*np.mean(tprs)
        else:
            print("Could not find results for experiment", conf)
            print("- Please save results at", fname)

    data = np.zeros((3,24))
    data[:,::3] = data_salvador
    data[:,1::3] = data_factai
    data[:,2::3] = data[:,1::3]-data[:,::3]

    data = data[:,:12]

    df = pd.DataFrame(data, columns=columnIndex, index=['Baseline', 'FairCal', 'Oracle'])

    dif_col_formatter = dict((col, '{:+5.1f}'.format) for col in df.columns if col[-1] == 'diff.')
    print(df.to_string(formatters=dif_col_formatter,float_format="{:.2f}".format, na_rep='TBD'))

    caption = "Accuracy measured by TPR at two different FPR thresholds for original and reproduced approaches."
    dif_col_formatter = dict((col, '{:+5.1f}') for col in df.columns if col[-1] == 'diff.')
    styler = df.style.format(dif_col_formatter, na_rep='TBD', precision=2, escape="latex")#.format("NA{.2f}",subset=[3])
    
    with open('table_accuracy.tex', 'w') as f:
        styler.to_latex(
            f, 
            multicol_align='c|', 
            hrules=True, 
            column_format='l|'+('rrr|'*4),
            caption=caption
        )