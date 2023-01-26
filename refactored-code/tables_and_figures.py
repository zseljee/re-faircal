import numpy as np
import os
import pandas as pd
import pickle
import itertools

from argparse import Namespace
from collections import defaultdict
from IPython.core.display import HTML, display_html as display

from approaches.utils import tpr_at_fpr
from constants import EXPERIMENT_FOLDER
from utils import get_experiment_folder

teamName = 'FACT-AI'

configurations = configurations = {
    'dataset': ['rfw', 'bfw'],
    'feature': ['facenet', 'facenet-webface'],
    'approach': ['baseline', 'faircal', 'oracle', 'fsn'],
}
skip_configurations = {
    ('bfw', 'facenet'),
    ('bfw', 'facenet', 'baseline'),
    ('bfw', 'facenet', 'faircal'),
    ('bfw', 'facenet', 'oracle'),
    ('bfw', 'facenet', 'fsn'),
    ('0.1\% FPR', 'bfw', 'facenet'),
    ('1.0\% FPR', 'bfw', 'facenet'),
}


def get_metrics():
    data = dict()

    for conf in itertools.product(*configurations.values()):
        if conf in skip_configurations: continue;

        data[conf] = defaultdict(int)

        _conf = Namespace(calibration_method='beta', **dict(zip(configurations.keys(), conf)))
        exp_folder = get_experiment_folder(_conf)
        exp_folder = os.path.join(EXPERIMENT_FOLDER,_conf.dataset, _conf.feature, _conf.approach, _conf.calibration_method)
        fname = os.path.join( exp_folder , 'results.npy' )

        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                results = pickle.load(f)

            _avgLists = defaultdict(list)
            # for fold in results:
            for fold in ['fold1', 'fold2', 'fold3', 'fold4', 'fold5']:
                tpr = results[fold]['metrics']['Global']['tpr']
                fpr = results[fold]['metrics']['Global']['fpr']
                thr = results[fold]['metrics']['Global']['thr']

                # KS_list = 100.*np.array([results[fold]['metrics'][subgroup]['ks'] for subgroup in results[fold]['metrics'] if subgroup != "Global"])
                KS_list = 100.*np.array([results[fold]['metrics'][subgroup]['ks'] for subgroup in results[fold]['metrics']])
                # print([subgroup for subgroup in results[fold]['metrics']])
                KS_mean = np.mean(KS_list)

                # Threshold at which the global FPR is 0.1%
                thr_at_globalfpr_001 = np.interp(0.001, fpr, thr)
                # Threshold at which the global FPR is 1.0%
                thr_at_globalfpr_01 = np.interp(0.01, fpr, thr)

                # Now collect results for each subgroup
                fpr_at_globalfpr_001 = []
                fpr_at_globalfpr_01 = []
                for subgroup in results[fold]['metrics']:
                    if subgroup == 'Global': continue;

                    fpr_subgroup = np.fmin(results[fold]['metrics'][subgroup]['fpr'],1.)
                    thr_subgroup = np.fmin(results[fold]['metrics'][subgroup]['thr'],1.)

                    # What is the FPR for the global threshold set above?
                    fpr_at_globalfpr_001.append( np.interp(thr_at_globalfpr_001, thr_subgroup[::-1], fpr_subgroup[::-1]) )
                    fpr_at_globalfpr_01.append( np.interp(thr_at_globalfpr_01, thr_subgroup[::-1], fpr_subgroup[::-1]) )

                fpr_at_globalfpr_001 = 100.*np.array(fpr_at_globalfpr_001)
                fpr_at_globalfpr_01 = 100.*np.array(fpr_at_globalfpr_01)

                fpr_001_mean = np.mean(fpr_at_globalfpr_001)
                fpr_01_mean = np.mean(fpr_at_globalfpr_01)

                # Add all metrics to the list for this fold
                _avgLists['TPR @ 0.1\% FPR'].append( 100.*tpr_at_fpr(tpr, fpr, .001) )
                _avgLists['TPR @ 1.0\% FPR'].append( 100.*tpr_at_fpr(tpr, fpr, .01) )

                _avgLists['KS - Mean'].append( KS_mean )
                _avgLists['KS - AAD'].append( np.mean( np.abs( KS_list - KS_mean ) ) )
                _avgLists['KS - MAD'].append( np.max( np.abs( KS_list - KS_mean ) ) )
                # print(conf, np.abs( KS_list - KS_mean ) )
                _avgLists['KS - STD'].append( np.std( KS_list ) )

                _avgLists['FPR @ 0.1\% global FPR - AAD'].append( np.mean( np.abs( fpr_at_globalfpr_001 - fpr_001_mean ) ) )
                _avgLists['FPR @ 0.1\% global FPR - MAD'].append( np.max( np.abs( fpr_at_globalfpr_001 - fpr_001_mean ) ) )
                _avgLists['FPR @ 0.1\% global FPR - STD'].append( np.std( fpr_at_globalfpr_001 ) )

                _avgLists['FPR @ 1.0\% global FPR - AAD'].append( np.mean( np.abs( fpr_at_globalfpr_01 - fpr_01_mean ) ) )
                _avgLists['FPR @ 1.0\% global FPR - MAD'].append( np.max( np.abs( fpr_at_globalfpr_01 - fpr_01_mean ) ) )
                _avgLists['FPR @ 1.0\% global FPR - STD'].append( np.std( fpr_at_globalfpr_01 ) )

            for metric in _avgLists:
                data[conf][metric] = np.mean(_avgLists[metric])
        else:
            print("Could not load experiment",_conf)
            print("Please save results at",fname)
    return data

metrics = get_metrics()


if True:
    for conf in metrics:
        print(conf)
        for metric in metrics[conf]:
            print(f"\t-{metric:40}={metrics[conf][metric]:.4f}")


def rename(val):
    if isinstance(val, (list,tuple)):
        dtype = type(val)
        return dtype(rename(v) for v in val)

    renamer = dict()
    renamer['rfw'] = 'RFW'
    renamer['bfw'] = 'BFW'
    renamer['facenet'] = '\pbox{1.8cm}{FaceNet\\break(VGGFace2)}'
    renamer['facenet-webface'] = '\pbox{1.8cm}{FaceNet\\break(WebFace)}'
    renamer['baseline'] = 'Baseline'
    renamer['faircal'] = 'FairCal'
    renamer['oracle'] = 'Oracle'
    renamer['fsn'] = 'FSN'
    renamer['dataset'] = 'Dataset'
    renamer['by'] = 'By'
    renamer['metric'] = 'Metric'
    renamer['feature'] = 'Feature'
    renamer['approach'] = 'Approach'
    renamer['threshold'] = 'Thr.'
    renamer['TPR @ 0.1\% FPR'] = 'TPR @ \\break0.1\% FPR'
    renamer['TPR @ 1.0\% FPR'] = 'TPR @ \\break1.0\% FPR'
    # renamer['0.1\% FPR'] = '\pbox{0.8cm}{0.1\%\\break FPR}'
    # renamer['1.0\% FPR'] = '\pbox{0.8cm}{1.0\%\\break FPR}'
    renamer['Salvador'] = 'Sal.'
    renamer['FACT-AI'] = 'Our'
    return renamer.get(val, val)


def dictsToIndex(columns, rows):
    columnTuples = filter(lambda T: (T[:2] not in skip_configurations), itertools.product(*columns.values()))
    columnTuples = rename(list(columnTuples))
    columnNames = rename(list(columns.keys()))
    columnIndex = pd.MultiIndex.from_tuples(columnTuples, names=columnNames)

    rowTuples = filter(lambda T: not all(x in T for x in ['bfw','facenet']), itertools.product(*rows.values()) )
    rowTuples = rename(list(rowTuples))

    rowNames = rename(list(rows.keys()))
    rowIndex = pd.MultiIndex.from_tuples(rowTuples, names=rowNames)

    return columnIndex, rowIndex


def gen_table_accuracy():
    # Data from Table 2 in Salvador (2022), excluding AUROC columns (order is kept)
    data_salvador = np.array([
        [18.42, 34.88,  11.18, 26.04,  33.61, 58.87,  ], # Baseline
        [23.55, 41.88,  20.64, 33.13,  46.74, 69.21,  ], # FairCal
        [21.40, 41.83,  16.71, 31.60,  45.13, 67.56,  ], # Oracle
        [23.01, 40.21,  17.33, 32.80,  47.11, 68.92,  ], # FSN
    ]).T # <-- ! Note the transpose!
    data_salvador = data_salvador
    data_factai = np.full_like(data_salvador, np.nan)

    columns = {
        # 'approach': ['faircal', 'oracle'],
        'approach': ['baseline', 'faircal', 'oracle', 'fsn'],
        'by': ['Salvador', teamName, 'diff.'],
    }
    rows = {
        'dataset': ['rfw', 'bfw'],
        'feature': ['facenet', 'facenet-webface'],
        'TPR @ ': ['0.1\% FPR', '1.0\% FPR'],
    }

    i=0
    for dataset, feature in itertools.product(rows['dataset'], rows['feature']):
        if (dataset,feature) in skip_configurations: continue;

        j=0
        for approach in columns['approach']:
            conf = (dataset, feature, approach)
            if conf in skip_configurations: continue;

            data_factai[i,j] = metrics[conf]['TPR @ 0.1\% FPR']
            data_factai[i+1,j] = metrics[conf]['TPR @ 1.0\% FPR']

            j+=1
        i+=2

    columnIndex, rowIndex = dictsToIndex(columns, rows)

    data = np.zeros((len(rowIndex), len(columnIndex)))

    data[:,::3] = data_salvador
    data[:,1::3] = data_factai
    data[:,2::3] = data_factai - data_salvador

    df = pd.DataFrame(data, columns=columnIndex, index=rowIndex)

    dif_col_formatter = dict((col, '<b>{:+.2f}</b>') for col in df.columns if col[-1] == 'diff.')
    styler = df.style.format(dif_col_formatter, na_rep='TBD', precision=2, escape="latex")
    display(HTML(styler.to_html()))


    caption = "Global accuracy measured by TPR at several FPR thresholds, comparing the original results (Sal.) with ours. (Higher is better.)"
    dif_col_formatter = dict((col, '{:+.2f}') for col in df.columns if col[-1] == 'diff.')
    styler = df.style.format(dif_col_formatter, na_rep='TBD', precision=2)
    ncolblocks = len(columnIndex)//3

    fname = os.path.join(EXPERIMENT_FOLDER, 'table_accuracy.tex')
    print("Saving to",fname)
    with open(fname, 'w') as f:
        table = styler.to_latex(
            None,
            multicol_align='c|',
            hrules=True,
            column_format='lp{1.6cm}p{1.6cm}|'+('rrr|'*ncolblocks),
            caption=caption,
            label="tab:accuracy",
            position_float='centering',
            clines='skip-last;index',
        )
        table = table.replace(r"\cline", r"\cmidrule")
        if resizebox:=True:
            table = table.replace(r"\begin{tabular}", "\\resizebox{\columnwidth}{!}{\n\\begin{tabular}")
            table = table.replace(r"\end{tabular}", "\\end{tabular}\n}")
        f.write(table)

gen_table_accuracy()


def gen_table_fairness(full=True):
    """
    full determines whether the AAD and MAD should be include
    """
    # Data from Table 3 in Salvador (2022)
    data_salvador = np.array([
        [6.37, 2.89, 5.73, 3.77, 5.55, 2.48, 4.97, 2.91, 6.77, 3.63, 5.96, 4.03, ], # Baseline
        [1.37, 0.28, 0.50, 0.34, 1.75, 0.41, 0.64, 0.45, 3.09, 1.34, 2.48, 1.55, ], # FairCal
        [1.18, 0.28, 0.53, 0.33, 1.35, 0.38, 0.66, 0.43, 2.23, 1.15, 2.63, 1.40, ], # Oracle
        [1.43, 0.35, 0.57, 0.40, 2.49, 0.84, 1.19, 0.91, 2.76, 1.38, 2.67, 1.60, ], # FSN
    ]).T # <-- ! Note the Transpose
    if not full:
        data_salvador = data_salvador[[0, 3, 4, 7, 8, 11]]
    data_factai = np.full_like(data_salvador, np.nan)

    columns = {
        # 'approach': ['faircal', 'oracle'],
        'approach': ['baseline', 'faircal', 'oracle', 'fsn'],
        'by': ['Salvador', teamName, 'diff.'],
    }
    metrics_to_include = ['Mean', 'AAD', 'MAD', 'STD'] if full else ['Mean', 'STD']
    rows = {
        'dataset': ['rfw', 'bfw'],
        'feature': ['facenet', 'facenet-webface'],
        'metric': metrics_to_include,
    }
    columnIndex, rowIndex = dictsToIndex(columns, rows)

    i=0
    for dataset, feature in itertools.product(rows['dataset'], rows['feature']):
        if (dataset,feature) in skip_configurations: continue;

        j=0
        for approach in columns['approach']:
            conf = (dataset, feature, approach)
            if conf in skip_configurations: continue;

            if full:
                data_factai[i+0,j] = metrics[conf]['KS - Mean']
                data_factai[i+1,j] = metrics[conf]['KS - AAD']
                data_factai[i+2,j] = metrics[conf]['KS - MAD']
                data_factai[i+3,j] = metrics[conf]['KS - STD']
            else:
                data_factai[i+0,j] = metrics[conf]['KS - Mean']
                data_factai[i+1,j] = metrics[conf]['KS - STD']

            j+=1
        i+=len(metrics_to_include)
    data = np.zeros((len(rowIndex), len(columnIndex)))
    data[:,::3] = data_salvador
    data[:,1::3] = data_factai
    data[:,2::3] = data_factai-data_salvador

    df = pd.DataFrame(data, columns=columnIndex, index=rowIndex)
    # if not full:
    #     df = df[df["Metric"].isin(['Mean', 'STD'])]

    dif_col_formatter = dict((col, '<b>{:+.2f}</b>') for col in df.columns if col[-1] == 'diff.')
    styler = df.style.format(dif_col_formatter, na_rep='TBD', precision=2, escape="latex")
    display(HTML(styler.to_html()))


    caption = "Fairness calibration measured by the mean KS across the sensitive subgroups. Showing the Mean, Average Absolute Deviation (AAD), Maximum Absolute Deviation (MAD) and Standard Deviation (STD). Comparing original results (Sal.) with ours. (Lower is better in all cases)"
    print(caption)
    dif_col_formatter = dict((col, '{:+.2f}') for col in df.columns if col[-1] == 'diff.')
    styler = df.style.format(dif_col_formatter, na_rep='TBD', precision=2)
    ncolblocks = len(columnIndex)//3

    fname = os.path.join(EXPERIMENT_FOLDER, 'table_fairness.tex' if full else 'table_fairness_partial.tex')
    print("Saving to",fname)
    with open(fname, 'w') as f:
        table = styler.to_latex(
            None,
            multicol_align='c|',
            hrules=True,
            column_format='lp{1.6cm}p{1.6cm}|'+('rrr|'*ncolblocks),
            caption=caption,
            label="tab:fairness-full" if full else "tab:fairness",
            position_float='centering',
            clines='skip-last;index',
        )
        table = table.replace(r"\cline", r"\cmidrule")
        if resizebox:=True:
            table = table.replace(r"\begin{tabular}", "\\resizebox{\columnwidth}{!}{\n\\begin{tabular}")
            table = table.replace(r"\end{tabular}", "\\end{tabular}\n}")
        f.write(table)

gen_table_fairness()
gen_table_fairness(full=False)


def gen_table_predictive_equality(full=True):
    # Data from Table 4 in Salvador (2022)
    data_salvador = np.array([
        [0.10, 0.15, 0.10,  0.14, 0.26, 0.16,  0.29, 1.00, 0.40,   0.68, 1.02, 0.74,  0.67, 1.23, 0.79,  2.42, 7.48, 3.22, ], # Baseline
        [0.09, 0.14, 0.10,  0.09, 0.16, 0.10,  0.09, 0.20, 0.11,   0.28, 0.46, 0.32,  0.29, 0.57, 0.35,  0.80, 1.79, 0.95,], # FairCal
        [0.11, 0.19, 0.12,  0.11, 0.20, 0.13,  0.12, 0.25, 0.15,   0.40, 0.69, 0.45,  0.41, 0.74, 0.48,  0.77, 1.71, 0.91, ], # Oracle
        [0.10, 0.18, 0.11,  0.11, 0.23, 0.13,  0.09, 0.20, 0.11,   0.37, 0.68, 0.46,  0.35, 0.61, 0.40,  0.87, 2.19, 1.05, ], # FSN
    ]).T # <-- ! Note the transpose
    if not full:
        data_salvador = data_salvador[[2, 5, 8, 11, 14, 17]]

    data_factai = np.full_like(data_salvador, np.nan)

    columns = {
        # 'approach': ['faircal', 'oracle'],
        'approach': ['baseline', 'faircal', 'oracle', 'fsn'],
        'by': ['Salvador', teamName, 'diff.'],
    }
    metrics_to_include = ['AAD', 'MAD', 'STD'] if full else ['STD']
    rows = {
        'threshold': ['0.1\% FPR','1.0\% FPR'],
        'dataset': ['rfw', 'bfw'],
        'feature': ['facenet', 'facenet-webface'],
        'metric': metrics_to_include,
    }

    i=0
    for dataset, feature in itertools.product(rows['dataset'], rows['feature']):
        if (dataset,feature) in skip_configurations: continue;

        j=0
        for approach in columns['approach']:
            conf = (dataset, feature, approach)

            if full:
                data_factai[i+0,j] = metrics[conf]['FPR @ 0.1\% global FPR - AAD']
                data_factai[i+1,j] = metrics[conf]['FPR @ 0.1\% global FPR - MAD']
                data_factai[i+2,j] = metrics[conf]['FPR @ 0.1\% global FPR - STD']

                data_factai[i+9,j] = metrics[conf]['FPR @ 1.0\% global FPR - AAD']
                data_factai[i+10,j] = metrics[conf]['FPR @ 1.0\% global FPR - MAD']
                data_factai[i+11,j] = metrics[conf]['FPR @ 1.0\% global FPR - STD']
            else:
                data_factai[i+0,j] = metrics[conf]['FPR @ 0.1\% global FPR - STD']
                data_factai[i+3,j] = metrics[conf]['FPR @ 1.0\% global FPR - STD']

            j+=1
        i+=3 if full else 1

    columnIndex, rowIndex = dictsToIndex(columns, rows)

    data = np.zeros((len(rowIndex), len(columnIndex)))
    data[:,::3] = data_salvador
    data[:,1::3] = data_factai
    data[:,2::3] = data_factai-data_salvador

    df = pd.DataFrame(data, columns=columnIndex, index=rowIndex)

    dif_col_formatter = dict((col, '<b>{:+.2f}</b>') for col in df.columns if col[-1] == 'diff.')
    styler = df.style.format(dif_col_formatter, na_rep='TBD', precision=2, escape="latex")
    display(HTML(styler.to_html()))


    caption = "Predictive equality: For two choices of global FPR compare the deviations in subgroup FPRs in terms of Average Absolute Deviation (AAD), Maximum Absolute Deviation (MAD), and Standard Deviation (STD). Comparing original results (Sal.) with ours. (Lower is better in all cases)"
    dif_col_formatter = dict((col, '{:+.2f}') for col in df.columns if col[-1] == 'diff.')
    styler = df.style.format(dif_col_formatter, na_rep='TBD', precision=2)
    ncolblocks = len(columnIndex)//3

    fname = os.path.join(EXPERIMENT_FOLDER, 'table_predictive_equality.tex' if full else 'table_predictive_equality_partial.tex')
    print("Saving to",fname)
    with open(fname, 'w') as f:
        table = styler.to_latex(
            None,
            multicol_align='c|',
            hrules=True,
            column_format='llp{1.6cm}p{1.6cm}|'+('rrr|'*ncolblocks),
            caption=caption,
            label="tab:predictive-equality-full" if full else "tab:predictive-equality",
            position_float='centering',
            clines='skip-last;index',
        )
        table = table.replace(r"\cline", r"\cmidrule")
        if resizebox:=True:
            table = table.replace(r"\begin{tabular}", "\\resizebox{\columnwidth}{!}{\n\\begin{tabular}")
            table = table.replace(r"\end{tabular}", "\\end{tabular}\n}")
        f.write(table)

gen_table_predictive_equality()

gen_table_predictive_equality(full=False)
