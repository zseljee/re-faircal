import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
import pickle
import itertools

from argparse import ArgumentParser, Namespace
from collections import defaultdict
from IPython.core.display import HTML, display_html as display
from sklearn.metrics import roc_curve

from approaches.utils import tpr_at_fpr
from constants import EXPERIMENT_FOLDER, DATA_FOLDER
from utils import get_experiment_folder

parser = ArgumentParser()
parser.add_argument("--use-arcface", action="store_true")
args = parser.parse_args()
EXCLUDE_ARCFACE = not args.use_arcface
del parser
del args
print("Ignoring non-existent ArcFace results" if EXCLUDE_ARCFACE else "Using results for ArcFace")

teamName = 'FACT-AI'

configurations = {
    'dataset': ['rfw', 'bfw'],
    'feature': ['facenet', 'facenet-webface', 'arcface'],
    'approach': ['uncalibrated', 'baseline', 'faircal', 'oracle', 'fsn', 'ftc'],
}
skip_configurations = {
    # bfw-facenet
    ('bfw', 'facenet'),
    ('bfw', 'facenet', 'uncalibrated'),
    ('bfw', 'facenet', 'baseline'),
    ('bfw', 'facenet', 'ftc'),
    ('bfw', 'facenet', 'fsn'),
    ('bfw', 'facenet', 'faircal'),
    ('bfw', 'facenet', 'oracle'),
    # rfw-arcface
    ('rfw', 'arcface'),
    ('rfw', 'arcface', 'uncalibrated'),
    ('rfw', 'arcface', 'baseline'),
    ('rfw', 'arcface', 'ftc'),
    ('rfw', 'arcface', 'fsn'),
    ('rfw', 'arcface', 'faircal'),
    ('rfw', 'arcface', 'oracle'),
    # other
    ('0.1\% FPR', 'bfw', 'facenet'),
    ('1.0\% FPR', 'bfw', 'facenet'),
}


def aad(values):
    values = np.array(values)
    mean = values.mean()
    return np.abs( values - mean ).mean()

def mad(values):
    values = np.array(values)
    mean = values.mean()
    return np.abs( values - mean ).max()

def get_metrics_fold(results):
    # ! In percentages!
    tpr = 100.*results['Global']['tpr']
    fpr = 100.*results['Global']['fpr']
    thr = results['Global']['thr']

    assert all( np.diff(fpr) >= 0 )

    # Thresholds for 0.1% and 1% global FPR
    thr_1fpr = np.interp(1.0, fpr, thr)
    thr_01fpr = np.interp(0.1, fpr, thr)

    ks = []
    fpr_1fpr = []
    fpr_01fpr = []

    for subgroup in results:
        if subgroup == 'Global':
            continue;

        ks.append(100.*results[subgroup]['ks'])

        fpr_subgroup = np.fmin(1.,results[subgroup]['fpr'])[::-1] *100. # ! In percentages!
        thr_subgroup = np.fmin(1.,results[subgroup]['thr'])[::-1]

        fpr_1fpr.append( np.interp(thr_1fpr, thr_subgroup, fpr_subgroup) )
        fpr_01fpr.append( np.interp(thr_01fpr, thr_subgroup, fpr_subgroup) )

    ks,fpr_1fpr,fpr_01fpr = map(np.array, [ks,fpr_1fpr,fpr_01fpr])

    return {
        'TPR @ 0.1\% FPR': np.interp(0.1, fpr, tpr),
        'TPR @ 1.0\% FPR': np.interp(1.0, fpr, tpr),

        'KS - Mean': ks.mean(),
        'KS - AAD': aad( ks ),
        'KS - MAD': mad( ks ),
        'KS - STD': np.std( ks ),

        'FPR @ 0.1\% global FPR - AAD': aad( fpr_01fpr ),
        'FPR @ 0.1\% global FPR - MAD': mad( fpr_01fpr ),
        'FPR @ 0.1\% global FPR - STD': np.std( fpr_01fpr ),

        'FPR @ 1.0\% global FPR - AAD': aad( fpr_1fpr ),
        'FPR @ 1.0\% global FPR - MAD': mad( fpr_1fpr ),
        'FPR @ 1.0\% global FPR - STD': np.std( fpr_1fpr ),
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

            folds = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5'] if True else ['fold1']

            metrics_per_fold = defaultdict(list)
            for fold in folds:

                metrics_fold = get_metrics_fold( results[fold]['metrics'] )
                for metric in metrics_fold:
                    metrics_per_fold[metric].append( metrics_fold[metric] )

            # Average over each fold
            for metric in metrics_per_fold:
                data[conf][metric] = np.mean( metrics_per_fold[metric] )

            data[conf]['score'] = np.vstack([results[fold]['scores'] for fold in folds])
        else:
            print("Could not load experiment", _conf)
            print("Please save results at", fname)
    return data

metrics = get_metrics()


if False:
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
    renamer['arcface'] = 'ArcFace'
    renamer['baseline'] = 'Baseline'
    renamer['ftc'] = 'FTC'
    renamer['fsn'] = 'FSN'
    renamer['faircal'] = 'FairCal'
    renamer['oracle'] = 'Oracle'
    renamer['dataset'] = 'Dataset'
    renamer['by'] = r'\hfill By \scriptsize $\rightarrow$'
    renamer['metric'] = r'Metric \scriptsize $\downarrow$'
    renamer['TPR @'] = r'TPR @ \scriptsize $\downarrow$'
    renamer['TPR @ '] = r'TPR @ \scriptsize $\downarrow$'
    renamer['feature'] = 'Feature'
    renamer['approach'] = r'\hfill Approach \scriptsize $\rightarrow$'
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

    rowTuples = filter(lambda T: not all(x in T for x in ['bfw','facenet']) and not all(x in T for x in ['rfw','arcface']), itertools.product(*rows.values()) )
    rowTuples = rename(list(rowTuples))

    rowNames = rename(list(rows.keys()))
    rowIndex = pd.MultiIndex.from_tuples(rowTuples, names=rowNames)

    return columnIndex, rowIndex


def fill_data(data_salvador, rows, metric_names, metrics_dict, metric_to_idx: dict[str, int]):
    """fills a numpy array in the shape of data_salvador with data from metrics.
    The metrics are taken from the keys of metric_to_idx and put at an offset of the value.
    Creates a dataframe from it in the end that also contains the difference.
    """
    # Meta-information setup
    columns = {
        'approach': ['baseline', 'ftc', 'fsn', 'faircal', 'oracle'],
        'by': ['Salvador', teamName, 'diff.'],
    }
    columnIndex, rowIndex = dictsToIndex(columns, rows)

    # Collect our results
    data_factai = np.full_like(data_salvador, np.nan)
    i=0
    for dataset, feature in itertools.product(rows['dataset'], rows['feature']):
        if (dataset,feature) in skip_configurations: continue;

        for j, approach in enumerate(columns['approach']):
            conf = (dataset, feature, approach)
            for key, offset in metric_to_idx.items():
                # In case the code doesn't work by auto-returning 0 for non-existing results,
                # uncomment the following line.  I wasn't able to test this.  (TODO)
                # data_factai[i+offset, j] = metrics_dict[conf].get(key, float("nan"))
                data_factai[i+offset, j] = metrics_dict[conf][key]

        i += len(metric_names)

    # Collect data into single dataframe
    data = np.zeros((len(rowIndex), len(columnIndex)))
    data[:,::3] = data_salvador
    data[:,1::3] = data_factai
    data[:,2::3] = data_factai - data_salvador
    df = pd.DataFrame(data, columns=columnIndex, index=rowIndex)

    if EXCLUDE_ARCFACE:
        arcface_rows = df.index.get_level_values("Feature") == "ArcFace"
        df = df.iloc[~arcface_rows]

    return df


def show_and_write_table(table_df: pd.DataFrame, caption: str, label: str, save_as: str):
    """Write the table to the save_as file with the given caption and label.
    """
    # Only useful in Jupyter...
    dif_col_formatter = dict((col, '<b>{:+.2f}</b>') for col in table_df.columns if col[-1] == 'diff.')
    styler = table_df.style.format(dif_col_formatter, na_rep='TBD', precision=2, escape="latex")
    display(HTML(styler.to_html()))

    # Meta-setup
    dif_col_formatter = dict((col, '{:+.2f}') for col in table_df.columns if col[-1] == 'diff.')
    styler = table_df.style.format(dif_col_formatter, na_rep='TBD', precision=2)
    ncolblocks = len(table_df.columns)//3

    fname = os.path.join(EXPERIMENT_FOLDER, save_as)
    print("Saving to",fname)
    table_code: str = styler.to_latex(
        None,
        multicol_align='c|',
        hrules=True,
        column_format='l'*(table_df.index.nlevels - 2)+'p{1.6cm}p{1.9cm}|*{'+str(ncolblocks)+'}{r@{\hspace{2.5mm}}r@{\hspace{2.5mm}}r|}',
        caption=caption,
        label=label,
        position_float='centering',
        clines='skip-last;index',
    )
    # Some post-processing of the generated table
    table_code: list[str] = table_code.split("\n")
    last_cline = -1
    for i, line in enumerate(table_code):
        if line.startswith(r"\cline"):
            last_cline = i
            # Drop multiple clines, because that happens by default
            cline_end = line.index("}")
            table_code[i] = line[:cline_end + 1]
    # Remove cline from bottom row, as there's a \bottomrule there
    table_code.pop(last_cline)
    table_code: str = "\n".join(table_code)
    table_code = table_code.replace(r"\cline", r"\cmidrule")
    # Add box surrounding table to auto-fit to width of text
    if resizebox:=True:
        # the % is needed to remove some extra space around the edges of the table
        table_code = table_code.replace(r"\begin{tabular}", "\\resizebox{\columnwidth}{!}{%\n\\begin{tabular}")
        table_code = table_code.replace(r"\end{tabular}", "\\end{tabular}%\n}")
    # Save table
    with open(fname, 'w') as f:
        f.write(table_code)


def gen_table_accuracy():
    # Data from Table 2 in Salvador (2022), excluding AUROC columns (order is kept)
    data_salvador = np.array([
        [18.42, 34.88,  11.18, 26.04,  33.61, 58.87,  86.27, 90.11], # Baseline
        [ 6.86, 23.66,   4.65, 18.40,  13.60, 43.09,  82.09, 88.24], # FTC
        [23.01, 40.21,  17.33, 32.80,  47.11, 68.92,  86.19, 90.06], # FSN
        [23.55, 41.88,  20.64, 33.13,  46.74, 69.21,  86.28, 90.14], # FairCal
        [21.40, 41.83,  16.71, 31.60,  45.13, 67.56,  86.41, 90.40], # Oracle
    ]).T # <-- ! Note the transpose!

    metric_names = ['0.1\% FPR', '1.0\% FPR']
    rows = {
        'dataset': ['rfw', 'bfw'],
        'feature': ['facenet', 'facenet-webface', 'arcface'],
        'TPR @ ': metric_names,
    }

    metrics_to_idx = {
        'TPR @ 0.1\% FPR': 0,
        'TPR @ 1.0\% FPR': 1,
    }

    df = fill_data(data_salvador, rows, metric_names, metrics, metrics_to_idx)

    show_and_write_table(df,
                         caption = "Global accuracy measured by TPR at several FPR thresholds, comparing the original results (Sal.) with ours. (Higher is better.)",
                         label="tab:accuracy",
                         save_as = 'table_accuracy.tex'
    )

gen_table_accuracy()


def gen_table_fairness(full=True):
    """
    full determines whether the AAD and MAD should be include
    """
    # Data from Table 3 in Salvador (2022)
    data_salvador = np.array([
        [6.37, 2.89, 5.73, 3.77,  5.55, 2.48, 4.97, 2.91,  6.77, 3.63, 5.96, 4.03,  2.57, 1.39, 2.94, 1.63], # Baseline
        [5.69, 2.32, 4.51, 2.95,  4.73, 1.93, 3.86, 2.28,  6.64, 2.80, 5.61, 3.27,  2.95, 1.48, 3.03, 1.74], # FTC
        [1.43, 0.35, 0.57, 0.40,  2.49, 0.84, 1.19, 0.91,  2.76, 1.38, 2.67, 1.60,  2.65, 1.45, 3.23, 1.71], # FSN
        [1.37, 0.28, 0.50, 0.34,  1.75, 0.41, 0.64, 0.45,  3.09, 1.34, 2.48, 1.55,  2.49, 1.30, 2.68, 1.52], # FairCal
        [1.18, 0.28, 0.53, 0.33,  1.35, 0.38, 0.66, 0.43,  2.23, 1.15, 2.63, 1.40,  1.41, 0.59, 1.30, 0.69], # Oracle
    ]).T # <-- ! Note the Transpose
    if not full:
        data_salvador = data_salvador[[0, 3, 4, 7, 8, 11, 12, 15]]

    metric_names = ['Mean', 'AAD', 'MAD', 'STD'] if full else ['Mean', 'STD']
    rows = {
        'dataset': ['rfw', 'bfw'],
        'feature': ['facenet', 'facenet-webface', 'arcface'],
        'metric': metric_names,
    }

    if full:
        metrics_to_idx = {
            'KS - Mean': 0,
            'KS - AAD': 1,
            'KS - MAD': 2,
            'KS - STD': 3,
        }
    else:
        metrics_to_idx = {
            'KS - Mean': 0,
            'KS - STD': 1,
        }

    df = fill_data(data_salvador, rows, metric_names, metrics, metrics_to_idx)

    if full:
        caption = "Fairness calibration measured by the mean KS across the sensitive subgroups. Showing the Mean, Average Absolute Deviation (AAD), Maximum Absolute Deviation (MAD) and Standard Deviation (STD). Comparing original results (Sal.) with ours. (Lower is better in all cases.)"
    else:
        caption = "Fairness calibration measured by the mean KS across the sensitive subgroups. Showing the Mean and Standard Deviation (STD). Comparing original results (Sal.) with ours. (Lower is better in all cases.)"
    show_and_write_table(df,
                         caption = caption,
                         label="tab:fairness-full" if full else "tab:fairness",
                         save_as = 'table_fairness.tex' if full else 'table_fairness_partial.tex',
    )

gen_table_fairness()
gen_table_fairness(full=False)


def gen_table_predictive_equality(full=True):
    """
    full determines whether the AAD and MAD should be include
    """
    # Data from Table 4 in Salvador (2022)
    data_salvador = np.array([
        [0.10, 0.15, 0.10,  0.14, 0.26, 0.16,  0.29, 1.00, 0.40,  0.12, 0.30, 0.15,   0.68, 1.02, 0.74,  0.67, 1.23, 0.79,  2.42, 7.48, 3.22,  0.72, 1.51, 0.85], # Baseline
        [0.10, 0.15, 0.11,  0.12, 0.23, 0.14,  0.24, 0.74, 0.32,  0.09, 0.20, 0.11,   0.60, 0.91, 0.66,  0.54, 1.05, 0.66,  1.94, 5.74, 2.57,  0.54, 1.04, 0.61], # FTC
        [0.10, 0.18, 0.11,  0.11, 0.23, 0.13,  0.09, 0.20, 0.11,  0.11, 0.28, 0.14,   0.37, 0.68, 0.46,  0.35, 0.61, 0.40,  0.87, 2.19, 1.05,  0.55, 1.27, 0.68], # FSN
        [0.09, 0.14, 0.10,  0.09, 0.16, 0.10,  0.09, 0.20, 0.11,  0.11, 0.31, 0.15,   0.28, 0.46, 0.32,  0.29, 0.57, 0.35,  0.80, 1.79, 0.95,  0.63, 1.46, 0.78], # FairCal
        [0.11, 0.19, 0.12,  0.11, 0.20, 0.13,  0.12, 0.25, 0.15,  0.12, 0.27, 0.14,   0.40, 0.69, 0.45,  0.41, 0.74, 0.48,  0.77, 1.71, 0.91,  0.83, 2.08, 1.07], # Oracle
    ]).T # <-- ! Note the transpose
    if not full:
        data_salvador = data_salvador[[2, 5, 8, 11, 14, 17, 20, 23]]

    metric_names = ['AAD', 'MAD', 'STD'] if full else ['STD']
    rows = {
        'threshold': ['0.1\% FPR','1.0\% FPR'],
        'dataset': ['rfw', 'bfw'],
        'feature': ['facenet', 'facenet-webface', 'arcface'],
        'metric': metric_names,
    }

    if full:
        metrics_to_idx = {
            'FPR @ 0.1\% global FPR - AAD': 0,
            'FPR @ 0.1\% global FPR - MAD': 1,
            'FPR @ 0.1\% global FPR - STD': 2,
            'FPR @ 1.0\% global FPR - AAD': 12,
            'FPR @ 1.0\% global FPR - MAD': 13,
            'FPR @ 1.0\% global FPR - STD': 14,
        }
    else:
        metrics_to_idx = {
            'FPR @ 0.1\% global FPR - STD': 0,
            'FPR @ 1.0\% global FPR - STD': 4,
        }

    df = fill_data(data_salvador, rows, metric_names, metrics, metrics_to_idx)

    if full:
        caption = "Predictive equality: For two choices of global FPR compare the deviations in subgroup FPRs in terms of Average Absolute Deviation (AAD), Maximum Absolute Deviation (MAD), and Standard Deviation (STD). Comparing original results (Sal.) with ours. (Lower is better in all cases.)"
    else:
        caption = "Predictive equality: For two choices of global FPR compare the deviations in subgroup FPRs in terms of Standard Deviation (STD). Comparing original results (Sal.) with ours. (Lower is better in all cases.)"
    show_and_write_table(df,
                         caption=caption,
                         label="tab:predictive-equality-full" if full else "tab:predictive-equality",
                         save_as = 'table_predictive_equality.tex' if full else 'table_predictive_equality_partial.tex',
    )


gen_table_predictive_equality()
# Add the following thing to the 1st, 2nd, 4th and 5th pbox, because they overlap otherwise...
# \rule[-6pt]{0pt}{1mm}
# Also, manually remove the metric column because it's only STD anyway
gen_table_predictive_equality(full=False)


def gen_plot_scores():

    plt.style.use('src/violinPlot.mplstyle')

    subgroups = ['African', 'Asian', 'Caucasian', 'Indian']

    dataset = 'rfw'
    approaches = ['uncalibrated', 'baseline', 'fsn', 'ftc', 'oracle', 'faircal']
    feature = 'facenet-webface'
    _df = pd.read_csv( os.path.join(DATA_FOLDER, dataset, f'{dataset}.csv') )

    fig, axs = plt.subplots(1,len(approaches), squeeze=False, figsize=(20,5))
    axs = axs.flatten()
    for ax,approach in zip(axs, approaches):
        conf = (dataset, feature, approach)
        all_scores = metrics[conf]['score']

        calibrated_score = np.full(len(_df), np.nan)
        for fold in range(1,6):
            select = _df['fold'] == fold
            calibrated_score[select] = all_scores[fold-1,select]

        _df['calibrated_score'] = all_scores[-1,:]
        df = _df.dropna(subset=['calibrated_score'])

        sns.violinplot(
            x ='ethnicity',
            hue="pair",
            y='calibrated_score',
            split=True,
            data=df,
            scale="count",
            inner="quartile",
            order=subgroups,
            palette={"Genuine": "royalblue", "Imposter": "skyblue"},
            ax=ax,
        )

        fpr, tpr, thr = roc_curve(y_true=df['same'],
                                y_score=df['calibrated_score'],
                                drop_intermediate=False)
        ax.axhline(
            y=np.interp(.05, fpr, thr),
            ls='-', c='black', lw=2, alpha=1.)

        for j,attr in enumerate(subgroups):
            select = (df['ethnicity'] == attr)

            fpr, tpr, thr = roc_curve(
                y_true=df['same'][select],
                y_score=df['calibrated_score'][select],
                drop_intermediate=False)

            ax.hlines(
                y=np.interp(.05, fpr, thr),
                xmin=j-.5,
                xmax=j+.5,
                lw=3, ls='-', color='crimson')

        ax.set_title(approach)
        ax.legend(loc='lower right')

    plt.show()

gen_plot_scores()


def gen_plot_fpr():

    plt.style.use('src/fprPlot.mplstyle')

    dataset = 'rfw'
    # approaches = ['baseline', 'fsn', 'ftc', 'faircal', 'oracle']
    approaches = ['baseline', 'fsn', 'faircal']
    feature = 'facenet-webface'
    _df = pd.read_csv( os.path.join(DATA_FOLDER, dataset, f'{dataset}.csv') )

    if dataset == 'rfw':
        subgroups = ['African', 'Asian', 'Indian', 'Caucasian']
    else:
        subgroups = ['B', 'A', 'W', 'I']

    fig, axs = plt.subplots(1,len(approaches), squeeze=False, figsize=(20,5))
    axs = axs.flatten()
    for ax,approach in zip(axs, approaches):

        conf = (dataset, feature, approach)
        all_scores = metrics[conf]['score']

        # calibrated_score = np.full(len(_df), np.nan)
        # for fold in range(1,6):
        #     select = _df['fold'] == fold
        #     calibrated_score[select] = all_scores[fold-1,select]

        # _df['calibrated_score'] = all_scores[-1,:]
        # df = _df.dropna(subset=['calibrated_score'])
        _df['calibrated_score'] = all_scores[-1,:]
        df = _df[ _df['fold'] == 5 ].copy()


        fpr_glob, tpr_glob, thr_glob = roc_curve(
            y_true=df['same'].astype(float),
            y_score=df['calibrated_score'].astype(float),
            drop_intermediate=False)

        for j,attr in enumerate(subgroups):
            if dataset == 'rfw':
                select = (df['ethnicity'] == attr)
            else:
                select = (df['e1'] == attr) | (df['e2'] == attr)

            fpr_sub, tpr_sub, thr_sub = roc_curve(
                y_true=df['same'][select].astype(float),
                y_score=df['calibrated_score'][select].astype(float),
                drop_intermediate=False)

            _fpr_glob = np.interp(thr_sub, thr_glob[::-1], fpr_glob[::-1])
            ax.plot( _fpr_glob, fpr_sub, label=attr, lw=2 )

        ax.plot([0.05, 0.05],[0,1],'--k',linewidth=2)

        ax.legend(loc='upper left')
        ax.set_title(approach)

        ax.set_xlim(0.0, 0.1)
        ax.set_xticks(np.linspace(0,.1,6))
        ax.set_xlabel("Global FPR")

        ax.set_ylim(0.0, 0.18)
        ax.set_yticks(np.linspace(0,.15,4))

    fig.suptitle(f"Subgroup FPR for a global threshold at 5% FPR, using feature {feature} and dataset {dataset}")

    plt.show()

gen_plot_fpr()
