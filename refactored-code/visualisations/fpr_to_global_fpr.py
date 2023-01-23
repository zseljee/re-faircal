import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import os
import numpy as np

plt.style.use(os.path.abspath('./refactored-code/visualisations/fprPlot.mplstyle'))

from approaches.utils import thr_at_fpr

def fpr2globalfpr(dataset, results):
    """
    WIP, this code is way too specific to be useful outside of `main.py`
    TODO make it more general, perhaps reading from the `results.pickle` files?
    """
    FOLD=5

    approach_rename = {'baseline': 'Baseline',
                       'faircal': 'FairCal'
                       }
    feature_rename = {'facenet': 'FaceNet (VGGFace2)',
                      'facenet-webface': 'FaceNet (WebFace)',
                      'arcface': 'ArcFace'}
    dataset_rename = {'rfw': 'RFW', 'bfw': 'BFW'}

    results = [(conf, data) for (conf, data) in results if conf.approach in approach_rename]

    fig, axs = plt.subplots(1,len(results), squeeze=False, figsize=(15,8), sharey=True)
    axs = axs.flatten()

    axs[0].set_ylabel("False Positive Rate")

    plt.subplots_adjust(top=0.88,
                        bottom=0.15,
                        left=0.07,
                        right=0.98,
                        hspace=0.15,
                        wspace=0.15)

    for (conf, data), ax in zip(results, axs):
        ax: matplotlib.axes

        data = data[f'fold{FOLD}']

        subgroups = [ethnicity for ethnicity in data['metrics'] if ethnicity != 'Global']
        subgroups = ['African', 'Asian', 'Indian', 'Caucasian']

        df = dataset.df.copy()
        df['calibrated_score'] = data['scores']

        df = df[ df['fold'] == FOLD ]

        # Thr has to be in INCREASING order, so reverse it
        global_thr = data['metrics']['Global']['thr'][::-1]
        global_fpr = data['metrics']['Global']['fpr'][::-1]

        for subgroup in subgroups:

            thr = data['metrics'][subgroup]['thr']
            fpr = data['metrics'][subgroup]['fpr']

            _x = np.interp(thr, global_thr, global_fpr)

            ax.plot(_x, fpr, label=subgroup, lw=2)

        ax.plot([0.05, 0.05],[0,1],'--k',linewidth=2)

        ax.legend(loc='upper left')
        ax.set_title(approach_rename[conf.approach])

        ax.set_xlim(0.0, 0.1)
        ax.set_xticks(np.linspace(0,.1,6))
        ax.set_xlabel("Global FPR")

        ax.set_ylim(0.0, 0.18)
        ax.set_yticks(np.linspace(0,.15,4))

    fig.suptitle("FPR to global FPR for dataset {dataset:} using feature {feature:}"\
        .format(dataset=dataset_rename[conf.dataset], feature=feature_rename[conf.feature]))

    fig.tight_layout()
    fname = os.path.abspath('./FPR_to_global_FPR.png')
    plt.savefig(fname)
    print("Figure saved to", fname)
    plt.show()
