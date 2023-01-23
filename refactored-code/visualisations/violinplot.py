import matplotlib.pyplot as plt
import seaborn as sns

import os

plt.style.use(os.path.abspath('./refactored-code/visualisations/violinPlot.mplstyle'))

from approaches.utils import thr_at_fpr

def violinplot(dataset, results):
    """
    WIP, this code is way too specific to be useful outside of `main.py`
    TODO make it more general, perhaps reading from the `results.pickle` files?
    """
    FOLD=5

    approach_rename = {'uncalibrated': 'Baseline',
                       'baseline': 'Baseline + Calibration',
                       'faircal': 'FairCal',
                       'oracle': 'Oracle'
                       }
    feature_rename = {'facenet': 'FaceNet (VGGFace2)',
                      'facenet-webface': 'FaceNet (WebFace)',
                      'arcface': 'ArcFace'}
    dataset_rename = {'rfw': 'RFW', 'bfw': 'BFW'}

    results = [(conf, data) for (conf, data) in results if conf.approach in approach_rename]

    fig, axs = plt.subplots(1,len(results), squeeze=False, sharey=True)
    axs = axs.flatten()

    for (conf, data), ax in zip(results, axs):

        data = data[f'fold{FOLD}']

        data['threshold'] = dict()
        for subgroup in data['metrics']:
            metrics = data['metrics'][subgroup]
            data['threshold'][subgroup] = thr_at_fpr(thr=metrics['thr'], fpr=metrics['fpr'], target_fpr=0.05)

        subgroups = [ethnicity for ethnicity in data['metrics'] if ethnicity != 'Global']

        df = dataset.df.copy()
        df['calibrated_score'] = data['scores']

        df = df[ df['fold'] == FOLD ]

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

        ax.axhline(y=data['threshold']['Global'], ls='-', c='black', lw=2, alpha=1.)

        for j,ethnicity in enumerate(subgroups):
            ax.hlines(y=data['threshold'][ethnicity],
                    xmin=j-.5,
                    xmax=j+.5,
                    lw=3, ls='-', color='crimson')

        ax.set_title(approach_rename[conf.approach])
        ax.legend(loc='lower right')

        ax.set_ylabel("Probability" if conf.approach != "uncalibrated" else "Cosine Similarity")
        ticks = ax.get_yticks().tolist()
        ax.set_yticks(ticks, ['']*len(ticks))

        ax.set_xlabel("Ethnicity")


    plt.subplots_adjust(top=0.80,
                        bottom=0.15,
                        left=0.07,
                        right=0.98,
                        hspace=0.15,
                        wspace=0.15)

    fig.suptitle("Bias reduction for dataset {dataset:} using feature {feature:}"\
        .format(dataset=dataset_rename[conf.dataset], feature=feature_rename[conf.feature]))

    fname = os.path.abspath('./violinplot.png')
    plt.savefig(fname)
    print("Figure saved to", fname)
    plt.show()
