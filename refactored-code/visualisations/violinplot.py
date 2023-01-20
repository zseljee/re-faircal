import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns

from argparse import Namespace
from approaches import uncalibrated, baseline, faircal
from approaches.utils import get_threshold
from dataset import Dataset

def violinplot(dataset, conf):

    fig, axs = plt.subplots(1,3, figsize=(40,20))

    dataset.set_fold(5)
    
    approaches = {
        'Baseline': uncalibrated,
        'Baseline + Calibrated': baseline,
        'FairCal': faircal,
    }
    for approach, ax in zip(approaches, axs):
        data = approaches[approach](dataset, conf)

        # Reset dataset selection
        dataset.select(None)

        subgroups = [ethnicity for ethnicity in data['threshold'] if ethnicity != 'global']

        df = dataset.df
        df_test = df[ df['fold'] == dataset.fold ].copy()
        
        df_test['temp'] = data['confidences']['test']

        sns.violinplot(
            x ='ethnicity',
            hue="pair",
            y='temp',
            split=True,
            data=df_test,
            scale="count",
            inner="quartile",
            order=subgroups,
            palette={"Genuine": "royalblue", "Imposter": "skyblue"},
            ax=ax,
        )

        ax.set_title(approach)
        # ax.set_ylabel("Cosine Similarity score")

        ax.axhline(y=data['threshold']['global'], ls='-', c='black', lw=2, alpha=1.)
        for j,ethnicity in enumerate(subgroups):
            ax.hlines(y=data['threshold'][ethnicity],
                    xmin=j-.5,
                    xmax=j+.5,
                    lw=3, ls='-', color='crimson')

        ax.set_title(approach)
        ax.set_ylabel("Probability")

    plt.show()