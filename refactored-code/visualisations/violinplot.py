import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics
import seaborn as sns

from argparse import Namespace
from approaches import uncalibrated, baseline, faircal
from approaches.utils import get_threshold
from dataset import Dataset

def violinplot(**approaches):

    fig, axs = plt.subplots(1,len(approaches), squeeze=False, figsize=(20,10), sharey=True)
    axs = axs.flatten()

    for approach, ax in zip(approaches, axs):
        data = approaches[approach]

        data = data['fold5']

        subgroups = [ethnicity for ethnicity in data['threshold'] if ethnicity != 'global']

        df = data['df'].copy()
        df_test = df[ df['test']==True ]

        sns.violinplot(
            x ='ethnicity',
            hue="pair",
            y='calibrated_score',
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
        ax.legend(loc='lower right')

    plt.show()