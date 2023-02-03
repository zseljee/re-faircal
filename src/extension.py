from typing import Optional
from dataset import Dataset
from constants import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io

DATA_ROOT = os.path.abspath( './data/rfw/data' )

def main(dataset:str = 'bfw', feature:str = 'facenet'):


    print('loading dataset....')
    ds = Dataset(name = dataset, feature = feature)
    print('dataset loaded successfully')

    df = ds.df
    ds.set_fold(1)
    kmeans = ds.train_cluster(save=True)
    embeddings, idx2path = ds.get_embeddings(train=None, return_mapper=True)

    cluster_assignments = kmeans.predict(embeddings)
    if dataset == 'rfw':
        df = pd.DataFrame.from_dict({'ethnicity':[path.split('/')[0] for path in idx2path], 'cluster assignment':cluster_assignments})
        COLORS = {'Indian': '#474747',
                'Asian': '#616161',
                'Caucasian': '#808080',
                'African': '#A6A6A6'}
    else:
        df = pd.DataFrame.from_dict({'ethnicity':[path.split('/')[0].split('_')[0] for path in idx2path], 'cluster assignment':cluster_assignments})
        COLORS = {'indian': '#474747',
                'asian': '#616161',
                'white': '#808080',
                'black': '#A6A6A6'}

    # make bar plot of ethnicities per cluster
    ax = plt.gca()

    df_ = df.value_counts(sort=True).to_frame('counts').reset_index()
    df_ = df_.pivot( index='ethnicity', columns='cluster assignment').fillna(value=0).astype(int)

    df_ = df_ / df_.sum(axis=0)

    df_ = df_.T
    df_['max_score'] = df_.max(axis=1)
    df_ = df_.sort_values(['max_score'])
    df_['max_score'].plot(ax=ax, c='r', label='max percentage', **{'linewidth':2.5})
    df_ = df_.drop(columns='max_score')
    df_.plot.bar(xticks = range(100) , stacked=True, ax=ax, color=COLORS, width=1.0)
    # df_.plot.bar(xticks = range(100) , stacked=True, ax=ax, width=1.0)
    labels = [df_.index[i][1] for i in range(len(df_.index))]
    ax.set_xticklabels(labels)
    ax.set_ylabel('percentage of ethnicities in cluster', {'fontsize': 'xx-large'})
    ax.legend(loc='upper center', bbox_to_anchor = (0.5, 1.01), ncol=6, fontsize='large')
    ax.set_xlabel('cluster assignment', {'fontsize': 'xx-large'})
    # plt.tight_layout()
    plt.show()

    # inspect_cluster(cluster_assignments, idx2path)

    return df_, cluster_assignments, idx2path

def inspect_cluster(cluster_assignment, idx2path, cluster_nr: Optional[int] = None, interactive=True):
    N_CLUSTERS = 100
    clusters: list[set[str]] = [set() for _ in range(N_CLUSTERS)]
    for i_emb, i_cluster in enumerate(cluster_assignment):
        clusters[i_cluster].add( idx2path[i_emb] )

    while interactive or cluster_nr is not None:
        if cluster_nr is None:
            cluster_nr = input('What cluster do you want to inspect?\n>')

            if cluster_nr == 'q':
                print('shutting down..')
                return 0
            elif not cluster_nr.isnumeric():
                print('not a valid cluster number')
                continue
            else:
                cluster_nr = int(cluster_nr)

        paths = clusters[cluster_nr]
        cluster_size = len(paths)
        GRID_SIZE = int(np.ceil(np.sqrt(cluster_size)))
        GRID_SIZE = 10
        fig, axs = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(30,30), gridspec_kw=dict(wspace=0.05, hspace=0.05, top=1. - 0.5 / (GRID_SIZE + 1), bottom=0.5 / (GRID_SIZE + 1),
                     left=0.5 / (GRID_SIZE + 1), right=1 - 0.5 / (GRID_SIZE + 1)),)
        axs = axs.flatten()
        for path, ax in zip(paths, axs):
            ax.imshow(io.imread(os.path.join(DATA_ROOT, path)))
            ax.set_axis_off()
        # plt.tight_layout()
        plt.savefig(f"cluster_{cluster_nr}.png")
        plt.close(fig)

        cluster_nr = None


if __name__ == '__main__':
    df_ = main()
