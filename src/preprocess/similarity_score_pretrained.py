"""
Small file to check out embedding quality
It plots histograms of scores for each combination of feature and dataset
Assumes, for each dataset, there exists a file at `./data/[dataset]/[dataset].csv`
And each dataset contains, for each feature, a column named the same as the feature.

Usage:
$ python ./src/preprocess/similarity_score_pretrained.py
This is such that the `./data/` folder is acording to the rest of the project
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FOLDER = os.path.abspath( './data/' )

# What datasets and embeddings to combine
dataset_names = ['bfw', 'rfw' ]
embedding_names = ['facenet', 'facenet-webface', 'arcface']
print("Comparing similarity scores")
print("Datasets:", dataset_names)
print("Embeddings:", embedding_names)

# Set up plto
fig, axs = plt.subplots(len(dataset_names), len(embedding_names), figsize=(20,20), squeeze=False, sharex='all', sharey='row')
fig.suptitle("Similarities")

# Iterate datasets
for dataset_name, axrow in zip(dataset_names, axs):
	dataset_path = os.path.join(DATA_FOLDER, dataset_name, '{}.csv'.format(dataset_name))
	print("Dataset {}; reading data from {}".format(dataset_name, dataset_path))

	if not os.path.isfile(dataset_path):
		print(f"Could not find dataset {dataset_name}, please save results at {dataset_path}")
		continue

	# Read as pd.DataFrame
	df = pd.read_csv(dataset_path)

	# Iterate embeddings
	for embedding_name, ax in zip(embedding_names, axrow):
		print("Embedding {}".format(embedding_name))

		ax.set_title("Dataset {}, embeddings {}".format(dataset_name, embedding_name))
		ax.set_xlim(-1,1)

		if embedding_name not in df:
			print(f"Could not find feature {embedding_name} in {dataset_name}, add a column named {embedding_name} in {dataset_path}")
			continue

		# Get similarities
		sims = df[embedding_name]

		# Plot histogram
		ax.hist( sims[df['same'] == False], color="red", label="Imposter", bins=100 )
		ax.hist( sims[df['same'] == True], color="green", label="Genuine", bins=100 )


plt.savefig('results.png')
plt.show()
