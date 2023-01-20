import numpy as np
import pandas as pd
import os
import pickle

import matplotlib.pyplot as plt

DATA_FOLDER = os.path.abspath( './../data/' )

# What datasets and embeddings to combine
dataset_names = { 'bfw', 'rfw' }
embedding_names = { 'facenet', 'facenet-webface' }#, 'arcface'}
print("Computing similarity scores")
print("Datasets:", dataset_names)
print("Embeddings:", embedding_names)

# Set up plto
fig, axs = plt.subplots(len(dataset_names), len(embedding_names), figsize=(20,20), squeeze=False)
fig.suptitle("Similarities")

# Iterate datasets
for dataset_name, axrow in zip(dataset_names, axs):
	dataset_path = os.path.join(DATA_FOLDER, dataset_name, '{}.csv'.format(dataset_name))
	print("Dataset {}; reading data from {}".format(dataset_name, dataset_path))
	
	# Read as pd.DataFrame
	df = pd.read_csv(dataset_path)

	# Iterate embeddings
	for embedding_name, ax in zip(embedding_names, axrow):
		embedding_path = os.path.join(DATA_FOLDER, dataset_name, '{}_embeddings.pickle'.format(embedding_name))
		print("Embeddings from model {}, reading from {}".format(embedding_name, embedding_path))
		with open(embedding_path, 'rb') as f:
			embs: dict[str, np.ndarray] = pickle.load(f)
			
			# Compute similarities
			sims: pd.Series = df[ ['path1', 'path2'] ].apply( lambda row: np.dot(embs[row['path1']][0], embs[row['path2']][0]), axis=1 )
			df[embedding_name] = sims

			# Plot histogram
			ax.hist( sims[df['same'] == False], color="red", label="Imposter", bins=100 )
			ax.hist( sims[df['same'] == True], color="green", label="Genuine", bins=100 )
			ax.set_xlim(-1,1)
			ax.set_title("Dataset {}, embeddings {}".format(dataset_name, embedding_name))

	# Update csv file
	df.to_csv(dataset_path, index=False)
plt.savefig('results.png')
plt.show()

