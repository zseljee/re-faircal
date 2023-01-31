import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

DATA_FOLDER = os.path.abspath( './data/' )

# What datasets and embeddings to combine
dataset_names = ['bfw', 'rfw' ]
# dataset_names = ['bfw' ]
embedding_names = ['facenet', 'facenet-webface', 'arcface']
# embedding_names = ['facenet', 'arcface']
print("Comparing similarity scores")
print("Datasets:", dataset_names)
print("Embeddings:", embedding_names)

# Set up plto
fig, axs = plt.subplots(len(dataset_names), len(embedding_names), figsize=(20,20), squeeze=False, sharex=True, sharey='row')
fig.suptitle("Similarities")

# Iterate datasets
for dataset_name, axrow in zip(dataset_names, axs):
	dataset_path = os.path.join(DATA_FOLDER, dataset_name, '{}.csv'.format(dataset_name))
	print("Dataset {}; reading data from {}".format(dataset_name, dataset_path))

	# Read as pd.DataFrame
	df = pd.read_csv(dataset_path)

	# Iterate embeddings
	for embedding_name, ax in zip(embedding_names, axrow):
		print("Embedding {}".format(embedding_name))

		# Get similarities
		sims = df[embedding_name]

		# Plot histogram
		ax.hist( sims[df['same'] == False], color="red", label="Imposter", bins=100 )
		ax.hist( sims[df['same'] == True], color="green", label="Genuine", bins=100 )
		# ax.set_xlim(-1,1)
		ax.set_title("Dataset {}, embeddings {}".format(dataset_name, embedding_name))


plt.savefig('results.png')
plt.show()
