import argparse
import os
import pandas as pd
import pickle


def collect_csv_bfw(args: argparse.Namespace) -> pd.DataFrame:
	"""
	Parse BFW datatable file (from raw dataset) with embeddings to the format expected in FairCal code
	This including rename some columns, removing some and adding some.
	These added columns will be initialized empty, so they should still be filled in elsewhere
	Output is saved to args.csv_path_bfw_out
	"""
	# Columns expected in output:
	# fold,path1,path2,same,id1,id2,att1,att2,g1,g2,e1,e2,facenet-webface,facenet,arcface,pair
	expected_columns = {'fold','path1','path2','same','id1','id2','att1','att2','g1','g2','e1','e2','facenet-webface','facenet','arcface','pair'}

	print("\nBFW", "="*20)

	df = pd.read_csv(args.csv_path_bfw_in)
	# fold,p1,p2,label,id1,id2,att1,att2,vgg16,resnet50,senet50,a1,a2,g1,g2,e1,e2

	# Rename columns
	renames = {
		'p1': 'path1',
		'p2': 'path2',
		'label': 'same'
	}
	df.rename(mapper=renames, axis='columns', inplace=True)

	# Set 'pair' column as either 'Genuine' or 'Imposter'
	df['pair'] = df['same'].map(lambda x: 'Genuine' if x else 'Imposter')

	# Drop extra ones
	drop = [col for col in df.columns if col not in expected_columns]
	print("Dropping columns", drop)
	df.drop(labels=drop, axis='columns', inplace=True)

	# Add missing ones
	add = [col for col in expected_columns if col not in df.columns]
	print("Adding columns", add)
	for col in add:
		df[col] = None

	# Read embeddings to find what paths are available
	with open(args.embeddings_path_bfw, 'rb') as f:
		embeddings = pickle.load(f)

	# Convert to set
	embedded_paths = set( embeddings.keys() )
	print("Found {} embedded paths".format(len(embedded_paths)))

	# Filter dataframe where path1 and path2 are both embedded
	mask = df['path1'].isin(embedded_paths) & df['path2'].isin(embedded_paths)
	print("Will keep {}/{} pairs".format(sum(mask), len(mask)))
	df = df[mask]

	# Save output
	print("Saving to", args.csv_path_bfw_out)
	df.to_csv(path_or_buf=args.csv_path_bfw_out, index=False)


def collect_csv_rfw(args: argparse.Namespace) -> pd.DataFrame:
	"""
	Parse RFW filtered_pairs.csv with embeddings to the format expected in FairCal code
	This including rename some columns, removing some and adding some.
	These added columns will be initialized empty, so they should still be filled in elsewhere
	Output is saved to args.csv_path_rfw_out
	"""
	# Columns expected in output:
	# ethnicity,id1,num1,id2,num2,same,facenet,fold,facenet-webface,arcface,pair
	expected_columns = {'ethnicity','path1','path2','id1','num1','id2','num2','same','facenet','fold','facenet-webface','arcface','pair'}

	print("\nRFW", "="*20)
	df = pd.read_csv(args.csv_path_rfw_in)
	# id1,id2,path1,path2,label,fold,ethnicity,num1,num2

	# Set some columns
	df['same'] = df['label'].astype(bool)
	df['pair'] = df['same'].map(lambda x: 'Genuine' if x else 'Imposter')

	# Drop extra columns
	drop = [col for col in df.columns if col not in expected_columns]
	print("Dropping columns", drop)
	df.drop(labels=drop, axis='columns', inplace=True)

	# Add missing columns
	add = [col for col in expected_columns if col not in df.columns]
	print("Adding columns", add)
	for col in add:
		df[col] = None

	# Read embeddings to see what paths are embedded
	with open(args.embeddings_path_rfw, 'rb') as f:
		embeddings = pickle.load(f)
	embedded_paths = set( embeddings.keys() )
	print("Found {} embedded paths".format(len(embedded_paths)))

	# Filter dataframe where path1 and path2 are both embedded
	mask = df['path1'].isin(embedded_paths) & df['path2'].isin(embedded_paths)
	print("Will keep {}/{} pairs".format(sum(mask), len(mask)))
	df = df[mask]

	# Save output
	print("Saving to", args.csv_path_rfw_out)
	df.to_csv(path_or_buf=args.csv_path_rfw_out, index=False)



if __name__ == '__main__':
	parser = argparse.ArgumentParser(
	description="Parse raw datasets to .csv files for approach",
	)

	parser.add_argument("--csv_path_bfw_in",
		type=str,
		default='./../data/bfw/bfw-v0.1.5-datatable.csv',
		help="Path to the `bfw-v0.1.5-datatable.csv` file",
	)

	parser.add_argument("--csv_path_bfw_out",
		type=str,
		default='./../data/bfw/bfw.csv',
		help="Path to save `bfw.csv` to",
	)

	parser.add_argument("--embeddings_path_bfw",
		type=str,
		default='./../data/bfw/facenet_embeddings.pickle',
		help="Path to a `*_embeddings.picke` file containing a dictionary mapping embedding image paths",
	)

	parser.add_argument("--csv_path_rfw_in",
		type=str,
		default='./../data/rfw/txts/filtered_pairs.csv',
		help="Path to the `filtered_pairs.csv` file",
	)

	parser.add_argument("--csv_path_rfw_out",
		type=str,
		default='./../data/rfw/rfw.csv',
		help="Path to save `rfw.csv` to",
	)

	parser.add_argument("--embeddings_path_rfw",
		type=str,
		default='./../data/rfw/facenet_embeddings.pickle',
		help="Path to a `*_embeddings.picke` file containing a dictionary mapping embedding image paths",
	)

	args = parser.parse_args()

	for keyword in vars(args):
		path = getattr(args, keyword)
		path = os.path.abspath(path)
		setattr(args, keyword, path)
		if not os.path.isfile(path):
			raise ValueError("Parameter {} with path {} not found".format(keyword, path))

	collect_csv_bfw(args)
	collect_csv_rfw(args)
