import pickle

embeddings = pickle.load(open('./data/rfw/facenet-webface_embeddings.pickle', 'rb'))

african_dict = {}
asian_dict = {}
caucasian_dict = {}
indian_dict = {}

['African', 'Asian', 'Caucasian', 'Indian']

for key, value in embeddings.items():
    print(key)
    print(value)
    break
    if 'African' in key:
        african_dict[key] = value
    elif 'Asian' in key:
        asian_dict[key] = value
    elif 'Caucasian' in key:
        caucasian_dict[key] = value
    elif 'Indian' in key:
        indian_dict[key] = value

with open('./data/rfw/African_facenet-webface_embeddings.pickle', 'wb') as f:
    pickle.dump(african_dict, f)

with open('./data/rfw/Asian_facenet-webface_embeddings.pickle', 'wb') as f:
    pickle.dump(asian_dict, f)

with open('./data/rfw/Caucasian_facenet-webface_embeddings.pickle', 'wb') as f:
    pickle.dump(caucasian_dict, f)

with open('./data/rfw/Indian_facenet-webface_embeddings.pickle', 'wb') as f:
    pickle.dump(indian_dict, f)