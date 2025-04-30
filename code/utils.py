import numpy as np
import json

def load_data(folder, dataset):
    X = np.load(f"{folder}/{dataset}/X.npy", allow_pickle=True).astype(np.float64)
    y = np.load(f"{folder}/{dataset}/y.npy", allow_pickle=True).astype(np.int64)
    return X, y

def store_projection(embedding, labels, filename):
    embedding_list =  [{"x": float(y_), "y": float(x_)} for x_, y_ in embedding]
    label = list(labels)

    data = {'embedding' : embedding_list,
            'labels' : label}
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)