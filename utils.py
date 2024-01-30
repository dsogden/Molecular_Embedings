import numpy as np
import pandas as pd

def get_smiles(dataframe, threshold):
    indices = []
    for idx, smile in enumerate(df['smiles']):
        if len(smile) <= threshold:
            indices.append(idx)
    smiles = df.iloc[indices]['smiles'].values
    return smiles

def encode_smiles(smiles, max_length, encoder, start_token, end_token):
    tokens = set()
    for smile in smiles:
        tokens = tokens.union(set(c for c in smile))
    tokens = sorted(list(tokens))
    smiles_encoder = {start_token: 0, end_token: 1}
    smiles_decoder = {0: start_token, 1: end_token}

    for idx, smile in enumerate(tokens):
        smiles_encoder[smile] = idx + 2
        smiles_decoder[idx + 2] = smile
    
    N = smiles.shape[0]
    features = np.zeros((N, max_length + 1))
    for idx, smile in enumerate(smiles):
        length = len(smile)
        for jdx, letter in enumerate(smile):
            features[idx, jdx] = encoder[letter]
        features[idx, length] = encoder[end_token]
    return features
