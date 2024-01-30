import numpy as np
import pandas as pd
import torch

def get_smiles(dataframe, threshold):
    indices = []
    for idx, smile in enumerate(df['smiles']):
        if len(smile) <= threshold:
            indices.append(idx)
    smiles = df.iloc[indices]['smiles'].values
    return smiles

def encode_smiles(smiles, max_length, start_token, end_token):
    tokens = set()
    for smile in smiles:
        tokens = tokens.union(set(c for c in smile))
    tokens = sorted(list(tokens))
    smiles_encoder = {start_token: 0, end_token: 1}

    for idx, smile in enumerate(tokens):
        smiles_encoder[smile] = idx + 2
    
    N = smiles.shape[0]
    features = np.zeros((N, max_length + 1))
    for idx, smile in enumerate(smiles):
        length = len(smile)
        for jdx, letter in enumerate(smile):
            features[idx, jdx] = smiles_encoder[letter]
        features[idx, length] = smiles_encoder[end_token]
    return features, len(smiles_encoder)

def train_step(train_data, labels, indices, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    total_loss = 0
    N = train_data.shape[0]
    for idx in indices:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden, encoder_cell = encoder(train_data[idx])
        decoder_outputs = decoder((encoder_hidden, encoder_cell), labels[idx])

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            labels[idx].view(-1).long()
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    return total_loss / N

def decode_smiles(inputs, encoder, decoder):
    batch_size = inputs.shape[0]
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(inputs)
        decoder_outputs = decoder(encoder_hidden, None)
    
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
    decoded_ids = decoded_ids.numpy()

    decoded_smiles = []
    for i in range(batch_size):
        current_smile = ''
        for idx in decoded_ids[i]:
            if smiles_decoder[idx] == end_token:
                break
            else:
                current_smile += smiles_decoder[idx]
        decoded_smiles.append(current_smile)
    return decoded_smiles
