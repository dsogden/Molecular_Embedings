from rdkit import Chem
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from utils import get_smiles, encode_smiles, train_step

path = 'tox21.csv'
df = pd.read_csv(path)
smiles = get_smiles(df, 25)

start_token = '<sos>'
end_token = '<eos>'

features, encoder_length = encode_smiles(smiles, 25, start_token, end_token)

encoder = Encoder(encoder_length, 512, 128, 2)
decoder = Decoder(encoder_length, 512, 128, encoder_length, 2)

inputs, targets = torch.tensor(features, dtype=torch.int32), torch.tensor(features, dtype=torch.int32)
N = inputs.shape[0] // 32
index = N * 32
inputs = inputs[:index].view(N, 32, 26)
targets = targets[:index].view(N, 32, 26)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

def main():
    indices = torch.arange(N)
    permutation = torch.randperm(indices.shape[0])
    indices = indices[permutation]
    epochs = 100
    
    criterion = nn.NLLLoss()
    
    encoder.train()
    decoder.train()
    training_loss = []
    for epoch in tqdm(range(epochs)):
        loss = train_step(
            inputs, targets, indices, encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion
        )
    
        print(f'Epoch: {epoch + 1}, training loss: {loss:.5f}')
        training_loss.append([epoch + 1, loss])
        permutation = torch.randperm(len(indices))
        indices = indices[permutation]
        
     molecules = decode_smiles(inputs[-1], encoder, decoder)

if __name__ == "__main__":
    main()
