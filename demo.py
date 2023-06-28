import torch

from tqdm import tqdm
from train import Trainer
from test import Tester

from transformer import Transformer

def main():
    device = torch.device("cpu") #= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1,5,6,4,3,9,2,1], [2,6,7,5,4,1,3,1]]).to(device)
    y = torch.tensor([[2,6,7,5,4,1,3,1], [1,5,6,4,3,9,2,1]]).to(device)

    src_vocab = 10
    trg_vocab = 10
    src_pad = 0
    trg_pad = 0

    # model parameters
    model = Transformer(
            src_vocab=src_vocab,
            trg_vocab=trg_vocab,
            src_pad=src_pad,
            trg_pad=trg_pad).to(device)

    # training parameters

    # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=lr)


    out = model(x, y)
    print(out[0])
    
if __name__ == "__main__":
    main()
