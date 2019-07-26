import argparse
from model import *
from mol_dataset import dataset2array
import numpy.ma as ma
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

def make_parser():
    """
    It is basic parser with some command-line helpers.
    """
    parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings ')
    parser.add_argument('--drop', type=float, default=0,
                        help='dropout')
    parser.add_argument('--hidden', type=int, default=500,
                        help='number of hidden units for the RNN encoder')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers of the RNN encoder')
    parser.add_argument('--bi', action='store_true',
                        help='[USE] bidirectional encoder')
    parser.add_argument('--cuda', action='store_true',
                        help='[DONT] use CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--r', type=int, default=10,
                        help='number of undependable heads')
    parser.add_argument('--hid_sa_val', type=int, default=100,
                        help='hidden value for self-attention aka d_a')
    parser.add_argument('--ckpt_name', type=str, help="PyTorch checkpoint name")

    parser.add_argument('--out_file', type=str, help="Name of output file")

    return parser


class OurRobustToNanScaler():
    """
    This class is equal to StandardScaler from sklearn but can work with NaN's (ignoring it) but
    sklearn's scaler can't do it.
    """
    def fit(self, data):
        masked = ma.masked_invalid(data)
        self.means = np.mean(masked, axis=0)
        self.stds = np.std(masked, axis=0)

    def fit_transform(self, data):
        self.fit(data)
        masked = ma.masked_invalid(data)
        masked -= self.means
        masked /= self.stds
        return ma.getdata(masked)

    def inverse_transform(self, data):
        masked = ma.masked_invalid(data)
        masked *= self.stds
        masked += self.means
        return ma.getdata(masked)

class ToxicDataset(Dataset):
    """
    Toxic Dataset class with three fields - x, y and mask. All NaN targer variables are swapped with np.nan_to_num.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mask = ~ma.masked_invalid(self.y).mask
        self.y = np.nan_to_num(self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.x[idx]), torch.from_numpy(np.float32(self.y[idx])),
                torch.from_numpy(np.float32(self.mask[idx])))


def mse(y_true, y_pred):
    """
    MSELoss implementation.
    :param y_true: True values of y.
    :param y_pred: Prediction values of y.
    :return:
    """
    return np.mean((y_true- y_pred)**2)

def seed_everything(seed, cuda=False):
    """
    Set the random seed manually for reproducibility.
        :param seed: seed for all initializers
        :param cuda: if cuda == True, torch.cuda will be manually seeded.
    :return: No return
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
      torch.cuda.manual_seed_all(seed)


def main():
    args = make_parser().parse_args()
    print("[Model hyperparams]: {}".format(str(args)))
    #Get name of endpoints
    df = pd.read_csv("data/df_tox_85165.csv")
    endpoints = list(df.columns[1:])
    #Get x, y, and other dictionaries.
    x, y, char2index, char2count, index2char  = dataset2array()
    number_of_words = x.shape[1]
    output_scaler = OurRobustToNanScaler()
    y = np.float32(y)

    #Transform and split it.
    #Maybe transformer should be saved?
    y = output_scaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y)

    #Obtain test loader.
    test_dataset = ToxicDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")

    seed_everything(seed=args.seed, cuda=cuda)

    n_endpoints = y_train.shape[1]
    args.nlabels = n_endpoints # hack to not clutter function arguments

    #Build our model with params from command-line console.
    ntokens = len(char2index)
    embedding = nn.Embedding(ntokens, args.emsize)

    encoder = Encoder(args.emsize, args.hidden, nlayers=args.nlayers,
                      dropout=args.drop, bidirectional=args.bi)

    attention_dim = args.hidden if not args.bi else 2*args.hidden
    attention = BahdanauSA(attention_dim, args.hid_sa_val, args.r, device)
    print("n_endpoints ", n_endpoints)
    model = Model(embedding, encoder, attention, number_of_words, n_endpoints)
    #Load model hear.
    model.load_state_dict(torch.load(args.ckpt_name))
    model.to(device)
    #Evaluate and score calculation.
    y_s = []
    masks = []
    outputs = []
    for batch_idx, (x, y, mask) in enumerate(test_loader):
      x, y, mask = x.to(device), y.to(device), mask.to(device)
      y_s.append(y.detach().cpu().numpy())
      masks.append(mask.detach().cpu().numpy())
      output = model(x)
      outputs.append(output.detach().cpu().numpy())
    y_s = np.vstack(y_s)
    masks = np.vstack(masks)
    outputs = np.vstack(outputs)

    #Inverse and transform of outputs and ys.
    outputs = output_scaler.inverse_transform(outputs)
    y_s = output_scaler.inverse_transform(y_s)

    #Calculate MSE
    mse_for_diff_endpoints = []
    for i in range(y_s.shape[1]):
      mse_for_diff_endpoints.append(mse((masks * y_s)[:, i][np.array(masks[:, i], dtype=bool)],(masks * outputs)[:, i][np.array(masks[:, i], dtype=bool)]))

    #Calculate r2 score.
    r2score_for_diff_endpoints = []
    for i in range(y_s.shape[1]):
      r2score_for_diff_endpoints.append(r2_score((masks * y_s)[:, i][np.array(masks[:, i], dtype=bool)],(masks * outputs)[:, i][np.array(masks[:, i], dtype=bool)]))
    #Save it to file.
    with open(args.out_file, 'w') as f:
      f.write("Endpoint MSE     r2      \n")
      for idx, endpoint in enumerate(endpoints):
        f.write(endpoint+" "+str(mse_for_diff_endpoints[idx])+" "+str(r2score_for_diff_endpoints[idx])+"\n")

if __name__ == '__main__':
    main()