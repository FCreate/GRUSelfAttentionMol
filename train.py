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

def make_parser():
    parser = argparse.ArgumentParser(description='PyTorch RNN Classifier w/ attention')

    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings [Uses pretrained on 50, 100, 200, 300]')
    parser.add_argument('--hidden', type=int, default=500,
                        help='number of hidden units for the RNN encoder')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers of the RNN encoder')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--drop', type=float, default=0,
                        help='dropout')
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
    parser.add_argument('--tensorboard', type=str, help = "tensorboard dir")
    parser.add_argument('--ckpt_name', type=str, help="PyTorch checkpoint name")
    parser.add_argument('--print_every', type=int, default=20,
                        help='hidden value for self-attention aka d_a')
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


def seed_everything(seed, cuda=False):
    # Set the random seed manually for reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

def train(model, data, optimizer, criterion, args, device, writer, epoch):
    """
    Train GRUSelfAttention Model.
    :param model: Model, which we want to evaluate.
    :param data: PyTorch dataloader class.
    :param optimizer: PyTorch optimizer class.
    :param criterion: Which metric will we evaluate.
    :param args: Args class from init
    :param device: PyTorch Device.
    :param writer: Tensorboard Writer.
    :param epoch: Number of epoch to successfully write it to tensorboard writer.
    :return:
    """
    model.train()
    total_loss = []
    for batch_num, (x, y, mask) in enumerate(data):
        model.zero_grad()
        x, y, mask = x.to(device, ), y.to(device), mask.to(device)
        output = model(x)
        loss = criterion(output, y)
        total_loss.append(loss.item())
        loss.backward()
        optimizer.step()
        if ((batch_num%args.print_every)==0):
            writer.add_scalar('train_loss', (sum(total_loss) / len(total_loss)), batch_num+(epoch*len(data)))
    return (sum(total_loss) / len(total_loss))


def evaluate(model, data, optimizer, criterion, args, device,writer, epoch):
    """
    Evaluate GRUSelfAttention Model.
    :param model: Model, which we want to evaluate
    :param data: PyTorch dataloader class.
    :param criterion:Which metric will we evaluate.
    :param args:Args class from init
    :param device: PyTorch Device.
    :param writer: Tensorboard Writer.
    :param epoch: Number of epoch to successfully write it to tensorboard writer.
    :return:
    """
    model.eval()

    total_loss = []
    with torch.no_grad():
        for  batch_num, (x, y, mask) in enumerate(data):
            x, y, mask = x.to(device, ), y.to(device), mask.to(device)
            output = model(x)
            total_loss.append(criterion(mask * output, mask * y))
            if ((batch_num % args.print_every) == 0):
                writer.add_scalar('test_loss', (sum(total_loss) / len(total_loss)), batch_num + (epoch * len(data)))
                print((sum(total_loss) / len(total_loss)))
    return (sum(total_loss) / len(total_loss))


def main():
    args = make_parser().parse_args()
    writer = SummaryWriter(log_dir=args.tensorboard)
    print("[Model hyperparams]: {}".format(str(args)))
    x, y, char2index, char2count, index2char  = dataset2array()
    output_scaler = OurRobustToNanScaler()
    y = np.float32(y)
    y = output_scaler.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y)
    train_dataset = ToxicDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    test_dataset = ToxicDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cpu") if not cuda else torch.device("cuda:0")

    seed_everything(seed=args.seed, cuda=cuda)

    n_endpoints = y_train.shape[1]
    args.nlabels = n_endpoints # hack to not clutter function arguments

    ntokens = len(char2index)
    embedding = nn.Embedding(ntokens, args.emsize)

    encoder = Encoder(args.emsize, args.hidden, nlayers=args.nlayers,
                      dropout=args.drop, bidirectional=args.bi)

    attention_dim = args.hidden if not args.bi else 2*args.hidden
    attention = BahdanauSA(attention_dim, args.hid_sa_val, args.r, device)
    print("n_endpoints ", n_endpoints)
    model = Classifier(embedding, encoder, attention, n_endpoints)
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, amsgrad=True)

    best_valid_loss = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, args, device, writer, epoch)
        writer.add_scalar('epoch_train_loss', train_loss, epoch)
        test_loss = evaluate(model,test_loader, optimizer, criterion, args, device, writer, epoch)
        writer.add_scalar('epoch_test_loss', test_loss, epoch)

        if not best_valid_loss or test_loss < best_valid_loss:
            best_valid_loss = test_loss
            torch.save(model.state_dict(), args.ckpt_name)



if __name__ == '__main__':
    main()