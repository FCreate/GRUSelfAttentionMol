from io import open
import pickle
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import re
from rdkit import Chem
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def _smiles_atom_tokenizer (smi):
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens


def canonize_smile (sm):
    m = Chem.MolFromSmiles(sm)
    try: return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except: return None

def canonize_mixture (mix):
    return '.'.join([canonize_smile(sm) for sm in mix.split('.')])

def dataset2array(name_of_file = "data/df_tox_85165.csv"):
    df = pd.read_csv(name_of_file)

    SOS_token = 0
    EOS_token = 1
    char2index = {"SOS":0, "EOS":1, "PAD":2, "UNK":3}
    char2count = {"SOS":1, "EOS":1, "PAD":1, "UNK":1}
    index2char = {0:"SOS", 1:"EOS", 2:"PAD", 3:"UNK"}
    n_words = 4

    smiles = list(df["SMILES"])
    del df["SMILES"]
    y = df.values

    for smile in smiles:
        chars = _smiles_atom_tokenizer(smile)
        for char in chars:
            if char in char2count:
                char2count[char] += 1
            else:
                index2char[n_words] = char
                char2count[char] = 1
                char2index[char] = n_words
                n_words += 1

    part_rare_words = 0.0001
    all_words = sum(char2count.values())

    SOS_token = 0
    EOS_token = 1
    all_char2index = {"SOS":0, "EOS":1, "PAD":2, "UNK":3}
    all_char2count = {"SOS":1, "EOS":1, "PAD":1, "UNK":1}
    all_index2char = {0:"SOS", 1:"EOS", 2:"PAD", 3:"UNK"}
    n_words = 4

    for key in list(char2index.keys())[4:]:
        if (((char2count[key]) / all_words) > part_rare_words):
            all_char2index[key] = n_words
            all_char2count[key] = char2count[key]
            all_index2char[n_words] = key
            n_words += 1
        else:
            all_char2count["UNK"] = all_char2count["UNK"] + char2count[key]

    char2count = all_char2count
    char2index = all_char2index
    index2char = all_index2char
    with open("pickle_dicts.pk", 'wb') as pickle_file:
        pickle.dump({"char2count":char2count, "char2index":char2index, "index2char":index2char},pickle_file)

    ###Encode smiles
    encoded_smiles = []
    for smile in smiles:
        chars = _smiles_atom_tokenizer(smile)
        encoded_smiles.append([0])
        for char in chars:
            if char in char2index.keys():
                append_key = char
            else:
                append_key = "UNK"
            encoded_smiles[len(encoded_smiles) - 1].append(char2index[append_key])
        encoded_smiles[len(encoded_smiles) - 1].append(1)


    max_len = 100
    lens = [len(encoded_smiles[i]) for i in range(len(encoded_smiles))]
    exclude = [i for i in range(len(lens)) if (lens[i]>max_len)]
    for i in exclude[::-1]:
        del encoded_smiles[i]
        del lens[i]
    for i in range(len(encoded_smiles)):
        while (len(encoded_smiles[i]))<max_len:
            encoded_smiles[i].append(char2index["PAD"])
    data = np.array(encoded_smiles)
    return data, y, char2index, char2count, index2char