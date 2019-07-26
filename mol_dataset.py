import numpy as np
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

def augment_smile (sm):
    mol = Chem.MolFromSmiles(sm)
    try:
        return Chem.MolToSmiles(mol, doRandom=True, isomericSmiles=True)
    except:
        return sm

def build_dicts(smiles, part_rare_words=0.0001):

    char2index = {"SOS":0, "EOS":1, "PAD":2, "UNK":3}
    char2count = {"SOS":1, "EOS":1, "PAD":1, "UNK":1}
    index2char = {0:"SOS", 1:"EOS", 2:"PAD", 3:"UNK"}
    n_words = 4


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

    all_words = sum(char2count.values())

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

    return char2index, char2count, index2char

def encode_smiles(smiles, char2index, max_len, augment = True):
    #Augment smiles
    if augment:
        augmented_smiles = [augment_smile(smile) for smile in smiles]
        smiles = [augmented_smiles[i] if len(augmented_smiles[i])<=98 else smiles[i] for i in range(len(smiles))]
    #Encode smiles
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
    #Cut smiles with max len. !Attention: user must exclude molecules manually or it will be cutted.
    lens = [len(encoded_smiles[i]) for i in range(len(encoded_smiles))]
    exclude = [i for i in range(len(lens)) if (lens[i] > max_len)]
    for i in exclude[::-1]:
        del encoded_smiles[i]
        del lens[i]
    for i in range(len(encoded_smiles)):
        while (len(encoded_smiles[i])) < max_len:
            encoded_smiles[i].append(char2index["PAD"])
    data = np.array(encoded_smiles)
    return data