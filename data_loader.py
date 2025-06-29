# -*- coding: utf-8 -*-

'''
@Time    : 2021/12/12 17:44
@Author  : Zou Hao

'''
from __future__ import annotations

import csv
import functools
import json
import os
import pickle as pk
import random
import warnings
# from functools import cache
from pathlib import Path
from typing import Optional
from typing import TYPE_CHECKING, Any
import pickle

import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from pymatgen.core import Composition
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import LongTensor, Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Subset, random_split, SequentialSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import DataLoader as Graph_Dataloader
from tqdm import tqdm

from collections import OrderedDict, defaultdict


from base import BaseDataLoader
# from utils.embed import crystal_des_embedding
# from utils.line_graph import Graph, StructureLineDataset
from utils.composition import generate_features, _element_composition

if TYPE_CHECKING:
    from collections.abc import Sequence

# import dgl

data_type_torch = torch.float32
data_type_np = np.float32



class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """
    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax+step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-(distances[..., np.newaxis] - self.filter)**2 /
                      self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """
    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {idx: atom_type for atom_type, idx in
                            self._embedding.items()}

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, '_decodedict'):
            self._decodedict = {idx: atom_type for atom_type, idx in
                                self._embedding.items()}
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """
    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value
                          in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)

class CompositionData(Dataset):
    """Dataset class for the Roost composition model."""

    def __init__(
        self,
        df: pd.DataFrame,
        task_dict: dict[str, str],
        elem_embedding: str = "matscholar200",
        inputs: str = "composition",
        identifiers: Sequence[str] = ("material_id", "composition"),
    ):
        """Data class for Roost models.

        Args:
            df (pd.DataFrame): Pandas dataframe holding input and target values.
            task_dict (dict[str, "regression" | "classification"]): Map from target
                names to task type. should include columns ['composition', 'material_id']
            elem_embedding (str, optional): One of "matscholar200", "cgcnn92",
                "megnet16", "onehot112" or path to a file with custom element
                embeddings. Defaults to "matscholar200".
            inputs (str, optional): df column name holding material compositions.
                Defaults to "composition".
            identifiers (list, optional): df columns for distinguishing data points.
                Will be copied over into the model's output CSV. Defaults to
                ["material_id", "composition"].
        """
        if len(identifiers) != 2:
            raise AssertionError("Two identifiers are required")

        self.inputs = inputs
        self.task_dict = task_dict
        self.identifiers = list(identifiers)
        self.df = df

        if elem_embedding in ["matscholar200", "cgcnn92", "megnet16", "onehot112"]:
            elem_embedding = f"data/embeddings/element/{elem_embedding}.json"

        with open(elem_embedding) as file:
            self.elem_features = json.load(file)

        self.elem_emb_len = len(next(iter(self.elem_features.values())))

        self.n_targets = []
        for target, task in self.task_dict.items():
            if task == "regression":
                self.n_targets.append(1)
            elif task == "classification":
                n_classes = np.max(self.df[target].values) + 1
                self.n_targets.append(n_classes)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        df_repr = f"cols=[{', '.join(self.df.columns)}], len={len(self.df)}"
        return f"{type(self).__name__}({df_repr}, task_dict={self.task_dict})"

    # Cache data for faster training
    # @cache  # noqa: B019
    def __getitem__(self, idx: int):
        """Get an entry out of the Dataset.

        Args:
            idx (int): index of entry in Dataset

        Returns:
            tuple: containing
            - tuple[Tensor, Tensor, LongTensor, LongTensor]: Roost model inputs
            - list[Tensor | LongTensor]: regression or classification targets
            - list[str | int]: identifiers like material_id, composition
        """
        row = self.df.iloc[idx]
        composition = row[self.inputs]
        material_ids = row[self.identifiers].to_list()

        comp_dict = Composition(composition).get_el_amt_dict()
        elements = list(comp_dict)

        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)
        try:
            elem_fea = np.vstack([self.elem_features[element] for element in elements])
        except AssertionError as exc:
            raise AssertionError(
                f"{material_ids} contains element types not in embedding"
            ) from exc
        except ValueError as exc:
            raise ValueError(
                f"{material_ids} composition cannot be parsed into elements"
            ) from exc

        n_elems = len(elements)
        self_idx = []
        nbr_idx = []
        for elem_idx in range(n_elems):
            self_idx += [elem_idx] * n_elems
            nbr_idx += list(range(n_elems))

        # convert all data to tensors
        elem_weights = Tensor(weights)
        elem_fea = Tensor(elem_fea)
        self_idx = LongTensor(self_idx)
        nbr_idx = LongTensor(nbr_idx)

        targets = []
        for target in self.task_dict:
            if self.task_dict[target] == "regression":
                targets.append(Tensor([row[target]]))
            elif self.task_dict[target] == "classification":
                targets.append(LongTensor([row[target]]))

        return (
            (elem_weights, elem_fea, self_idx, nbr_idx),
            targets,
            *material_ids,
        )


def composition_collate_batch(
    samples: tuple[
        tuple[Tensor, Tensor, LongTensor, LongTensor],
        list[Tensor | LongTensor],
        list[str | int],
    ],
) -> tuple[Any, ...]:
    """Collate a list of data and return a batch for predicting crystal properties.

    Args:
        samples (list): list of tuples for each data point where each tuple contains:
            (elem_fea, nbr_fea, nbr_idx, target)
            - elem_fea (Tensor):  _description_
            - nbr_fea (Tensor):
            - self_idx (LongTensor):
            - nbr_idx (LongTensor):
            - target (Tensor | LongTensor): target values containing floats for
                regression or integers as class labels for classification
            - cif_id: str or int

    Returns:
        tuple[
            tuple[Tensor, Tensor, LongTensor, LongTensor, LongTensor]: batched Roost
                model inputs,
            tuple[Tensor | LongTensor]: Target values for different tasks,
            # TODO this last tuple is unpacked how to do type hint?
            *tuple[str | int]: Identifiers like material_id, composition
        ]
    """
    # define the lists
    batch_elem_weights = []
    batch_elem_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_elem_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0
    for idx, (inputs, target, *cry_ids) in enumerate(samples):
        elem_weights, elem_fea, self_idx, nbr_idx = inputs

        n_sites = elem_fea.shape[0]  # number of atoms for this crystal

        # batch the features together
        batch_elem_weights.append(elem_weights)
        batch_elem_fea.append(elem_fea)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_elem_idx.append(torch.tensor([idx] * n_sites))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        cry_base_idx += n_sites

    return (
        (
            torch.cat(batch_elem_weights, dim=0),
            torch.cat(batch_elem_fea, dim=0),
            torch.cat(batch_self_idx, dim=0),
            torch.cat(batch_nbr_idx, dim=0),
            torch.cat(crystal_elem_idx),
        ),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        *zip(*batch_cry_ids),
    )



class EDMDataset(Dataset):
    """
    Get X and y from EDM dataset.
    """

    def __init__(self, dataset, n_comp):
        self.data = dataset
        self.n_comp = n_comp

        self.X = np.array(self.data[0])
        self.y = np.array(self.data[1])
        self.formula = np.array(self.data[2])

        self.shape = [(self.X.shape), (self.y.shape), (self.formula.shape)]

    def __str__(self):
        string = f'EDMDataset with X.shape {self.X.shape}'
        return string

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :, :]
        y = self.y[idx]
        formula = self.formula[idx]

        X = torch.as_tensor(X, dtype=data_type_torch)
        y = torch.as_tensor(y, dtype=data_type_torch)

        return (X, y, formula)

def get_edm(path, elem_prop='mat2vec', n_elements='infer',
            inference=False, verbose=True, drop_unary=True,
            scale=True):
    """
    Build a element descriptor matrix.

    Parameters
    ----------
    path : str
        DESCRIPTION.
    inference : bool, optional, default=False, IF True, (index reset)
    elem_prop : str, optional
        DESCRIPTION. The default is 'oliynyk'.

    Returns
    -------
    X_scaled : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    formula : TYPE
        DESCRIPTION.

    """
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    # path can either be string to csv or a dataframe with data already
    if isinstance(path, str):
        df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    else:
        df = path

    if 'formula' not in df.columns.values.tolist():
        df['formula'] = df['cif_id'].str.split('_ICSD').str[0]

    df['count'] = [len(_element_composition(form)) for form in df['formula']]
    if drop_unary:
        df = df[df['count'] != 1]  # drop pure elements
    if not inference:
        df = df.groupby(by='formula').mean().reset_index()  # mean of duplicates

    list_ohm = [OrderedDict(_element_composition(form))
                for form in df['formula']]
    list_ohm = [OrderedDict(sorted(mat.items(), key=lambda x:-x[1]))
                for mat in list_ohm]

    y = df['target'].values.astype(data_type_np)
    formula = df['formula'].values
    if n_elements == 'infer':
        # cap maximum elements at 16, and then infer n_elements
        n_elements = 16

    edm_array = np.zeros(shape=(len(list_ohm),
                                n_elements,
                                len(all_symbols)+1),
                         dtype=data_type_np)
    elem_num = np.zeros(shape=(len(list_ohm), n_elements), dtype=data_type_np)
    elem_frac = np.zeros(shape=(len(list_ohm), n_elements), dtype=data_type_np)
    for i, comp in enumerate(tqdm(list_ohm,
                                  desc="Generating EDM",
                                  unit="formulae",
                                  disable=not verbose)):
        for j, (elem, count) in enumerate(list_ohm[i].items()):
            if j == n_elements:
                # Truncate EDM representation to n_elements
                break
            try:
                edm_array[i, j, all_symbols.index(elem) + 1] = count
                elem_num[i, j] = all_symbols.index(elem) + 1
            except ValueError:
                print(f'skipping composition {comp}')

    if scale:
        # Normalize element fractions within the compound
        for i in range(edm_array.shape[0]):
            frac = (edm_array[i, :, :].sum(axis=-1)
                    / (edm_array[i, :, :].sum(axis=-1)).sum())
            elem_frac[i, :] = frac
    else:
        # Do not normalize element fractions, even for single-element compounds
        for i in range(edm_array.shape[0]):
            frac = edm_array[i, :, :].sum(axis=-1)
            elem_frac[i, :] = frac

    if n_elements == 16:
        n_elements = np.max(np.sum(elem_frac > 0, axis=1, keepdims=True))
        elem_num = elem_num[:, :n_elements]
        elem_frac = elem_frac[:, :n_elements]

    elem_num = elem_num.reshape(elem_num.shape[0], elem_num.shape[1], 1)
    elem_frac = elem_frac.reshape(elem_frac.shape[0], elem_frac.shape[1], 1)
    out = np.concatenate((elem_num, elem_frac), axis=1)

    return out, y, formula

class EDM_CsvLoader():
    """
    Parameters
    ----------
    csv_data: str
        name of csv file containing cif and properties
    csv_val: str
        name of csv file containing cif and properties
    val_frac: float, optional (default=0.75)
        train/val ratio if val_file not given
    batch_size: float, optional (default=64)
        Step size for the Gaussian filter
    random_state: int, optional (default=123)
        Random seed for sampling the dataset. Only used if validation data is
        not given.
    shuffle: bool (default=True)
        Whether to shuffle the datasets or not
    """

    def __init__(self, csv_data, batch_size=64,
                 num_workers=1, random_state=0, shuffle=True,
                 pin_memory=True, n_elements=6, inference=False,
                 verbose=True,
                 drop_unary=True,
                 scale=True):
        self.csv_data = csv_data
        self.main_data = list(get_edm(self.csv_data, elem_prop='mat2vec',
                                      n_elements=n_elements,
                                      inference=inference,
                                      verbose=verbose,
                                      drop_unary=drop_unary,
                                      scale=scale))
        self.n_train = len(self.main_data[0])
        self.n_elements = self.main_data[0].shape[1]//2

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.random_state = random_state

    def get_data_loaders(self, inference=False):
        '''
        Input the dataset, get train test split
        '''
        shuffle = not inference  # don't shuffle data when inferencing
        pred_dataset = EDMDataset(self.main_data, self.n_elements)
        pred_loader = DataLoader(pred_dataset,
                                 batch_size=self.batch_size,
                                 pin_memory=self.pin_memory,
                                 shuffle=shuffle)
        return pred_loader

def mag_data(data_path):
    """
    Load magnetic data from csv file.
    """
    data = np.load(data_path, allow_pickle=True)
    data = torch.tensor(data)
    return data

class MagpieDataset(Dataset):
    """
    PyTorch Dataset class for loading magnetic data from a CSV file.
    """
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_magpie_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Function to create a DataLoader for the MagneticDataset.
    """
    dataset = MagpieDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class MeredigDataset(Dataset):
    """
    PyTorch Dataset class for loading magnetic data from a CSV file.
    """
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_meredig_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Function to create a DataLoader for the MagneticDataset.
    """
    dataset = MeredigDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

class RSCDataset(Dataset):
    """
    PyTorch Dataset class for loading magnetic data from a CSV file.
    """
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class RSC_dataloader(DataLoader):
    def __init__(self, data_path, batch_size=32, shuffle=True, validation_split = 0.1, test_split = 0.1, random_seed = 42,
                 use_Kfold = True, total_fold = None, nth_fold = None, custom_collate_fn = None):
        dataset = RSCDataset(data_path)


        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.collate_fn = custom_collate_fn

        self.use_Kfold = use_Kfold
        self.total_fold = total_fold
        self.nth_fold = nth_fold

        super(RSC_dataloader, self).__init__(dataset, batch_size=batch_size,
                                              shuffle=shuffle)

    def _split_sampler(self):

        if self.use_Kfold:
            return self._Kfold_split_sampler(total_fold=self.total_fold, nth_fold=self.nth_fold)
        else:
            dataset_size = len(self.dataset)
            test_size = int(dataset_size * self.test_split)
            val_size = int(dataset_size * self.validation_split)
            train_size = dataset_size - val_size - test_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.random_seed)
            )

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)

            return train_loader, val_loader, test_loader

    def _Kfold_split_sampler(self, total_fold=5, nth_fold=0):
        """
         Split the dataset into train, validation, and test sets using KFold cross-validation.

         Args:
             KFold (int): Number of folds for cross-validation.
             nth_fold (int): The fold index to use for validation.
         """
        from sklearn.model_selection import train_test_split, KFold

        # Split the dataset into train+val and test sets
        train_val_indices, test_indices = train_test_split(
            range(len(self.dataset)), test_size=self.test_split, random_state=self.random_seed
        )

        # Create KFold object
        kf = KFold(n_splits=total_fold, shuffle=True, random_state=self.random_seed)

        # Split train+val set into train and val sets
        train_indices, val_indices = list(kf.split(train_val_indices))[nth_fold]

        # Map indices back to the original dataset
        train_indices = [train_val_indices[i] for i in train_indices]
        val_indices = [train_val_indices[i] for i in val_indices]

        # Create DataLoaders for train, val, and test sets
        train_loader = DataLoader(
            Subset(self.dataset, train_indices), batch_size=self.batch_size, shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            Subset(self.dataset, val_indices), batch_size=self.batch_size, shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )
        test_loader = DataLoader(
            Subset(self.dataset, test_indices), batch_size=self.batch_size, shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )

        return train_loader, val_loader, test_loader


def get_RSC_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Function to create a DataLoader for the MagneticDataset.
    """
    dataset = RSCDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


class ECDataset(Dataset):
    """
    PyTorch Dataset class for loading magnetic data from a CSV file.
    """
    def __init__(self, data_path):
        self.data = np.load(data_path, allow_pickle=True)
        self.data = torch.tensor(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_EC_dataloader(data_path, batch_size=32, shuffle=True):
    """
    Function to create a DataLoader for the MagneticDataset.
    """
    dataset = ECDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# class combine_loader(BaseDataLoader):
#     def __init__(self, Roost_data_path, CrabNet_data_path, Magpie_data_path, batch_size, shuffle, inference):
#         super().__init__()
#         self.Roost_data_path = Roost_data_path

import torch
from torch.utils.data import Dataset, DataLoader, random_split


class CombinedDataset(Dataset):
    def __init__(self, roost_data, crabnet_data, magpie_data, meredig_data, rsc_data, ec_data):
        self.roost_data = roost_data
        self.crabnet_data = crabnet_data
        self.magpie_data = magpie_data
        self.meredig_data = meredig_data
        self.rsc_data = rsc_data
        self.ec_data = ec_data

        self.length = min(len(roost_data), len(crabnet_data), len(magpie_data))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        roost_item = self.roost_data[idx]
        crabnet_item = self.crabnet_data[idx]
        magpie_item = self.magpie_data[idx]
        meredig_item = self.meredig_data[idx]
        rsc_item = self.rsc_data[idx]
        ec_item = self.ec_data[idx]
        return roost_item, crabnet_item, magpie_item, meredig_item, rsc_item, ec_item

def custom_collate_fn(batch):
    roost_batch, crabnet_batch, magpie_batch, meredig_batch, rsc_batch, ec_batch = zip(*batch)

    batch_elem_weights = []
    batch_elem_fea = []
    batch_self_idx = []
    batch_nbr_idx = []
    crystal_elem_idx = []
    batch_targets = []
    batch_cry_ids = []

    cry_base_idx = 0
    for idx, (inputs, target, *cry_ids) in enumerate(roost_batch):
        elem_weights, elem_fea, self_idx, nbr_idx = inputs

        n_sites = elem_fea.shape[0]  # number of atoms for this crystal

        # batch the features together
        batch_elem_weights.append(elem_weights)
        batch_elem_fea.append(elem_fea)

        # mappings from bonds to atoms
        batch_self_idx.append(self_idx + cry_base_idx)
        batch_nbr_idx.append(nbr_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_elem_idx.append(torch.tensor([idx] * n_sites))

        # batch the targets and ids
        batch_targets.append(target)
        batch_cry_ids.append(cry_ids)

        # increment the id counter
        cry_base_idx += n_sites

    # Collate CrabNet data
    crabnet_src = torch.cat([item[0].unsqueeze(0) for item in crabnet_batch], dim=0)
    crabnet_targets = torch.stack([item[1] for item in crabnet_batch])
    crabnet_ids = [item[2] for item in crabnet_batch]

    # Collate Magpie data
    magpie_fea = torch.cat([item.unsqueeze(0) for item in magpie_batch], dim=0)

    meredig_fea = torch.cat([item.unsqueeze(0) for item in meredig_batch], dim=0)

    rsc_fea = torch.cat([item.unsqueeze(0) for item in rsc_batch], dim=0)

    ec_fea = torch.cat([item.unsqueeze(0) for item in ec_batch], dim=0)

    return ((torch.cat(batch_elem_weights, dim=0),
        torch.cat(batch_elem_fea, dim=0),
        torch.cat(batch_self_idx, dim=0),
        torch.cat(batch_nbr_idx, dim=0),
        torch.cat(crystal_elem_idx)),
        tuple(torch.stack(b_target, dim=0) for b_target in zip(*batch_targets)),
        batch_cry_ids,
        crabnet_src,
        crabnet_targets,
        crabnet_ids,
        magpie_fea,
        meredig_fea,
        rsc_fea,
        ec_fea
    )


def get_combined_dataset(roost_data_path, crabnet_data_path, magpie_data_path, meredig_data_path, rsc_data_path, ec_data_path):
    roost_data = CompositionData(df=pd.read_csv(roost_data_path, na_values=[], keep_default_na=False), task_dict={'target': 'classification'})
    crabnet_data = EDMDataset(dataset=list(get_edm(crabnet_data_path)), n_comp=6)
    magpie_data = MagpieDataset(data_path=magpie_data_path)
    meredig_data = MeredigDataset(data_path=meredig_data_path)
    rsc_data = RSCDataset(data_path=rsc_data_path)
    ec_data = ECDataset(data_path=ec_data_path)
    combined_dataset = CombinedDataset(roost_data, crabnet_data, magpie_data, meredig_data, rsc_data, ec_data)
    return combined_dataset


class CombinedDataLoader(DataLoader):
    def __init__(self, roost_data_dir, crabnet_data_dir, magpie_data_dir, meredig_data_dir, rsc_data_dir, ec_data_dir, load_from_local_file = None,
                 batch_size=32, shuffle=True, validation_split=0.1, test_split=0.1, random_seed=42, use_Kfold=True, total_fold=None, nth_fold=None, collate_fn=custom_collate_fn):

        self.dataset = get_combined_dataset(roost_data_dir, crabnet_data_dir, magpie_data_dir, meredig_data_dir, rsc_data_dir, ec_data_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.validation_split = validation_split
        self.test_split = test_split
        self.random_seed = random_seed
        self.collate_fn = custom_collate_fn

        self.use_Kfold = use_Kfold
        self.total_fold = total_fold
        self.nth_fold = nth_fold

        self.load_from_local_file = load_from_local_file

        super().__init__(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

    def _split_sampler(self):
        if self.load_from_local_file:
            train_indices = pickle.load(open(self.load_from_local_file, 'rb'))[0]
            val_indices = pickle.load(open(self.load_from_local_file, 'rb'))[1]
            test_indices = pickle.load(open(self.load_from_local_file, 'rb'))[2]

            train_loader = DataLoader(
                Subset(self.dataset, train_indices), batch_size=self.batch_size, shuffle=self.shuffle,
                collate_fn=self.collate_fn
            )
            val_loader = DataLoader(
                Subset(self.dataset, val_indices), batch_size=self.batch_size, shuffle=self.shuffle,
                collate_fn=self.collate_fn
            )
            test_loader = DataLoader(
                Subset(self.dataset, test_indices), batch_size=self.batch_size, shuffle=self.shuffle,
                collate_fn=self.collate_fn
            )

            return train_loader, val_loader, test_loader


        if self.use_Kfold:
            return self._Kfold_split_sampler(total_fold=self.total_fold, nth_fold=self.nth_fold)
        else:
            dataset_size = len(self.dataset)
            test_size = int(dataset_size * self.test_split)
            val_size = int(dataset_size * self.validation_split)
            train_size = dataset_size - val_size - test_size

            train_dataset, val_dataset, test_dataset = random_split(
                self.dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(self.random_seed)
            )

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn)

            return train_loader, val_loader, test_loader

    def _Kfold_split_sampler(self, total_fold=5, nth_fold=0):
        """
         Split the dataset into train, validation, and test sets using KFold cross-validation.

         Args:
             KFold (int): Number of folds for cross-validation.
             nth_fold (int): The fold index to use for validation.
         """
        from sklearn.model_selection import train_test_split, KFold

        # Split the dataset into train+val and test sets
        train_val_indices, test_indices = train_test_split(
            range(len(self.dataset)), test_size=self.test_split, random_state=self.random_seed
        )

        # Create KFold object
        kf = KFold(n_splits=total_fold, shuffle=True, random_state=self.random_seed)

        # Split train+val set into train and val sets
        train_indices, val_indices = list(kf.split(train_val_indices))[nth_fold]

        # Map indices back to the original dataset
        train_indices = [train_val_indices[i] for i in train_indices]
        val_indices = [train_val_indices[i] for i in val_indices]

        # Create DataLoaders for train, val, and test sets
        train_loader = DataLoader(
            Subset(self.dataset, train_indices), batch_size=self.batch_size, shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )
        val_loader = DataLoader(
            Subset(self.dataset, val_indices), batch_size=self.batch_size, shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )
        test_loader = DataLoader(
            Subset(self.dataset, test_indices), batch_size=self.batch_size, shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )

        return train_loader, val_loader, test_loader

def collate_alignn(batch):
    """Dataloader helper to batch graphs cross `samples`."""
    graphs,  labels = zip(*batch)
    batched_graph = dgl.batch(graphs)

    return batched_graph, torch.tensor(labels)


#
if __name__ == '__main__':
    # a = get_edm('data/mp_2018_small/crabnet_template.csv', inference=True)
    roost_data_path = 'data/MP/mp_data_Roost.csv'
    df = pd.read_csv(roost_data_path)[:1000]
    import dgl

    # dataset = CompositionData(df, task_dict={'target':'classification'})
    # graph = []
    # for i in range(len(dataset)):
    #     a = dataset[i]
    #     b = a[0]
    #
    #     node_fea = b[1]
    #     edge_i = b[2]
    #     edge_j = b[3]
    #
    #     g = dgl.graph((edge_i, edge_j), num_nodes=node_fea.size()[0])
    #     graph.append(g)

    # torch.save(graph, 'data/MP/graph.pt')

    bbb = torch.load('data/MP/graph.pt')
    y = df['target'].values

    class CustomDataset(Dataset):
        def __init__(self, X, y):
            super(CustomDataset, self).__init__()
            self.graph = X
            self.y = y
        def __len__(self):
            return len(self.y)
        def __getitem__(self, index):
            return self.graph[index], self.y[index]

    dataset = CustomDataset(bbb, y)

    loader = DataLoader(dataset, batch_size=8, collate_fn=collate_alignn)

    for batch in loader:
        g,  y = batch
        break


    # for data1, data2, data3 in zip(roost_loader, crabnet_loader, mag_loader):
    #     inputs1, targets1, *ids1 = data1
    #     inputs2, targets2, *ids2 = data2
    #     inputs3 = data3
    #     y = model(inputs1, inputs2, inputs3)
    #     print(y.size())
    #     break


'''
Roost

    from model.composition import Roost

    a = CompositionData(df=pd.read_csv('data/mp_2018_small/description_small.csv'),task_dict={'bulk_moduli':'regression'})
    loader = DataLoader(a, batch_size=1, collate_fn=composition_collate_batch)
    model = Roost(elem_emb_len=200, n_targets=[1],  n_graph=3)

    for idx, (inputs, targets, *ids) in enumerate(loader):
        print(inputs)
        print(targets)
        print(ids)
        a, b, c, d, e = inputs
        y = model(a, b, c, d, e)
        if idx == 1:
            break


'''

'''
CrabNet

    crabnet_loader = EDM_CsvLoader('data/mp_2018_small/crabnet_template.csv', batch_size=16, shuffle=False, inference=True)
    crabnet_loader = crabnet_loader.get_data_loaders(inference=True)

    for idx, (inputs, targets, *ids) in enumerate(crabnet_loader):
        print(inputs)
        print(targets)
        print(ids)
        if idx == 1:
            break

    src, frac = inputs.squeeze(-1).chunk(2, dim=1)
    frac = frac * (1 + (torch.randn_like(frac)) * 0.02)  # normal
    frac = torch.clamp(frac, 0, 1)
    frac[src == 0] = 0
    frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])

    src = src.long()
    from model.composition import CrabNet
    crab_model = CrabNet()
    crab_y = crab_model(src, frac)





'''
