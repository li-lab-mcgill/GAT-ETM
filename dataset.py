import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix

from IPython import embed

def coo2tensor(coo, device):
    i = torch.LongTensor(np.vstack((coo.row, coo.col)))
    v = torch.FloatTensor(coo.data)
    shape = coo.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape), device=device)

def csr2tensor(csr, device):
    return coo2tensor(csr.tocoo(), device)

def csc2tensor(csc, device):
    return coo2tensor(csc.tocoo(), device)


def normalize(data, vocab_cum, bow_norm=True):
    if not bow_norm:
        return data
    normalized_data = torch.zeros(data.shape, device=data.device)
    for i, _ in enumerate(vocab_cum[:-1]):
        data_i = data[:,vocab_cum[i]:vocab_cum[i+1]]
        sum_i = data_i.sum(-1).unsqueeze(-1) + 1e-6
        normalized_data[:,vocab_cum[i]:vocab_cum[i+1]] = data_i / sum_i
    return normalized_data
    


class NontemporalDataset(Dataset):
    def __init__(self, phase, npy_file, code_type_info, device, transform=None, drug_imputation=False, drug_count_thr=1):
        self.phase = phase
        self.device = device
        if phase == 'test': 
            npy_file, npy_file_1, npy_file_2 = npy_file
            self.bow = np.load(npy_file, allow_pickle=True).item()
            self.bow_1 = np.load(npy_file_1, allow_pickle=True).item()
            self.bow_2 = np.load(npy_file_2, allow_pickle=True).item()
            # FIXME: 10-14 sparse tensor dataset
            self.bow_1 = csr2tensor(self.bow_1, device)
            self.bow_2 = csr2tensor(self.bow_2, device)
        else:
            self.bow = np.load(npy_file, allow_pickle=True).item()

        self.code_types, self.vocab_size, self.vocab_cum = code_type_info

        self.N = self.bow.shape[0]
        self.V = self.bow.shape[1]

        self.transform = transform

        if phase == 'train': 
            bow_coo = self.bow.tocoo()
            self.stack_bow = {}
            for i, c in enumerate(self.code_types):
                chs = (bow_coo.col >= self.vocab_cum[i]) * (bow_coo.col < self.vocab_cum[i+1])
                stack_row_i = list(bow_coo.row[chs])
                stack_col_i = list(bow_coo.col[chs] - self.vocab_cum[i])
                stack_data_i = list(bow_coo.data[chs])

                self.stack_bow[c] = coo_matrix((stack_data_i, (stack_row_i, stack_col_i)), shape=(self.N, self.vocab_size[i]))

            if drug_imputation:
                self.drug_count_thr = drug_count_thr
                non_drug_count = np.array(self.bow[:,:self.vocab_cum[1]].sum(1)).squeeze(1)
                drug_count = np.array(self.bow[:,self.vocab_cum[1]:].astype('bool').sum(1)).squeeze(1)
                # np.save(open('train_drug_count.npy','wb'), drug_count[non_drug_count>0])
                drug_count = drug_count * (drug_count >= self.drug_count_thr)
                self.imputation_idx = np.nonzero(non_drug_count * drug_count)[0]
                self.imputation_data = self.bow[self.imputation_idx]
                imputation_drug_occur = self.imputation_data.astype('bool')[:, self.vocab_cum[1]:]
                self.imputation_drug_freq = torch.from_numpy(np.array(imputation_drug_occur.tocsc().sum(0)).squeeze(0)).to(device)
                print(f"# (imputataion train samples (drug_counts>={drug_count_thr})): {self.imputation_data.shape[0]}")
                self.imputation_data = csr2tensor(self.imputation_data, device).coalesce()

        if phase == 'test':
            if drug_imputation:
                self.drug_count_thr = drug_count_thr
                non_drug_count = np.array(self.bow[:,:self.vocab_cum[1]].sum(1)).squeeze(1)
                drug_count = np.array(self.bow[:,self.vocab_cum[1]:].astype('bool').sum(1)).squeeze(1)
                # drug_count = np.array(self.bow[:,self.vocab_cum[1]:].sum(1)).squeeze(1)
                # np.save(open('test_drug_count.npy','wb'), drug_count[non_drug_count>0])
                drug_count = drug_count * (drug_count >= drug_count_thr)
                self.imputation_idx = np.nonzero(non_drug_count * drug_count)[0]
                self.imputation_data = self.bow[self.imputation_idx]
                imputation_drug_occur = self.imputation_data.astype('bool')[:, self.vocab_cum[1]:]
                self.imputation_drug_freq = torch.from_numpy(np.array(imputation_drug_occur.tocsc().sum(0)).squeeze(0)).to(device)
                print(f"# (imputataion test samples (drug_counts>={drug_count_thr})): {self.imputation_data.shape[0]}")
                self.imputation_data = csr2tensor(self.imputation_data, device).coalesce()
        
        self.bow = csr2tensor(self.bow, device).coalesce()

    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        # data = torch.from_numpy(self.bow[idx].todense()).squeeze(0)
        # FIXME: 10-14 sparse tensor dataset
        # data = csr2tensor(self.bow[idx])
        data = self.bow[idx]

        sample = {'Data': data}

        if self.phase == 'test':
            # FIXME: 10-14 sparse tensor dataset
            # data_1 = csr2tensor(self.bow_1[idx])
            # data_2 = csr2tensor(self.bow_2[idx])
            data_1 = self.bow_1[idx]
            data_2 = self.bow_2[idx]

            sample.update({'Data_1': data_1, 'Data_2': data_2})

        if self.transform:
            sample = self.transform(sample)

        return sample, idx


class TemporalDataset(Dataset):
    def __init__(self, phase, npy_file, code_type_info, transform=None):
        self.aggregate = aggregate 
        self.bow = np.load(npy_file, allow_pickle=True)
        self.code_types, self.vocab_size, self.vocab_cum = code_type_info
        self.transform = transform
        self.phase = phase

        # FIXME: copied from non-temporal case
        T = 11
        N = self.bow.shape[0]
        V = self.bow.shape[1]


        self.stack_bow = {}
        for i, c in enumerate(self.code_types):
            chs = (self.bow.col >= self.vocab_cum[i]) * (self.bow.col < self.vocab_cum[i+1])
            stack_row_i = list(self.bow.row[chs])
            stack_col_i = list(self.bow.col[chs] - self.vocab_cum[i])
            stack_data_i += list(doc.data[chs])

            self.stack_bow[c] = coo_matrix((stack_data_i, (stack_row_i, stack_col_i)), shape=(N, self.vocab_size[i]))

    def __len__(self):
        return len(self.bow)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.bow[idx]
        
        if isinstance(data, coo_matrix):
            data = coo2tensor(data)
        else:
            data = torch.from_numpy(data)

        sample = {'Data': data}


        if self.transform:
            sample = self.transform(sample)

        return sample, idx
        
