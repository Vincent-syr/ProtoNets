"""

"""

from torch.utils.data import Sampler
import torch
import numpy as np


class CategoriesSampler(Sampler):
    def __init__(self, label, n_batch, n_cls=5, n_per=30):   
        """[summary]

        Args:
            label ([type]): [description]
            n_batch ([type]): number of batch for an epoch
            n_cls (int, optional): k_way * num_task
            n_per (int, optional): n_shot + q_query

        Returns:
            [type]: [description]
        """
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        print(n_batch)
        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)     


    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        """ iterater of each episode
        """
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
            

    
    
    
    
