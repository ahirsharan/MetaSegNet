""" Sampler for dataloader. """
import torch
import numpy as np
import random

# Customize such as total way number of distinct classes to segment in a meta task

class CategoriesSampler():
    """The class to generate episodic data"""
    def __init__(self, labeln, n_batch, K, N, Q):
        #K Way, N shot(train query), Q(test query)
        
        self.n_batch = n_batch
        self.K = K
        self.N = N
        self.Q = Q

        labeln = np.array(labeln)
        self.m_ind = []
        for i in range(max(labeln) + 1):
            ind = np.argwhere(labeln == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            classes = torch.randperm(len(self.m_ind))[:(self.K-1)]
            lr=[]
            dr=[]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:(self.N +self.Q)]
                m=l[pos]
                
                for i in range(0,self.N):
                    lr.append(m[i])
                    
                for i in range(self.N, (self.N +self.Q)):
                    dr.append(m[i])
            
            # redundancy for background
            h=random.randint(0,self.K-1)
            c=classes[h]
            l = self.m_ind[c]
            pos = torch.randperm(len(l))[:(self.N +self.Q)]
            m=l[pos]

            for i in range(0,self.N):
                lr.append(m[i])

            for i in range(self.N, (self.N +self.Q)):
                dr.append(m[i])   

            batch=[]
            for i in range(len(lr)):
                batch.append(lr[i])
            
            for i in range(len(dr)):
                batch.append(dr[i])
                        
            batch = torch.stack(batch).t().reshape(-1)      
                
            yield batch
