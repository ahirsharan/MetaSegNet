import torch.nn.functional as F
import torch

def onehot(X,num_classes):
    ident=torch.eye(num_classes,dtype=int)
    X_onehot=ident[X]
    return X_onehot
