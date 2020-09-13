import torch.nn.functional as F
import torch

def onehot(X,num_classes):
    net=[]
    for x in X:
        xp=x.long()
        xp[xp>=num_classes]=num_classes-1
        xp[xp<0]=0
        m=F.one_hot(xp,num_classes)
        y=torch.ones((num_classes,xp.shape[0],xp.shape[1])).long()
    
        for i in range(m.shape[-1]):
            y[i,:,:] = m[:,:,i]
    
        if(torch.cuda.is_available()):
            y=y.cuda()
        net.append(y)
        
    net = torch.stack(net)
    net=net.double()
    if(torch.cuda.is_available()):
        net=net.cuda()
        
    return net
