""" Model for meta-transfer learning. """
import  torch
import torch.nn as nn
import torch.nn.functional as F
from models.metasegnet import resnet9
from utils.losses import CE_DiceLoss
from utils.onehot import onehot

class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.vars = nn.ParameterList()
        self.lamb = nn.Parameter(torch.ones([1]))
        self.vars.append(self.lamb)
        self.alpha = nn.Parameter(torch.ones([1]))
        self.vars.append(self.alpha)
        self.beta= nn.Parameter(torch.zeros([1]))
        self.vars.append(self.beta)

    def forward(self, Xtr, ytr, Xte, Yte, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        l = the_vars[0]
        a = the_vars[1]
        b = the_vars[2]
        
   
        # weight of ridgle classifier calculated using train data
        id=(torch.eye(self.args.way+1))
        if(torch.cuda.is_available()):
            id=id.cuda()
         
        XT=torch.transpose(Xtr,0,1)
        
        if(torch.cuda.is_available()):
            XT=XT.cuda()        
            ytr=ytr.cuda()
        
        print(ytr.shape)
        w1 = torch.inverse(torch.matmul(XT,Xtr)+ (l*id))
        w2 = torch.matmul(XT,ytr.float())
        w  = torch.matmul(w1,w2)
        net=[]
        
        dim=Yte.shape[1]
        vec=torch.ones((dim,self.args.way+1))
        
        if(torch.cuda.is_available()):
            vec=vec.cuda()
            w=w.cuda()
            
        for x in Xte:
            net.append(a*torch.matmul(x,w) + b*vec)     
        return net

    def parameters(self):
        return self.vars
        
    def reinit(self):
        self.vars[0]=nn.Parameter(torch.ones([1]))
        self.vars[1]=nn.Parameter(torch.ones([1]))
        self.vars[2]=nn.Parameter(torch.zeros([1]))

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        self.base_learner = BaseLearner(args)
        num_classes=self.args.way+1
        self.encoder = resnet9(3,num_classes)  
        self.CD=CE_DiceLoss()
        
    def forward(self, im_train, Ytr, im_test, Yte):
        Xtr,_ = self.encoder(im_train)
        _,Xte = self.encoder(im_test)
        print(Xtr.shape)
        Gte = self.base_learner(Xtr, Ytr, Xte, Yte)
        loss =  self.CD(Gte,Yte)
        loss.requires_grad=True
        '''
        print(self.base_learner.parameters())
        for par in self.base_learner.parameters():
            print(par.data)
            print(par.requires_grad)
        '''
        print(loss)
        grad = torch.autograd.grad(loss, self.base_learner.parameters(),retain_graph=True,allow_unused=True)
        print(grad)
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
     
        for _ in range(1, self.update_step):
            Gte = self.base_learner(Xtr, Ytr, Xte, Yte, fast_weights)
            loss =  self.CD(Gte,Yte)
            loss.requires_grad=True
            grad = torch.autograd.grad(loss, fast_weights,retain_graph=True,allow_unused=True)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
        
        Gte = self.base_learner(Xtr, Ytr, Xte, Yte, fast_weights) 
        self.base_learner.reinit()
        return Gte
