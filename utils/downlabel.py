import torch

def index(arr,x):
    in=(-1)
    for i in range(len(arr)):
        if(arr[i]==x):
            in=i
            break
    return in

#Bring the labels from global labels to (0 to K-1) for meta tasks
def downlabel(labels,K,category):
    visited=[False]*100
    uniq=[]
    for x in labels:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if(visited[x[i][j]] == False):
                    visited[x[i][j]]=True
                    uniq.append(int(x[i][j]))
    uniqn=[]
    uniqn.append(uniq[0]) #Included Background Class    
    
    if(category=='Train'):
        for i in range(1,K):
            uniqn.append(uniq[i])
    else:
        for i in range(len(uniq)-1,len(uniq)-K,-1):
            uniqn.append(uniq[i])       
    
    uniqn.sort()
    for x in labels:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                in=index(uniqn,int(x[i][j]))
                if(in!=(-1)):
                    x[i][j]=in
                else:
                    x[i][j]=0
    
    #print(torch.unique(x))
    return labels
  
