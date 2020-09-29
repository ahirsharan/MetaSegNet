import torch

def index(arr,x):
    ind=(-1)
    for i in range(len(arr)):
        if(arr[i]==x):
            ind=i
            break
    return ind

#Bring the labels from global labels to (0 to K-1) for meta tasks based on maximum occurence

def downlabel(labels,K):
    visited=[False]*100
    count=[0]*1005
    for x in labels:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                    count[int(x[i][j])]=count[int(x[i][j])]+1
                    if(visited[int(x[i][j])]==False):
                        visited[int(x[i][j])]=True
    '''
    uniqn=[]
    uniqn.append(uniq[0]) #Included Background Class    
    
    if(category=='Train'):
        for i in range(1,min(K,len(uniq))):
            uniqn.append(uniq[i])
    else:
        for i in range(len(uniq)-1,max(len(uniq)-K,0),-1):
            uniqn.append(uniq[i])       
    
    uniqn.sort()
    '''
    uniqn=[]
    for i in range(K):
        maxv=0
        inx=-1
        for j in range(100):
            if(visited[j]==True and count[j]>=maxv):
                maxv=count[j]
                inx=j
        if(inx!=-1):
            uniqn.append(inx)
            visited[inx]=False
    uniqn.sort()                          
                                                        
    for x in labels:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                ind=index(uniqn,int(x[i][j]))
                if(ind!=(-1)):
                    x[i][j]=ind
                else:
                    x[i][j]=0
    
    #print(torch.unique(x))
    return labels
  
