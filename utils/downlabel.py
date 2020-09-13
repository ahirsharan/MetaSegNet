import torch

#Bring the labels from global labels to (0 to K-1) for meta tasks
def downlabel(labels,K):
    visited=[False]*100
    uniq=[]
    for x in labels:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if(visited[x[i][j]] == False):
                    visited[x[i][j]]=True
                    uniq.append(int(x[i][j]))

    uniq.sort()
    print(uniq)
    for x in labels:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i][j]=uniq.index(int(x[i][j]))
    return labels
  
