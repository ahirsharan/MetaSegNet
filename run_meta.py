""" Generate commands for meta-train phase. """
import os

# N shot and K way
def run_exp(num_batch=50, N=1, Q=1, lr=0.001, update_step=20):
    max_epoch = 4
    step_size = 1
    K = 2            #Background class not included. Adjust accordingly further.
    gpu = 0
    mtype = 'Net'
    valdata = 'No'
    
    #dataset='COCO'
    #dataset_dir='../Datasets/COCOAug/'
    
    #dataset='FSS1000'
    #dataset_dir='../Datasets/FSS1000Aug/'
    
    dataset='PASCAL'
    dataset_dir='../Datasets/Pascal5Aug/'
    
    #dataset='PASCALv'
    #dataset_dir='../Datasets/Pascal5ValAug/'   
    
    #num_batch is episodes   
    the_command = 'python3 main.py' \
        + ' --dataset=' +str(dataset) \
        + ' --dataset_dir=' +str(dataset_dir) \
        + ' --mtype=' +str(mtype) \
        + ' --valdata=' +str(valdata) \
        + ' --max_epoch=' + str(max_epoch) \
        + ' --num_batch=' + str(num_batch) \
        + ' --train_query=' + str(N) \
        + ' --test_query=' + str(Q) \
        + ' --meta_lr=' + str(lr) \
        + ' --step_size=' + str(step_size) \
        + ' --gpu=' + str(gpu) \
        + ' --update_step=' + str(update_step) \
        + ' --way=' + str(K) 

    os.system(the_command + ' --phase=train')
    os.system(the_command + ' --phase=test')

#N shots should be changed
#Q can be kept the same ie 2
run_exp(num_batch=2, N=1, Q=2, lr=0.001, update_step=100)
