""" Generate commands for meta-train phase. """
import os

# N shot and K way
def run_exp(num_batch=50, N=1, Q=1, lr=0.001, update_step=20):
    max_epoch = 40
    step_size = 20
    K = 2            #Backround class not included. Adjust accordingly further.
    gpu = 1
    #num_batch is episodes   
    the_command = 'python3 main.py' \
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

run_exp(num_batch=1000, N=5, Q=3, lr=0.001, update_step=50)
