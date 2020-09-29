""" Main function for this repo. """
import argparse
import torch
from utils.misc import pprint
from utils.gpu_tools import set_gpu
from trainer.meta import MetaTrainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Basic parameters
    parser.add_argument('--model_type', type=str, default='MetaSegNet', choices=['MetaSegNet']) # The network architecture
    parser.add_argument('--dataset', type=str, default='PASCAL', choices=['COCO','PASCAL','FSS1000']) # Dataset
    parser.add_argument('--phase', type=str, default='train', choices=['train','test']) # Phase
    parser.add_argument('--seed', type=int, default=0) # Manual seed for PyTorch, "0" means using random seed
    parser.add_argument('--gpu', default='1') # GPU id
    parser.add_argument('--mtype', type=str, default='Net', choices=['Net','Net-NG','Conv']) # Model Type: Net = MetaSegNet; Net-NG = MetaSegNet-NG; Conv = MetaSegConv
    parser.add_argument('--dataset_dir', type=str, default='../Datasets/Pascal5Aug/', choices=['../Datasets/COCOAug/','../Datasets/Pascal5Aug/','../Datasets/FSS1000Aug/']) # Dataset folder

    # Parameters for meta-train phase    
    parser.add_argument('--max_epoch', type=int, default=40) # Epoch number for meta-train phase
    parser.add_argument('--num_batch', type=int, default=1000) # The number for different tasks used for meta-train
    parser.add_argument('--way', type=int, default=3) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=3) # (Shot) The number of meta train samples for each class in a task
    parser.add_argument('--test_query', type=int, default=2) # The number of meta test samples for each class in a task
    parser.add_argument('--meta_lr', type=float, default=0.001) # Learning rate for feature embedding
    parser.add_argument('--update_step', type=int, default=50) # The number of updates for the inner loop
    parser.add_argument('--step_size', type=int, default=20) # The number of epochs to reduce the meta learning rates
    parser.add_argument('--gamma', type=float, default=0.5) # Gamma for the meta-train learning rate decay
    parser.add_argument('--eval_weights', type=str, default=None) # The meta-trained weights for inference
    parser.add_argument('--base_lr', type=float, default=0.01) # Learning rate for the inner loop
    parser.add_argument('--meta_label', type=str, default='exp1') # Additional label for meta-train

    # Set and print the parameters
    args = parser.parse_args()
    pprint(vars(args))

    # Set the GPU id
    set_gpu(args.gpu)

    # Set manual seed for PyTorch
    if args.seed==0:
        print ('Using random seed.')
        torch.backends.cudnn.benchmark = True
    else:
        print ('Using manual seed:', args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Start trainer for pre-train, meta-train or meta-eval
    if args.phase=='train':
        trainer = MetaTrainer(args)
        trainer.train()
    elif args.phase=='test':
        trainer = MetaTrainer(args)
        trainer.eval()
    else:
        raise ValueError('Please set correct phase.')
