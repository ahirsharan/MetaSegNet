""" Trainer for meta-train phase. """
import os.path as osp
import os
import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.mtl import MtlLearner
from utils.misc import Averager, Timer, ensure_path
from tensorboardX import SummaryWriter
from dataloader.samplers import CategoriesSampler
from utils.metrics import eval_metrics
from utils.losses import CE_DiceLoss
from utils.downlabel import downlabel
from dataloader.dataset_loader import DatasetLoader as Dataset
from torchvision import transforms
from utils.decode import decode_segmap
import math
from utils.onehot import onehot
from PIL import Image

class MetaTrainer(object):
    """The class that contains the code for the meta-train phase and meta-eval phase."""
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        save_image_dir='../results1/'
        if not osp.exists(save_image_dir):
            os.mkdir(save_image_dir)        
        
        log_base_dir = '../logs1/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        meta_base_dir = osp.join(log_base_dir, 'meta')
        if not osp.exists(meta_base_dir):
            os.mkdir(meta_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'MTL'])
        save_path2 = 'shot' + str(args.train_query) + '_way' + str(args.way) + '_query' + str(args.train_query) + \
            '_step' + str(args.step_size) + '_gamma' + str(args.gamma) + '_lr' + str(args.meta_lr) + \
            '_batch' + str(args.num_batch) + '_maxepoch' + str(args.max_epoch) + \
            '_baselr' + str(args.base_lr) + '_updatestep' + str(args.update_step) + \
            '_stepsize' + str(args.step_size) + '_' + args.meta_label
        args.save_path = meta_base_dir + '/' + save_path1 + '_' + save_path2
        args.save_image_dir=save_image_dir
        ensure_path(args.save_path)

        # Set args to be shareable in the class
        self.args = args

        # Load meta-train set
        self.trainset = Dataset('train', self.args)
        self.train_sampler = CategoriesSampler(self.trainset.labeln, self.args.num_batch, self.args.way+1, self.args.train_query, self.args.test_query)
        self.train_loader = DataLoader(dataset=self.trainset, batch_sampler=self.train_sampler, num_workers=8, pin_memory=True)
        
        # Build meta-transfer learning model
        self.model = MtlLearner(self.args)
        self.CD=CE_DiceLoss()
        
        # Set optimizer 
        self.optimizer = torch.optim.Adam([{'params': filter(lambda p: p.requires_grad, self.model.encoder.parameters())}, \
            {'params': self.model.base_learner.parameters(), 'lr': self.args.meta_lr}], lr=self.args.meta_lr)
        
        # Set learning rate scheduler 
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)        
        
        # load pretrained model
        self.model_dict = self.model.state_dict()

        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()

    def _reset_metrics(self):
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0
    
    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union
    
    def _get_seg_metrics(self,n_class):
        self.n_class=n_class
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "Pixel_Accuracy": np.round(pixAcc, 3),
            "Mean_IoU": np.round(mIoU, 3),
            "Class_IoU": dict(zip(range(self.n_class), np.round(IoU, 3)))
        }
        
    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """  
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))           

    def train(self):
        """The function for the meta-train phase."""

        # Set the meta-train log
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['train_acc'] = []
        trlog['train_iou'] = []
        trlog['max_iou'] = 0.0
        trlog['max_iou_epoch'] = 0

        # Set the timer
        timer = Timer()
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)
                
        K=self.args.way+1       #included Background as class
        N=self.args.train_query
        Q=self.args.test_query
        
        # Start meta-train
        for epoch in range(1, self.args.max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            train_iou_averager = Averager()

            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            self._reset_metrics()
            for i, batch in enumerate(tqdm_gen, 1):
                # Update global count number 
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, labels,_ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    labels = batch[1]
                
                #print(data.shape)
                #print(labels.shape)
                p = K*N
                im_train, im_test = data[:p], data[p:]
                
                #Adjusting labels for each meta task
                labels=downlabel(labels,K) 
                out_train, out_test = labels[:p],labels[p:]
                
                '''
                print(im_train.shape)
                print(im_test.shape)
                print(out_train.shape)
                print(out_test.shape)
                '''
                if(torch.cuda.is_available()):
                    im_train=im_train.cuda()
                    im_test=im_test.cuda()
                    out_train=out_train.cuda()
                    out_test=out_test.cuda()
                        
                #Reshaping train set ouput
                Ytr = out_train.reshape(-1)
                Ytr = onehot(Ytr,K) #One hot encoding for loss
                
                Yte = out_test.reshape(out_test.shape[0],-1)
                if(torch.cuda.is_available()):
                    Ytr=Ytr.cuda()
                    Yte=Yte.cuda()
                
                # Output logits for model
                Gte = self.model(im_train,Ytr,im_test, Yte)
                GteT=torch.transpose(Gte,1,2)
                # Calculate meta-train loss
                loss = self.CD(GteT,Yte)
                
                # Calculate meta-train accuracy
                seg_metrics = eval_metrics(GteT, Yte, K)
                self._update_seg_metrics(*seg_metrics)
                pixAcc, mIoU, _ = self._get_seg_metrics(K).values()
                
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f} IoU={:.4f}'.format(epoch, loss.item(), pixAcc*100.0,mIoU))

                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(pixAcc)
                train_iou_averager.add(mIoU)

                # Loss backwards and optimizer updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Update the averagers
            train_loss_averager = train_loss_averager.item()
            train_acc_averager = train_acc_averager.item()
            train_iou_averager = train_iou_averager.item()

            # Update best saved model
            if train_iou_averager > trlog['max_iou']:
                print("New Best!")
                trlog['max_iou'] = train_iou_averager
                trlog['max_iou_epoch'] = epoch
                self.save_model('max_iou')
                
            # Save model every 5 epochs
            if epoch % 2 == 0:
                self.save_model('epoch'+str(epoch))

            # Update the logs
            trlog['train_loss'].append(train_loss_averager)
            trlog['train_acc'].append(train_acc_averager)
            trlog['train_iou'].append(train_iou_averager)
            
            # Save log
            torch.save(trlog, osp.join(self.args.save_path, 'trlog'))

            if epoch % 1 == 0:
                print('Running Time: {}, Estimated Time: {}'.format(timer.measure(), timer.measure(epoch / self.args.max_epoch)))
                print('Epoch:{}, Average CE Loss: {:.4f}, Average mean IoU: {:.4f}'.format(epoch, train_loss_averager, train_iou_averager))
            
        writer.close()

    def eval(self):
        """The function for the meta-evaluate (test) phase."""
        # Load the logs
        trlog = torch.load(osp.join(self.args.save_path, 'trlog'))

        # Load meta-test set
        self.test_set = Dataset('test', self.args)
        self.sampler = CategoriesSampler(self.test_set.labeln, self.args.num_batch, self.args.way+1, self.args.train_query, self.args.test_query)
        self.loader = DataLoader(dataset=self.test_set, batch_sampler=self.sampler, num_workers=8, pin_memory=True)
    
        # Load model for meta-test phase
        if self.args.eval_weights is not None:
            self.model.load_state_dict(torch.load(self.args.eval_weights)['params'])
        else:
            self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_iou' + '.pth'))['params'])
        # Set model to eval mode
        self.model.eval()

        # Set accuracy averager
        ave_acc = Averager()

        # Start meta-test
        K=self.args.way+1
        N=self.args.train_query
        Q=self.args.test_query        
        
        self._reset_metrics()
        count=1
        for i, batch in enumerate(self.loader, 1):
            if torch.cuda.is_available():
                data, labels,_ = [_.cuda() for _ in batch]
            else:
                data = batch[0]
                labels = batch[1]

            p = K*N
            im_train, im_test = data[:p], data[p:]
            
            #Adjusting labels for each meta task  
            labels=downlabel(labels,K)
            out_train, out_test = labels[:p], labels[p:]
            
            if(torch.cuda.is_available()):
                im_train=im_train.cuda()
                im_test=im_test.cuda()
                out_train=out_train.cuda()
                out_test=out_test.cuda()
            
            #Reshaping train set ouput
            Ytr = out_train.reshape(-1)
            Ytr = onehot(Ytr,K) #One hot encoding for loss
                
            Yte = out_test.reshape(out_test.shape[0],-1)
            
            if(torch.cuda.is_available()):
                Ytr=Ytr.cuda()
                Yte=Yte.cuda()            
            # Output logits for model
            Gte = self.model(im_train,Ytr,im_test, Yte)
            GteT=torch.transpose(Gte,1,2)           
           
            # Calculate meta-train accuracy
            seg_metrics = eval_metrics(GteT, Yte, K)
            self._update_seg_metrics(*seg_metrics)     
            pixAcc, mIoU, _ = self._get_seg_metrics(K).values()
            
            ave_acc.add(pixAcc)
            #test_acc_record[i-1] = acc
            #if i % 100 == 0:
                #print('batch {}: {Average Accuracy:.2f}({Pixel Accuracy:.2f} {IoU :.2f} )'.format(i, ave_acc.item() * 100.0, pixAcc * 100.0,mIoU))
                
            #Saving Test Image, Ground Truth Image and Predicted Image
            for j in range(K*Q):
                
                x1 = im_test[j].detach().cpu()
                y1 = out_test[j].detach().cpu()
                z1 = GteT[j].detach().cpu()
                z1 = torch.argmax(z1,axis=0)
                
                m=int(math.sqrt(z1.shape[0])) 
                z2 = z1.reshape(m,m)
                
                x = transforms.ToPILImage()(x1).convert("RGB")
                y = Image.fromarray(decode_segmap(y1,K))
                z = Image.fromarray(decode_segmap(z2,K))
                
                px=self.args.save_image_dir+str(count)+'a.jpg'
                py=self.args.save_image_dir+str(count)+'b.png'
                pz=self.args.save_image_dir+str(count)+'c.png'
                x.save(px)
                y.save(py)
                z.save(pz)
                count=count+1
                
            print("Images Saved!")
        # Calculate the confidence interval, update the logs
        #print('Val Best Epoch {}, Acc {:.4f}, Test Acc {:.4f}'.format(trlog['max_acc_epoch'], trlog['max_acc']*100.0, ave_acc.item()*100.0))
        
