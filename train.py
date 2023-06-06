"""
author:hao
date:10/21/2022

point cloud data preprocessing

python train.py --data_dir {data_dir} --checkpoint {checkpoint_dir} --log_dir {log_dir}

"""
import sys
import argparse
import numpy as np
import os
import torch
import datetime
import logging
from tqdm import tqdm
import sys
import importlib
import shutil
import torch.nn as nn
from torchsampler import ImbalancedDatasetSampler
from sklearn.metrics import roc_auc_score,roc_curve
import utils.provider as provider
from utils.provider import EarlyStopping
from utils.dataloader import DataLoader,collate_fn
import random
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR,'utils'))

from tensorboardX import SummaryWriter

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointDE Training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='pmlp_igm', help='model name [default: pointnet2_msg]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=1000, help='Point Number [default: 50]')

    parser.add_argument('--epoch',  default=100, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--use_res', type=bool, default=True, help='Use residual features')


    parser.add_argument('--data_dir', type=str, default='dockground61_1000', help='experiment root')
    parser.add_argument('--log_dir', type=str, default='log_dockground61_1000', help='experiment root')
    parser.add_argument('--checkpoint', type=str, default='checkpoint',help='checkpoints path')
    return parser.parse_args()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def log_string(logger,str):
    logger.info(str)
    print(str)

def main(args):
    args = parse_args()


    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    checkpoint_dir = Path(os.path.join(ROOT_DIR,args.checkpoint))
    checkpoint_dir.mkdir(exist_ok=True)

    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir_root = Path(os.path.join(ROOT_DIR,'log/'))
    experiment_dir_root.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir_root = experiment_dir_root.joinpath(timestr)
    else:
        experiment_dir_root = experiment_dir_root.joinpath(args.log_dir)
    experiment_dir_root.mkdir(exist_ok=True)
    experiment_dir = experiment_dir_root

    logger = logging.getLogger("model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (experiment_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string(logger,'PARAMETER ...')
    log_string(logger,args)

    '''TensorboardX'''
    # defual folder: ./runs
    writer = SummaryWriter()

    '''DATA LOADING'''
    log_string(logger,'Load dataset ...')
    DATA_PATH = os.path.join(ROOT_DIR,'data')
    DATA_PATH = os.path.join(DATA_PATH,args.data_dir)

    TRAIN_DATASET = DataLoader(root=DATA_PATH, split='train', use_res=args.use_res)
    VALID_DATASET = DataLoader(root=DATA_PATH, split='valid', use_res=args.use_res)

    '''MODEL LOADING'''
    log_string(logger,'Load model ...')
    MODEL = importlib.import_module(args.model)
    shutil.copy(os.path.join(ROOT_DIR, 'model','%s.py' % args.model), experiment_dir)
    

    patience = 5
    early_stopping = EarlyStopping(patience, verbose=True)
    
    '''SEED'''
    provider.set_seed(args.seed)
    logger.info('Set random seed ...')

    


    classifier = MODEL.PMLP_IGM().cuda()
    criterion = nn.BCELoss()
    #criterion = nn.CrossEntropyLoss().cuda()
    try:
        checkpoint = torch.load(os.path.join(args.checkpoint,'last_model.pth'))
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string(logger,'Use pretrain model')
    except:
        log_string(logger,'No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    train_size = TRAIN_DATASET.get_size()
    valid_size = VALID_DATASET.get_size()

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET,sampler=ImbalancedDatasetSampler(TRAIN_DATASET),batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True, collate_fn=collate_fn,worker_init_fn=seed_worker)
    validDataLoader = torch.utils.data.DataLoader(VALID_DATASET,sampler=ImbalancedDatasetSampler(VALID_DATASET),batch_size=args.batch_size, shuffle=False, num_workers=4 ,drop_last=True, collate_fn=collate_fn,worker_init_fn=seed_worker)

    #TRANING
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):

        log_string(logger,'Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))

        train_aucs = []
        mean_correct = []
        running_loss = 0.0
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            points, target, npoint = data
            points = provider.random_point_dropout(points)
            points[:,:, 0:3] = provider.shift_point_cloud(points[:,:, 0:3])

            points = torch.Tensor(points)
            target = torch.Tensor(target)
            #target = target[:, 0]

            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()

            classifier = classifier.train()
            pred = classifier(points)
            loss = criterion(pred, target)

            
            zero = torch.zeros_like(pred)
            one = torch.ones_like(pred)
            pred_choice = torch.where(pred > 0.5, one, zero)
            correct = pred_choice.eq(target.long().data).cpu().sum()

            mean_correct.append(correct.item() / float(points.size()[0]))
            running_loss+= loss.item()

            #train_aucs.append(metrics.roc_auc_score(target.data.cpu(),pred_choice.data.cpu()))

            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        #train_auc = np.mean(train_aucs)

        log_string(logger,'Train Instance Accuracy: %f' % train_instance_acc)
        #log_string('Train Instance AUC: %f' % train_auc)

        running_loss = running_loss / train_size
        writer.add_scalar('train_loss',running_loss,global_step=epoch)
        #writer.add_scalar('train_auc',train_auc,global_step=epoch)
        writer.add_scalar('train_acc',train_instance_acc,global_step=epoch)

        #Valid
        with torch.no_grad():
            mean_correct = []
            valid_aucs = []
            running_loss = 0.0
            class_acc = np.zeros((2,3))
            for j, data in tqdm(enumerate(validDataLoader), total=len(validDataLoader)):
                points, target, npoint = data
                points = torch.Tensor(points)
                target = torch.Tensor(target)
            
                points = points.transpose(2, 1)
                points, target= points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred = classifier(points)
                loss = criterion(pred, target)

                zero = torch.zeros_like(pred)
                one = torch.ones_like(pred)
                pred_choice = torch.where(pred > 0.5, one, zero)
                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target==cat].eq(target[target==cat].long().data).cpu().sum()
                    cat = int(cat)
                    class_acc[cat,0]+= classacc.item()
                    class_acc[cat,1]+= points[target==cat].size()[0]
                
                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item()/float(points.size()[0]))
                #valid_aucs.append(metrics.roc_auc_score(target.data.cpu(),pred_choice.data.cpu()))

                running_loss+= loss.item()
        

            class_acc[:,2] =  class_acc[:,0]/ class_acc[:,1]
            class_acc = np.mean(class_acc[:,2])
            valid_instance_acc = np.mean(mean_correct)
            running_loss = running_loss / valid_size

            #valid_auc = np.mean(valid_aucs)

            writer.add_scalar('valid_loss',running_loss,global_step=epoch)
            writer.add_scalar('valid_acc',valid_instance_acc,global_step=epoch)
            #writer.add_scalar('valid_auc',valid_auc,global_step=epoch)
            #writer.add_scalar('valid_class_acc',class_acc,global_step=epoch)
            #writer.add_scalar('lr',get_lr(optimizer),global_step=epoch)

            log_string(logger,'Valid Instance Accuracy: %f, Class Accuracy: %f'% (valid_instance_acc, class_acc))
            log_string(logger,'Best Instance Accuracy: %f, Class Accuracy: %f'% (best_instance_acc, best_class_acc))

            if (class_acc >= best_class_acc and epoch>5):
                best_instance_acc = valid_instance_acc
                best_class_acc = class_acc
                best_epoch = epoch + 1
                logger.info('Save model...')
                savepath = os.path.join(args.checkpoint,'best_model.pth')
                log_string(logger,'Saving at %s'% savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': valid_instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

            savepath = os.path.join(args.checkpoint,'last_model.pth')
            log_string(logger,'Saving at %s'% savepath)
            state = {
                'epoch': epoch,
                'instance_acc': valid_instance_acc,
                'class_acc': class_acc,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }
            torch.save(state, savepath)

            if (epoch>5):
                early_stopping(1-class_acc, classifier)                

            if early_stopping.early_stop:
                print('Early stop')
                break

    logger.info('End of training...')
    logger.removeHandler(file_handler)

if __name__ == '__main__':
    args = parse_args()
    main(args)
