"""
Usage:
python eval_ssr.py --data_dir {data_dir} --checkpoint {checkpoint_dir} --fold {fold_number} --sv_dir {save_folder}
"""
import sys
import argparse
from turtle import pos
import numpy as np
import os
import torch
from tqdm import tqdm
import sys
import importlib
import torch.nn as nn
import utils.provider as provider
from utils.dataloader import DataLoaderForHitOnComplex,collate_fn_hit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'model'))
sys.path.append(os.path.join(ROOT_DIR,'utils'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('evaluate success rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training [default: 16]')
    parser.add_argument('--model', default='pmlp_igm', help='model name [default: pmlp_igm]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--npoint', type=int, default=1000, help='Point cloud total number [default: 1000]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer for training [default: Adam]')
    parser.add_argument('--data_dir', type=str, default='dockground61_1000', help='Data root')
    parser.add_argument('--sv_folder', type=str, default='ssr_sv', help='Suceess rate save root')
    parser.add_argument('--sv_dir', type=str, default='dockground61_1000', help='Suceess rate save dir')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: True]')
    parser.add_argument('--fold', type=int, default=-1, help='K Fold Number [default: -1]')
    parser.add_argument('--use_res', type=bool, default=True, help='Use residual features')
    parser.add_argument('--seed', type=int, default=1024, help='random seed')
    parser.add_argument('--checkpoint', type=str, default='checkpoint',help='checkpoints path')
    return parser.parse_args()

def main():
    args = parse_args()
    provider.set_seed(args.seed)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    '''DATA LOADING'''
    print('Load dataset ...')
    DATA_PATH = os.path.join(ROOT_DIR,'data')
    DATA_PATH = os.path.join(DATA_PATH,args.data_dir)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    CKPT_PATH = os.path.join(ROOT_DIR,'checkpoint')

    RESULT_PATH = os.path.join(ROOT_DIR,args.sv_folder)
    if not os.path.exists(RESULT_PATH):
        os.mkdir(RESULT_PATH)

    sv_path = os.path.join(RESULT_PATH,args.sv_dir)
    if not os.path.exists(sv_path):
        os.mkdir(sv_path)

    classifiers = []
    if args.fold==-1:
        for fold_id in range(4):
            savepath = os.path.join(CKPT_PATH,'fold{}'.format(fold_id+1),'best_model.pth') 
            model_CKPT = torch.load(savepath)
            '''MODEL LOADING'''
            classifier = MODEL.PMLP_IGM().cuda()
            classifier.load_state_dict(model_CKPT['model_state_dict'])
            classifier.eval()
            classifiers.append(classifier)
    else:
        savepath = os.path.join(CKPT_PATH,'fold{}'.format(args.fold),'best_model.pth') 
        model_CKPT = torch.load(savepath)
        '''MODEL LOADING'''
        classifier = MODEL.PMLP_IGM().cuda()
        classifier.load_state_dict(model_CKPT['model_state_dict'])
        classifier.eval()
        classifiers.append(classifier)

            
    test_complex_names = os.listdir(DATA_PATH)
    test_complex_names = [x for x in test_complex_names if os.path.isdir(os.path.join(DATA_PATH,x))]

    '''EVAL ALL TEST COMPLEXES'''
    for test_complex_name in test_complex_names:
        COMPlEX_DATA_PATH = os.path.join(DATA_PATH,test_complex_name)
        HITDATA = DataLoaderForHitOnComplex(COMPlEX_DATA_PATH, npoint=args.npoint,use_res=args.use_res)
        result_dir = os.path.join(sv_path,'{}.txt'.format(test_complex_name))
        hitDataLoader = torch.utils.data.DataLoader(HITDATA,batch_size=args.batch_size, shuffle=False,collate_fn=collate_fn_hit)
        pred_list = [[] for _ in range(len(classifiers))]
        file_list = []
        with torch.no_grad():
            for j, data in tqdm(enumerate(hitDataLoader), total=len(hitDataLoader)):
                points, target, npoints,ps = data
                points = torch.Tensor(points)
                target = torch.Tensor(target)
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                for index,classifier in enumerate(classifiers):
                    classifier.eval()
                    pred = classifier(points)
                    pred_list[index]+=pred.cpu().numpy().tolist()
                file_list+=ps
        pred_list = np.array(pred_list)
        pred_list = np.mean(pred_list, axis=0)
        file_list = np.array(file_list)
        sort = np.argsort(pred_list)
        with open(result_dir,'w+') as f:
            file_sort = file_list[sort][::-1]
            pred_sort = pred_list[sort][::-1]
            for i in range(pred_list.shape[0]):
                file_name = str(file_sort[i]).split(os.sep)[-1]
                f.write('{}'.format(file_name)+'\t'+str(pred_sort[i])+'\n')

if __name__ == '__main__':
    main()
