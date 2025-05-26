from __future__ import print_function
#yeah
import os
import argparse
import torch
from utils import config
from dataset.loader import create_dataloader, create_dataset
import time
from dataset.reader import read_h_matrix_file_list
from PointSegBase.eval import test
from utils.io import IOStream, save_model
import utils.builder as builder
from utils.metric import MetricRecorder
import numpy as np 
from datetime import datetime

def get_datetime():
    now = datetime.now()
    filename = now.strftime("%m%d%H%M%S")
    return filename



def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp run.py checkpoints' + '/' + args.exp_name + '/' + 'run.py.backup')
    os.system('cp -r ' + args.config_dir +' checkpoints/' + args.exp_name + '/configs')

def process_device(batch_data, device=torch.device('cuda')):
    for k in ['points', 
            'feats', 
            'labels', 
            'extra_label', 
            'sparse_points', 
            'sparse_labels',
            'ori_labels',
            ]: 
        if k in batch_data.keys():
            if batch_data[k] is not None and not isinstance(batch_data[k], list):
                batch_data[k] = batch_data[k].to(device)
    return batch_data

def train(args, 
          io, 
          cfg, 
          h_matrices, 
          _,
          valid_recorder):
    device = torch.device("cuda" if args.cuda else "cpu")
    max_epoch = cfg.TRAIN.MAX_EPOCH

    loss_fun = builder.build_loss(cfg, h_matrices)
    model = builder.build_model(cfg, h_matrices, device)
    opt = builder.build_opt(cfg, model)
    scheduler = builder.build_scheduler(cfg, opt)
    
    train_dataset = create_dataset(cfg, set="TRAIN")
    train_loader = create_dataloader(train_dataset,
                                     batch_size=cfg.TRAIN.BATCH_SIZE,
                                     num_workers=cfg.TRAIN.MAX_WORKERS)
    validation_dataset = create_dataset(cfg, set="VALIDATION")
    validation_loader = create_dataloader(validation_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.TRAIN.MAX_WORKERS)
    io.cprint('length of train loader: %d' % (len(train_loader)))

    starttime = time.time()
    print('Training strat!')
    
    for epoch in range(max_epoch):
        ####################
        # Train
        ####################
        io.cprint('___________________epoch %d_____________________' % (epoch))
        train_loss = 0.0
        count = 0
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = process_device(batch_data)
            opt.zero_grad()
            torch.cuda.synchronize()
            seg_pred = model(batch_data)
            if 'sparse_labels' in batch_data.keys():
                target = batch_data['sparse_labels']
                pred = seg_pred['sparse_logits']
            else :
                target = batch_data['labels']
                pred = seg_pred['logits']

            loss = loss_fun(pred[0].F, target.F)
            loss.backward()
            opt.step()
            count += len(batch_data['points'])
            train_loss += loss.item()

            if batch_idx != 0 and batch_idx % 50 == 0:
                io.cprint('batch: %d, _loss: %f' % (batch_idx, loss))
        scheduler.step()
        
        io.cprint('train %d, loss: %f' % (epoch, train_loss * 1.0 / count))

        ####################
        # Test(validation)
        ####################
        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(validation_loader):
                    batch_data = process_device(batch_data)
                    opt.zero_grad()
                    seg_pred = model(batch_data)
                    if 'sparse_labels' in batch_data.keys():
                        target = batch_data['sparse_labels']
                        pred = seg_pred['sparse_logits']
                    else:
                        target = batch_data['labels']
                        pred = seg_pred['logits']
                        
                    valid_recorder.update(seg_pred, batch_data, batch_idx)
            maccs = valid_recorder.mean_acc
            mious = valid_recorder.miou
            accs = valid_recorder.overall_acc

            valid_recorder.renew()

            endtime = time.time()
            
            io.cprint('mean IoUs: {}'.format(["{:.2f}".format(miou*100) for miou in mious]))
            io.cprint('MAs: {}'.format(["{:.2f}".format(acc*100) for acc in maccs]))
            io.cprint('OAs: {}'.format(["{:.2f}".format(acc*100) for acc in accs]))
            
            io.cprint('Total Time: {} mins'.format(round(endtime - starttime, 2) / 60))

        if epoch % 1 == 0:
            save_model(model, cfg, args, 'model_%d' % (epoch))
    save_model(model, cfg, args, 'model_final')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_class', type=int, default=12, help='number of classes if the hierarchcal matrix is empty')
    parser.add_argument('--ignore_label', type=int, default=None, help='label to be ignored.')
    parser.add_argument('--config_dir', '-c', type=str, default='configs', help='config directory (default: ./configs)')
    parser.add_argument('--vis', type=bool, default=False, help='visualize the data')
    parser.add_argument('--fid_save', type=bool, default=False, help='save the fid')
    parser.add_argument('--fid_name', type=str, default='default.npy', help='name of the fid')
    args = parser.parse_args()
    abs_cfg_dir = os.path.abspath(os.path.join(__file__, "../" + args.config_dir))
    config.merge_cfg_from_dir(abs_cfg_dir)
    args.exp_name = args.exp_name + '_' +get_datetime()
    cfg = config.CONFIG

    if len(cfg.data.h_matrix_list_file) > 0:
        hierarchical_matrices = read_h_matrix_file_list(os.path.join(abs_cfg_dir, cfg.data.h_matrix_list_file)).hierarchical_matrices
    else:
        hierarchical_matrices = [np.eye(args.num_class)]
        
    cfg.data.data_list_file = os.path.join(abs_cfg_dir, cfg.data.data_list_file)
    _init_()
    name_dict = {True: "eval", False: ""}
    io = IOStream('checkpoints/' + args.exp_name + '/{}run.log'.format(name_dict[args.eval]))
    
    io.cprint('___________________CONFIG_____________________')
    io.cprint(str(cfg))
    io.cprint('___________________CONFIG_____________________')

    train_recorder = MetricRecorder(h_matrices=hierarchical_matrices, record_all=True, ignore_label=args.ignore_label)
    valid_recorder = MetricRecorder(h_matrices=hierarchical_matrices, record_all=True, ignore_label=args.ignore_label)

    args.cuda = torch.cuda.is_available()
    torch.manual_seed(cfg.DEVICES.SEED)

    if args.cuda:
        torch.cuda.set_device(cfg.DEVICES.GPU_ID[0])
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(cfg.DEVICES.SEED)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io, cfg, hierarchical_matrices, train_recorder, valid_recorder)
    else:
        test(args, io, cfg, hierarchical_matrices, valid_recorder)
