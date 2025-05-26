from __future__ import print_function
import torch
from dataset.loader import create_dataloader, create_dataset
import utils.builder as builder
from utils.io import load_model
import numpy as np


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

def test(args, 
          io, 
          cfg, 
          h_matrices, 
          valid_recorder):
    seg_vis = args.vis
    device = torch.device("cuda" if args.cuda else "cpu")
    _model = builder.build_model(cfg, h_matrices, device)
    model = load_model(args, cfg, _model)
    
    validation_dataset = create_dataset(cfg, set="TEST")
    validation_loader = create_dataloader(validation_dataset, 
                                          batch_size=cfg.TRAIN.BATCH_SIZE, 
                                          num_workers=cfg.TRAIN.MAX_WORKERS)

    io.cprint('length of train loader: %d' % (len(validation_loader)))

    print('Testing strat!')

    batch_pooling_list = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(validation_loader):
            batch_data = process_device(batch_data)
            seg_pred = model(batch_data)
            print("seg_pred: ", seg_pred)
            if args.fid_save:
                batch_pooling_list.append(seg_pred['pooling'])
            valid_recorder.update(seg_pred, batch_data, batch_idx, seg_vis=seg_vis)

        if args.fid_save:
            concatenated_pooling = torch.cat(batch_pooling_list, dim=0).cpu().numpy()
            print(concatenated_pooling.shape)
            file_path = args.fid_name
            np.save(file_path, concatenated_pooling)

    maccs = valid_recorder.mean_acc
    mious = valid_recorder.miou
    accs = valid_recorder.overall_acc

    valid_recorder.renew()
    
    io.cprint('mean IoUs: {}'.format(["{:.2f}".format(miou*100) for miou in mious]))
    io.cprint('MAs: {}'.format(["{:.2f}".format(acc*100) for acc in maccs]))
    io.cprint('OAs: {}'.format(["{:.2f}".format(acc*100) for acc in accs]))
