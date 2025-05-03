import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd
import torch
import torch.utils.tensorboard
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from scipy import stats
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
import sys
sys.path.append(os.path.abspath('./'))
import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
import logging

from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        logger = logging.getLogger(__name__)
        logger.info(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)

def train(it, model, optimizer, config, train_iterator, args, logger):
    model.train()
    optimizer.zero_grad()
    for _ in range(config.train.n_acc_batch):
        batch = next(train_iterator).to(args.device)
            
        embedding_feature = []
        for i in range(len(batch.protein_filename)):
            name = batch.protein_filename[i] + '_' + batch.protein_molecule_name[i]
            global export_embedding
            try:
                embedding_feature.append(export_embedding[name].to(args.device))
            except:
                embedding_feature.append(np.array([]))
         
        protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
        gt_protein_pos = batch.protein_pos + protein_noise
        results = model.get_diffusion_loss(
            protein_pos=gt_protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            affinity=batch.affinity.float(),
            batch_protein=batch.protein_element_batch,

            ligand_pos=batch.ligand_pos,
            ligand_v=batch.ligand_atom_feature_full,
            batch_ligand=batch.ligand_element_batch,
                
            embedding_feature=embedding_feature,
            )
        if args.value_only:
            results['loss'] = results['loss_exp']
                
        loss, loss_pos, loss_v, loss_exp = results['loss'], results['loss_pos'], results['loss_v'], results['loss_exp']
        loss = loss / config.train.n_acc_batch
        loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
    optimizer.step()

    if it % args.train_report_iter == 0:
        logger.info(
                '[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | exp %.6f) | Lr: %.6f | Grad Norm: %.6f' % (
                it, loss, loss_pos, loss_v, loss_exp, optimizer.param_groups[0]['lr'], orig_grad_norm
            )
        )

def validate(it, model, val_loader, args,config, scheduler, logger):
    # fix time steps
    sum_loss, sum_loss_pos, sum_loss_v, sum_loss_exp, sum_n = 0, 0, 0, 0, 0
    all_pred_v, all_true_v , all_pred_exp, all_true_exp= [], [], [], []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(val_loader, desc='Validate'):
                
            embedding_feature = []
            for i in range(len(batch.protein_filename)):
                name = batch.protein_filename[i] + '_' + batch.protein_molecule_name[i]
                global export_embedding
                try:
                    # print(export_embedding[name])
                    embedding_feature.append(export_embedding[name].to(args.device))
                except:
                    embedding_feature.append(np.array([]))
                
            batch = batch.to(args.device)
            batch_size = batch.num_graphs
            for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                time_step = torch.tensor([t] * batch_size).to(args.device)
                results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        affinity=batch.affinity.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        batch_ligand=batch.ligand_element_batch,
                        time_step=time_step,

                        embedding_feature=embedding_feature,
                )
                loss, loss_pos, loss_v, loss_exp, pred_exp = results['loss'], results['loss_pos'], results['loss_v'], results['loss_exp'], results['pred_exp']

                sum_loss += float(loss) * batch_size
                sum_loss_pos += float(loss_pos) * batch_size
                sum_loss_v += float(loss_v) * batch_size
                sum_loss_exp += float(loss_exp) * batch_size
                sum_n += batch_size
                all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                all_pred_exp.append(pred_exp.detach().cpu().numpy())
                all_true_exp.append(batch.affinity.float().detach().cpu().numpy())

    avg_loss = sum_loss / sum_n
    avg_loss_pos = sum_loss_pos / sum_n
    avg_loss_v = sum_loss_v / sum_n
    avg_loss_exp = sum_loss_exp / sum_n
    atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)
        
    if config.train.scheduler.type == 'plateau':
        scheduler.step(avg_loss)
    elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
    else:
        scheduler.step()

    logger.info(
        '[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f e-3 | Loss exp %.6f e-3 | Avg atom auroc %.6f' % (
                it, avg_loss, avg_loss_pos, avg_loss_v * 1000, avg_loss_exp * 1000, atom_auroc
            )
    )
        
    if args.value_only:
        return avg_loss_exp
        
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--value_only', action='store_true')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--embedding_file', type=str, default='./data/ligand_embedding.pt')
    parser.add_argument('--start_validation', type=int, default=500000)
    parser.add_argument('--start_it', type=int, default=0)
   
    args = parser.parse_args()
    # load ckpt
    if args.ckpt:
        print(f'loading {args.ckpt}...')
        ckpt = torch.load(args.ckpt, map_location=args.device)
        config = ckpt['config']
        # config = misc.load_config(args.config)
    else:
        # Load configs
        config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)
    
    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    # writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)

    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
        trans.NormalizeVina("pl")
    ]

    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    dataset, subsets = get_dataset(
        config=config.data,
        transform=transform,
    )
    train_set, val_set, test_set = subsets['train'], subsets['test'], []

    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)} Test: {len(test_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)


    # Model
    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    
    logger.info(f'Building model. trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')
    
    # print(model)
    logger.info(f'protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}')
    # logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)
    
    start_it = args.start_it
    if args.ckpt:
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_it = ckpt['iteration']
        # 
        best_loss, best_iter = 0.729363, 762000

    try:
        if args.ckpt:
            logger.info(f"USing defined best_loss: {best_loss}, best_inter: {best_iter}")
        else:
            best_loss, best_iter = None, None
        
        export_embedding = torch.load(args.embedding_file)
        logger.info(f"Number of pretrained ligand embedding: {len(export_embedding.keys())}")
        
        for it in range(start_it, config.train.max_iters):
            # with torch.autograd.detect_anomaly():
            train(it, model, optimizer, config, train_iterator, args, logger)
            
            if (it % config.train.val_freq == 0 or it == config.train.max_iters) and it >args.start_validation:
                val_loss = validate(it, model, val_loader, args,config, scheduler, logger)
                
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
    except KeyboardInterrupt:
        logger.info('Terminating...')
        
        
if __name__ == '__main__':
    main()
