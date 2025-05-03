import os
import sys
sys.path.append(os.path.abspath('./'))
import argparse
import shutil
from glob import glob
import pickle
import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model import ScorePosNet3D
from scripts.sample_diffusion import sample_diffusion_ligand
from utils.data import PDBProtein
from datasets.pl_pair_dataset import parse_sdf_file

from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_vina import VinaDockingTask

from utils.eval import eval_result
import pickle



def pdb_to_pocket_data(protein_root, protein_fn, ligand_fn):
    pocket_dict = PDBProtein(os.path.join(protein_root, protein_fn)).to_dict_atom()
    ligand_dict = parse_sdf_file(os.path.join(protein_root, ligand_fn))
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict = torchify_dict(ligand_dict),
    )
    data.protein_filename = protein_fn
    data.ligand_filename = ligand_fn
    return data

def evaluate_data(result_path, sample_fn, args, pocket_fn, logger):
    results_fn_list = [os.path.join(result_path, sample_fn)]
    vina_path = os.path.join(result_path, 'vina')
    processed_count_path = result_path + "/results.pkl"
    processed_count = 0
    
    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        
        
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        all_pred_exp_traj = r['pred_exp_traj']
        all_pred_exp_score = r['pred_exp']
        all_pred_exp_atom_traj = r['pred_exp_atom_traj']
        # all_pred_exp_atom_traj = [np.zeros_like(all_pred_ligand_v[0]) for i in range(len(all_pred_exp_score))]
        num_samples += len(all_pred_ligand_pos)
       
        for sample_idx, (pred_pos, pred_v, pred_exp_score, pred_exp_atom_weight) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v, all_pred_exp_score, all_pred_exp_atom_traj)):
            
            processed_count += 1
            pred_pos, pred_v, pred_exp, pred_exp_atom_weight = pred_pos[args.eval_step], pred_v[args.eval_step], pred_exp_score, pred_exp_atom_weight[args.eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)

            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            
            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist
            
            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic, pred_exp_atom_weight)
                smiles = Chem.MolToSmiles(mol)
                print(smiles)    
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                continue
            n_recon_success += 1
            
            
            if '.' in smiles:
                continue
            n_complete += 1
            
            try:
                chem_results = scoring_func.get_chem(mol)
                
                vina_task = VinaDockingTask.from_generated_mol(
                    mol, pocket_fn, protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                
                if args.docking_mode == 'vina_dock':
                    docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                    vina_results['dock'] = docking_results
                    
                sdf_path = os.path.join(result_path, f"sdf")
                os.makedirs(sdf_path, exist_ok=True)
                writer = Chem.SDWriter(os.path.join(sdf_path, f'res_{sample_idx}.sdf'))
                writer.write(mol)
                writer.close()
                
                n_eval_success += 1

            except:
                if args.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')  
                continue
            
            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)

            mol_params = {
                'mol': mol,
                'smiles': smiles,
                'ligand_filename': r['data'].ligand_filename,
                'pred_pos': pred_pos,
                'pred_v': pred_v,
                'chem_results': chem_results,
                'vina': vina_results,
                'pred_exp': pred_exp,
                'atom_exp': {
                    atom.GetIdx(): float(atom.GetProp('_affinity_weight')) for atom in mol.GetAtoms()
                },
            }

            results.append(mol_params)      
        
        pickle.dump(results, open(vina_path + str(example_idx) + '.pkl', 'wb'))

        logger.info('This eval sample is %s' % f'{example_idx}')    
        eval_result(all_mol_stable, all_atom_stable, n_recon_success, n_eval_success, n_complete, num_samples, all_n_atom, all_bond_dist,
                success_pair_dist, success_atom_types, results , result_path, logger, args)
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protein_root', type=str, default='./test_complex/other_pdb/')
    parser.add_argument('--protein_fn', type=str, default='5liu_X_rec.pdb')
    parser.add_argument('--ligand_fn', type=str, default='5liu_X_rec_4gq0_qap_lig_tt_min_0.sdf')
    parser.add_argument('--config', type=str, default='./configs/sampling.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--re_sample', type=bool, default=True)

    parser.add_argument('--guide_mode', type=str, default='joint', choices=['joint', 'vina', 'valuenet', 'wo'])  
    parser.add_argument('--type_grad_weight', type=float, default=100)
    parser.add_argument('--pos_grad_weight', type=float, default=25)
    parser.add_argument('--result_path', type=str, default='./test_complex')
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, default='vina_dock', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)


    args = parser.parse_args()
    result_path = args.result_path
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    logger = misc.get_logger('sampling', log_dir=result_path)

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)
    
    # Load checkpoint
    ckpt = torch.load(config.model['joint_ckpt'], map_location=args.device)
    value_ckpt = None
    
    logger.info(f"Training Config: {ckpt['config']}")
    logger.info(f"args: {args}")
    
    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    
    if value_ckpt is not None:
        # value model
        value_model = ScorePosNet3D(
            value_ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim
        ).to(args.device)
        value_model.load_state_dict(value_ckpt['model'])
    else:
        value_model = None
    

    # Load pocket
    pocket_fn = args.protein_fn.split('.')[0] + "_pocket10.pdb"
    
    data = pdb_to_pocket_data(args.protein_root, pocket_fn, args.ligand_fn)
    data = transform(data)
 
    sample_fn = f'result_{os.path.basename(args.protein_fn)[:-4]}.pt'
    if args.re_sample == True:
        pred_pos, pred_v, pred_exp, pred_pos_traj, pred_v_traj, pred_exp_traj, pred_v0_traj, pred_vt_traj, pred_exp_atom_traj, time_list = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms,
            guide_mode=args.guide_mode,
            value_model=value_model,
            type_grad_weight=args.type_grad_weight,
            pos_grad_weight=args.pos_grad_weight
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_exp': pred_exp,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'pred_exp_traj': pred_exp_traj,
            'pred_exp_atom_traj': pred_exp_atom_traj,
            'time': time_list
        }
        torch.save(result, os.path.join(result_path, sample_fn))
        logger.info(f'Sample done! Result saved to {result_path}/{sample_fn}')

    evaluate_data(result_path, sample_fn, args, pocket_fn, logger)

if __name__ == '__main__':
    main()