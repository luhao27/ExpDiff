import argparse
import os
import sys
sys.path.append(os.path.abspath('./'))

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask

from utils.eval import eval_result
import pickle


def main():
    parser = argparse.ArgumentParser()
    # WARN: important turn on when evaluate pdbbind related proteins
    ################
    parser.add_argument('--eval_pdbbind', action='store_true')
    ################
    
    parser.add_argument('--sample_path', type=str, default='./logs_diffusion/762000_4/')
    parser.add_argument('--verbose', type=eval, default=True)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set/')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, default='vina_dock', choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--start_it', type=int, default=0)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')

    num_samples = 0
    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    num_samples_single = 0
    all_mol_stable_single, all_atom_stable_single, all_n_atom_single = 0, 0, 0
    n_recon_success_single, n_eval_success_single, n_complete_single = 0, 0, 0
    results = []
    all_pair_dist_single, all_bond_dist_single = [], []
    all_atom_types_single = Counter()
    success_pair_dist_single, success_atom_types_single = [], Counter()

    if not os.path.exists("results"):
        os.mkdir("results")

    processed_count = -1
    start = args.start_it
    vina_path = "results/vina"
    processed_count_path = "results/processed_count" + str(start-1) + ".pkl"
        
    if start > 0:
        
        processed_metric = pickle.load(open(processed_count_path, "rb"))
        all_mol_stable = processed_metric["all_mol_stable"]
        all_atom_stable = processed_metric["all_atom_stable"]
        all_n_atom = processed_metric["all_n_atom"]
        n_recon_success = processed_metric["n_recon_success"]
        n_eval_success = processed_metric["n_eval_success"]
        n_complete = processed_metric["n_complete"]
        num_samples = processed_metric["num_samples"]
        all_pair_dist = processed_metric["all_pair_dist"]
        all_bond_dist = processed_metric["all_bond_dist"]
        all_atom_types = processed_metric["all_atom_types"]
        success_pair_dist = processed_metric["success_pair_dist"]
        success_atom_types = processed_metric["success_atom_types"]
        processed_count = processed_metric["processed_count"]
        
        logger.info(f"start from {start}")
        results_fn_list = results_fn_list[start:]
    
    # Evaluate 
    for example_idx, r_name in enumerate(tqdm(results_fn_list, desc='Eval')):
        logger.info(f'Evaluate {r_name}**********************')
        
        r = torch.load(r_name)  # ['data', 'pred_ligand_pos', 'pred_ligand_v', 'pred_ligand_pos_traj', 'pred_ligand_v_traj']
        all_pred_ligand_pos = r['pred_ligand_pos_traj']  # [num_samples, num_steps, num_atoms, 3]
        all_pred_ligand_v = r['pred_ligand_v_traj']
        all_pred_exp_traj = r['pred_exp_traj']
        all_pred_exp_score = r['pred_exp']
        all_pred_exp_atom_traj = r['pred_exp_atom_traj']
        # all_pred_exp_atom_traj = [np.zeros_like(all_pred_ligand_v[0]) for i in range(len(all_pred_exp_score))]
        num_samples += len(all_pred_ligand_pos)
        
        num_samples_single += len(all_pred_ligand_pos)

        for sample_idx, (pred_pos, pred_v, pred_exp_score, pred_exp_atom_weight) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v, all_pred_exp_score, all_pred_exp_atom_traj)):
            processed_count += 1

            pred_pos, pred_v, pred_exp, pred_exp_atom_weight = pred_pos[args.eval_step], pred_v[args.eval_step], pred_exp_score, pred_exp_atom_weight[args.eval_step]

            # stability check
            pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
            all_atom_types += Counter(pred_atom_type)
            
            all_atom_types_single += Counter(pred_atom_type)

            r_stable = analyze.check_stability(pred_pos, pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            
            
            all_mol_stable_single += r_stable[0]
            all_atom_stable_single += r_stable[1]
            all_n_atom_single += r_stable[2]

            pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
            all_pair_dist += pair_dist
            
            all_pair_dist_single += pair_dist

            # reconstruction
            try:
                pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
                mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic, pred_exp_atom_weight)
                smiles = Chem.MolToSmiles(mol)
                #print(smiles)
                
                
            except reconstruct.MolReconsError:
                if args.verbose:
                    logger.warning('Reconstruct failed %s' % f'{example_idx+start}_{sample_idx}')
                continue
            n_recon_success += 1
            
            n_recon_success_single += 1
            
            if '.' in smiles:
                continue
            n_complete += 1
            
            n_complete_single += 1
            
            # chemical and docking check
            try:
                chem_results = scoring_func.get_chem(mol)
                
                logger.info('eval other dataset')
                protein_fn = os.path.join(
                            os.path.dirname(r['data'].ligand_filename),
                            os.path.basename(r['data'].ligand_filename)[:10] + '.pdb'
                        )
                
                print("Now ID : ", example_idx+start)
                vina_task = VinaDockingTask.from_generated_mol(
                    mol, protein_fn, protein_root=args.protein_root)
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                vina_results = {
                        'score_only': score_only_results,
                        'minimize': minimize_results
                    }
                if args.docking_mode == 'vina_dock':
                    docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                    vina_results['dock'] = docking_results
                    
                sdf_path = os.path.join(result_path, f"sdf_{r_name[:-3].split('_')[-1]}")
                os.makedirs(sdf_path, exist_ok=True)
                writer = Chem.SDWriter(os.path.join(sdf_path, f'res_{sample_idx}.sdf'))
                writer.write(mol)
                writer.close()
                
                n_eval_success += 1

                n_eval_success_single += 1
            except:
                if args.verbose:
                    logger.warning('Evaluation failed for %s' % f'{example_idx+start}_{sample_idx}')
                    
                continue
            
            # now we only consider complete molecules as success
            bond_dist = eval_bond_length.bond_distance_from_mol(mol)
            all_bond_dist += bond_dist
            
            all_bond_dist_single += bond_dist

            success_pair_dist += pair_dist
            success_atom_types += Counter(pred_atom_type)
            
            success_pair_dist_single += pair_dist
            success_atom_types_single += Counter(pred_atom_type)
            

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
        
        results_count={
            'all_mol_stable': all_mol_stable,
            'all_atom_stable': all_atom_stable,
            'all_atom_types': all_atom_types,
            'n_recon_success': n_recon_success,
            'n_eval_success': n_eval_success,
            'n_complete': n_complete,
            'num_samples': num_samples,
            'all_n_atom': all_n_atom,
            'all_pair_dist': all_pair_dist,
            'all_bond_dist': all_bond_dist,
            'success_pair_dist': success_pair_dist,
            'success_atom_types': success_atom_types,
            "processed_count": processed_count
        }

        processed_count_path = "results/processed_count" + str(example_idx + start) + ".pkl"
        pickle.dump(results_count, open(processed_count_path, 'wb'), )
        pickle.dump(results, open(vina_path + str(example_idx + start) + '.pkl', 'wb'))
        
        logger.info('This eval sample is %s' % f'{example_idx+start}')    
        eval_result(all_mol_stable_single, all_atom_stable_single, n_recon_success_single, n_eval_success_single, n_complete_single, num_samples_single, all_n_atom_single, all_bond_dist_single,
                success_pair_dist_single, success_atom_types_single, results , result_path, logger, args)
        results = []
        
        num_samples_single = 0
        all_mol_stable_single, all_atom_stable_single, all_n_atom_single = 0, 0, 0
        n_recon_success_single, n_eval_success_single, n_complete_single = 0, 0, 0
        all_pair_dist_single, all_bond_dist_single = [], []
        all_atom_types_single = Counter()
        success_pair_dist_single, success_atom_types_single = [], Counter() 

    logger.info('***************This is eval all the samples!***************')

    with open(processed_count_path, 'rb') as f:
        all_count = pickle.load(f)
        all_mol_stable, all_atom_stable, all_n_atom = all_count['all_mol_stable'], all_count['all_atom_stable'], all_count['all_n_atom']
        n_recon_success, n_eval_success, n_complete = all_count['n_recon_success'], all_count['n_eval_success'], all_count['n_complete']
        num_samples = all_count['num_samples']
        all_pair_dist, all_bond_dist = all_count['all_pair_dist'], all_count['all_bond_dist']
        success_pair_dist, success_atom_types = all_count['success_pair_dist'], all_count['success_atom_types']
    
    
    my_fn_list = glob(os.path.join("./results/", 'vina*.pkl'))

    # print(my_fn_list)
    # print(len(my_fn_list))
    
    for idx, fn in enumerate(my_fn_list):
        with open(fn, 'rb') as f:
            results += pickle.load(f)
    
    eval_result(all_mol_stable, all_atom_stable, n_recon_success, n_eval_success, n_complete, num_samples, all_n_atom, all_bond_dist,
                success_pair_dist, success_atom_types, results , result_path, logger, args)

if __name__ == '__main__':
    main()