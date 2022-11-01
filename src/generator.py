import os
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Dataset, DataLoader
from torch import BoolTensor
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol, Descriptors
from typing import Dict, Union, Any, List, Optional, Callable

from .model import BlockConnectionPredictor
from .utils import brics, feature

RDLogger.DisableLog('rdApp.*')

class MoleculeBuilder() :
    def __init__(self, cfg, filter_fn : Optional[Callable] = None) :
        self.cfg = cfg
        self.max_iteration = cfg.max_iteration

        if filter_fn :
            self.filter_fn = filter_fn
        else :
            self.filter_fn = lambda x : True

        library_builtin_model_path = self.cfg.get('library_builtin_model_path', None)
        # Load Model & Library
        if library_builtin_model_path is not None and os.path.exists(library_builtin_model_path) :
            self.model, self.library = self.load_library_builtin_model(library_builtin_model_path)
        else :
            self.model, self.library = self.load_model(cfg.model_path), self.load_library(cfg.library_path)
            self.embed_model_with_library(library_builtin_model_path)

        # Setup Library Information
        library_freq = torch.from_numpy(self.library.freq)
        self.library_freq = library_freq / library_freq.sum()
        self.library_freq_weighted = library_freq ** cfg.alpha
        self.library_allow_brics_list = self.get_library_allow_brics_list()
        self.n_lib_sample = min(len(self.library), cfg.n_library_sample)

        # Setup after self.setup()
        self.target_properties = self.model.cond_keys
        self.cond = None

    def setup(self, condition) :
        self.cond = self.model.get_cond(condition).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        scaffold: Union[Mol, str, None],
        ) :
        assert self.cond is not None, \
            'MoleculeBuilder is not setup. Please call MoleculeBuilder.setup(condition: Dict[property_name, target_value])\n' + \
            f'required property: {list(self.target_properties)}' 

        if scaffold is not None :
            initial_step = 0
            if isinstance(scaffold, str) :
                scaffold = Chem.MolFromSmiles(scaffold)
            if len(brics.BRICSCompose.get_possible_brics_labels(scaffold)) == 0 :
                return None
        else :
            initial_step = 1
            scaffold = self.get_random_scaffold(max_try = 30)
            if scaffold is None :
                return None
        
        core_mol = scaffold
        for _ in range(initial_step, self.max_iteration) :
            # Graph Embedding
            h_core_0, adj_core = self.get_core_feature(core_mol)
            h_core_1, Z_core = self.model.graph_embedding_core(h_core_0, adj_core, self.cond)

            # Predict Termination
            termination = self.predict_termination(Z_core)
            if termination :
                return core_mol

            # Sampling building blocks
            prob_dist_block = self.get_prob_dist_block(core_mol, Z_core, num_use_blocks=self.n_lib_sample)
                                                                                                    # (N_lib)
            compose_success = False
            for _ in range(5) :
                if not torch.is_nonzero(prob_dist_block.sum()):
                    return None

                # Sample block
                block_idx = self.sample_block(prob_dist_block)
                block_mol = self.library.get_mol(block_idx)
                Z_block = self.model.Z_lib[block_idx].unsqueeze(0)

                # Predict Index
                atom_idx = self.predict_atom_idx(core_mol, block_mol, Z_core, Z_block, h_core_0, h_core_1, adj_core)
                if atom_idx is None :
                    prob_dist_block[block_idx] = 0
                    continue

                # Compose
                composed_mol = self.compose(core_mol, block_mol, atom_idx)
                if composed_mol is not None :
                    compose_success = True
                    break

            if compose_success : 
                core_mol = composed_mol
            else :
                return None

        return None 

    __call__ = generate

    def get_random_scaffold(self, max_try = 20) :
        block_idxs = torch.multinomial(self.library_freq_weighted, max_try).tolist()
        for block_idx in block_idxs :
            scaffold = self.library.get_mol(block_idx)
            scaffold = brics.preprocess.remove_brics_label(scaffold, returnMol = True)
            valid = (len(brics.BRICSCompose.get_possible_brics_labels(scaffold)) > 0)
            if valid :
                return scaffold
        return None

    def get_core_feature(self, core_mol) :
        h = feature.get_atom_features(core_mol, brics=False).unsqueeze(0)
        adj = feature.get_adj(core_mol).unsqueeze(0)
        return h, adj

    def predict_termination(self, Z_core) :
        p_term = self.model.predict_termination(Z_core)
        termination = Bernoulli(probs=p_term).sample().bool().item()
        return termination

    def get_prob_dist_block(self, core_mol, Z_core, num_use_blocks: int) :
        brics_labels = brics.BRICSCompose.get_possible_brics_labels(core_mol)
        block_mask = torch.zeros((len(self.library),), dtype=torch.bool)
        for brics_label in brics_labels :
            block_mask += self.library_mask[int(brics_label)]
        
        use_blocks = torch.arange(len(self.library))[block_mask]
        if use_blocks.size(0) == 0 :
            return None

        if num_use_blocks < len(use_blocks) :
            freq = self.library_freq_weighted[block_mask]
            idxs = torch.multinomial(freq, num_use_blocks, False)
            use_blocks = use_blocks[idxs]

        prob_dist_use_block = self.model.predict_frag_id(Z_core, probs=True, use_lib=use_blocks.unsqueeze(0)).squeeze(0)
        prob_dist_use_block.nan_to_num_(0)
        prob_dist_block = torch.zeros((len(self.library),), dtype=torch.float)
        prob_dist_block[use_blocks] = prob_dist_use_block
        return prob_dist_block
    
    def sample_block(self, prob_dist_block) :
        block_idx = Categorical(probs = prob_dist_block).sample().item()
        return block_idx

    def predict_atom_idx(self, core_mol, block_mol, Z_core, Z_block, h_core_0, h_core_1, adj_core) :
        prob_dist_idx = self.model.predict_idx(
                h_core_0, adj_core, h_core_1, Z_core, Z_block, probs=True).squeeze(0)
        # Masking
        idx_mask = torch.ones((core_mol.GetNumAtoms(),), dtype=torch.bool)
        idxs = [idx for idx, _ in brics.BRICSCompose.get_possible_indexs(core_mol, block_mol)]
        idx_mask[idxs] = False
        prob_dist_idx.masked_fill_(idx_mask, 0)

        # Sampling
        if not torch.is_nonzero(prob_dist_idx.sum()) :
            return None
        else :
            # Choose Index
            atom_idx = Categorical(probs = prob_dist_idx).sample().item()
            return atom_idx

    def compose(self, core_mol, block_mol, atom_idx) :
        try :
            composed_mol = brics.BRICSCompose.compose(core_mol, block_mol, atom_idx, 0, 
                                                returnMol=True, force=self.cfg.compose_force)
        except :
            composed_mol = None
        return composed_mol

    def load_model(self, model_path) :
        model = BlockConnectionPredictor.load(model_path, map_location = 'cpu')
        model.eval()
        return model

    def load_library(self, library_path) :
        return brics.BRICSLibrary(library_path, save_mol = True)

    def load_library_builtin_model(self, library_builtin_model_path) :
        print(f"Load {library_builtin_model_path}")
        checkpoint = torch.load(library_builtin_model_path, map_location = 'cpu')
        model = BlockConnectionPredictor(checkpoint['config'], checkpoint['cond_scale'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.set_Z_lib(checkpoint['Z_lib'])
        model.eval()
        
        library = brics.BRICSLibrary(smiles_list = checkpoint['library_smiles'], freq_list = checkpoint['library_freq'], save_mol = True)
        return model, library

    def embed_model_with_library(self, library_builtin_model_path) :
        print("Setup Library Building Blocks' Graph Vectors")
        with torch.no_grad() :
            h, adj = self.load_library_feature()
            Z_lib = self.model.graph_embedding_frag(h, adj)
            self.model.Z_lib = Z_lib
        print("Finish")
        if library_builtin_model_path is not None :
            print(f"Create Local File ({library_builtin_model_path})")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.model._cfg,
                'cond_scale': self.model.cond_scale,
                'library_smiles': self.library.smiles,
                'library_freq': self.library.freq,
                'Z_lib': Z_lib
            }, library_builtin_model_path)
        else :
            print("You can save graph vectors by setting generator_config.library_builtin_model_path")

        
    def load_library_feature(self) :
        # Load node feature / adjacency matrix
        library_feature_path = os.path.splitext(self.cfg.library_path)[0] + '.npz'
        if os.path.exists(library_feature_path) :
            f = np.load(library_feature_path)
            h = torch.from_numpy(f['h']).float()
            adj = torch.from_numpy(f['adj']).bool()
            f.close()
        else:
            max_atoms = max([m.GetNumAtoms() for m in self.library.mol])
            h, adj = [], []
            for m in self.library.mol :
                h.append(feature.get_atom_features(m, max_atoms, True))
                adj.append(feature.get_adj(m, max_atoms))

            h = torch.stack(h)
            adj = torch.stack(adj)
            np.savez(library_feature_path, h=h.numpy(), adj=adj.numpy().astype('?'), \
                     freq=self.library.freq)
            h = h.float()
            adj = adj.bool()
        return h, adj

    def get_library_allow_brics_list(self) :
        library_mask = torch.zeros((len(self.library), 17), dtype=torch.bool)
        for i, brics_label in enumerate(self.library.brics_label_list) : 
            allow_brics_label_list = brics.constant.BRICS_ENV_INT[brics_label]
            for allow_brics_label in allow_brics_label_list :
                library_mask[i, allow_brics_label] = True
        self.library_mask = library_mask.T  # (self.library, 17)
    
