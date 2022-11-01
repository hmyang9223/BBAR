from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# rdkit Descriptors
mol_desc_list = {
    'mw': Descriptors.ExactMolWt,
    'logp': Descriptors.MolLogP,
    'tpsa': Descriptors.TPSA,
    'qed': Descriptors.qed,
}

property_list = ['mw', 'tpsa', 'logp', 'qed']
floating_point = {
    'mw': 2,
    'tpsa': 3,
    'logp': 5,
    'qed': 5,
}

with open('./all.txt') as f :
    lines = f.readlines()

with open('./property.csv', 'w') as w :
    w.write(lines[0].strip() + ',' + ','.join(property_list) + '\n')
    for l in tqdm(lines[1:]) :
        smiles = l.split(',')[1]
        properties = []
        for key in property_list :
            if key in smiles_desc_list :
                value = smiles_desc_list[key](smiles)
                properties.append(f'{value:.{floating_point[key]}f}')
            else :
                mol = Chem.MolFromSmiles(smiles)
                value = mol_desc_list[key](mol)
                properties.append(f'{value:.{floating_point[key]}f}')
        w.write(f'{l.strip()},{",".join(properties)}\n')
