import logging
import pathlib
from omegaconf import OmegaConf

import sys
sys.path.append(".")
sys.path.append("..")

from bbar.trainer import Trainer

from options.train_options import Train_ArgParser
from utils.logger import setup_logger
from uils.seed import set_seed

def setup_trainer(args) :
    # Setup Trainer
    trainer_cfg = OmegaConf.load(args.trainer_config)
    model_cfg = OmegaConf.load(args.model_config)
    data_cfg = OmegaConf.load(args.data_config)
    OmegaConf.resolve(trainer_cfg)
    OmegaConf.resolve(model_cfg)
    OmegaConf.resolve(data_cfg)
    properties: list = args.property

    # Save Config
    OmegaConf.save({
        'property': properties,
        'model_config': model_cfg,
        'trainer_config': trainer_cfg,
        'data_config': data_cfg,
    }, config_path)

    # Print Config
    logging.info(
            'Training Information\n' +
            'Argument\n' + '\n'.join([f'{arg}:\t{getattr(args,arg)}' for arg in vars(args)]) + '\n\n' +
            'Trainer Config\n' + OmegaConf.to_yaml(trainer_cfg) + '\n' +
            'Data Config\n' + OmegaConf.to_yaml(data_cfg)
    )

    trainer = Trainer(trainer_cfg, model_cfg, data_cfg, properties, checkpoint_dir)
    return trainer

def main() : 
    set_seed(0)
    parser = Train_ArgParser()
    args = parser.parse_args()
    
    # Setup Logger
    run_dir: pathlib.Path = setup_logger(args.exp_dir, args.name)
    config_path = run_dir / 'config.yaml'
    checkpoint_dir = run_dir / 'checkpoint'

    trainer = setup_trainer(args)
    trainer.fit()

if __name__ == '__main__' :
    main()
