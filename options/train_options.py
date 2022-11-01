import configargparse

class Train_ArgParser(configargparse.ArgParser) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)

        # Required Parameters
        required_args = self.add_argument_group('train information')
        required_args.add_argument('--name', type=str, help='job name', required=True)
        required_args.add_argument('--exp_dir', type=str, help='path of experiment directory', default='./result/')
        required_args.add_argument('-p', '--property', type=str, nargs='+', help='property list')

        # Modules
        module_args = self.add_argument_group('module config(required)')
        module_args.add_argument('--trainer_config', type=str, default='./config/trainer.yaml')
        module_args.add_argument('--model_config', type=str, default='./config/model.yaml')
        module_args.add_argument('--data_config', type=str, default='./config/data.yaml')

        # Configuration Files (Optional)
        cfg_args = self.add_argument_group('config (optional)')
        cfg_args.add_argument('-c', '--config', is_config_file=True, type=str)
