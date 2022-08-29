import argparse
import yaml
import torch

from cfgs.base_cfgs import ExpConfig
from datasets.datagen import BOWAddUpdateData, custom_collate
from torch.utils.data import DataLoader


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Train neural drawer model.")

    parser.add_argument(
        '--RUN', dest='RUN_MODE',
        choices=['train', 'val', 'test'],
        help='{train, val, test}',
        required=True
    )

    parser.add_argument(
        '--MODEL_CONFIG', dest='MODEL_CONFIG',
        help='experiment configuration file',
        type=str, required=True
    )

    parser.add_argument(
        '--GPU', dest='GPU',
        help='select gpu, e.g. "0, 1, 2" ',
        type=str
    )

    parser.add_argument(
        '--SEED', dest='SEED',
        help='fix random seed',
        type=int
    )

    parser.add_argument(
        '--VERSION', dest='VERSION',
        help='model version identifier',
        type=str
    )

    parser.add_argument(
        '--RESUME', dest='RESUME',
        help='resume training',
        action='store_true',
    )

    parser.add_argument(
        '--PINM', dest='PIN_MEM',
        help='disable pin memory',
        action='store_false',
    )

    parser.add_argument(
        '--NW', dest='NUM_WORKERS',
        help='multithreaded loading to accelerate IO',
        default=4,
        type=int
    )

    parser.add_argument(
        '--CKPT_V', dest='CKPT_VERSION',
        help='checkpoint version',
        type=str
    )

    parser.add_argument(
        '--CKPT_E', dest='CKPT_EPOCH',
        help='checkpoint epoch',
        type=int
    )

    parser.add_argument(
        '--CKPT_PATH', dest='CKPT_PATH',
        help='load checkpoint path, if \
        possible use CKPT_VERSION and CKPT_EPOCH',
        type=str
    )

    parser.add_argument(
        '--SPLIT', dest='TRAIN_SPLIT',
        choices=['train', 'train+valid'],
        help='set training split',
        type=str
    )

    args = parser.parse_args()
    return args


def main(cfgs):
    run_mode = cfgs.RUN_MODE

    if run_mode == 'train':
        
        data_bowaddupdate_a = BOWAddUpdateData(cfgs, split='a')
        data_bowaddupdate_b = BOWAddUpdateData(cfgs, split='b')

        # dont forget to use custom collate function
        dataloader_a = DataLoader(
            data_bowaddupdate_a, batch_size=cfgs.BATCH_SIZE, shuffle=True,
            num_workers=cfgs.NUM_WORKERS, pin_memory=cfgs.PIN_MEM,
            collate_fn=custom_collate
        )

        dataloader_b = DataLoader(
            data_bowaddupdate_b, batch_size=cfgs.BATCH_SIZE, shuffle=True,
            num_workers=cfgs.NUM_WORKERS, pin_memory=cfgs.PIN_MEM,
            collate_fn=custom_collate
        )

        print("len_data_a", len(data_bowaddupdate_a))
        print("len_data_b", len(data_bowaddupdate_b))

        # for idx, sample in enumerate(dataloader_a):
        #     sample_iter = print(sample)
        #     break
        
        
        data_bowaddupdate_dev = BOWAddUpdateData(cfgs, split='dev')
        data_bowaddupdate_test = BOWAddUpdateData(cfgs, split='test')

        print("len_data_dev", len(data_bowaddupdate_dev))
        print("len_data_test", len(data_bowaddupdate_test))


        # include logic if resuming training
        # also func to select device
        # set model to train mode
        # possible gradient accumulation?
        # can use key: 'dev' or 'test' instead of 'a' or 'b' to access other splits
    elif run_mode == 'val':
        pass
    elif run_mode == 'test':
        pass
    else:
        exit(-1)


if __name__ == "__main__":
    cfgs = ExpConfig()

    args = parse_args()
    args_dict = cfgs.parse_to_dict(args)

    model_cfg_file = './cfgs/{}.yml'.format(args.MODEL_CONFIG)
    with open(model_cfg_file, 'r') as model_f:
        model_yaml = yaml.safe_load(model_f)

    args_dict = {**args_dict, **model_yaml}

    cfgs.add_args(args_dict)
    cfgs.init_path()
    cfgs.setup()

    print("Hyperparameters:")
    print(cfgs)

    main(cfgs)
