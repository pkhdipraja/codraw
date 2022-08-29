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
        choices=['train', 'train+dev'],
        help='set training split',
        type=str
    )

    args = parser.parse_args()
    return args


def main(cfgs):
    run_mode = cfgs.RUN_MODE

    if run_mode == 'train':
        
        
        # Define model
        # Custom loss is defined within the model itself
        model_a = None
        model_b = None


        model_a.cuda()
        model_b.cuda()
        model_a.train()
        model_b.train()

        if cfgs.N_GPU > 1:
            model_a = torch.nn.DataParallel(model_a, device_ids=cfgs.DEVICES)
            model_b = torch.nn.DataParallel(model_b, device_ids=cfgs.DEVICES)

        if cfgs.RESUME:
            #TODO: define loading func for both model A and B
            # print("Resume training")
            # if cfgs.CKPT_PATH is not None:
            #     path = cfgs.CKPT_PATH
            # else:
            #     path = cfgs.CKPTS_PATH + \
            #            'ckpt_' + cfgs.CKPT_VERSION + \
            #            'epoch' + str(cfgs.CKPT_EPOCH) + '.pkl'
            
            # # Load model parameters
            # ckpt = torch.load(path)
            # model.load_state_dict(ckpt['state_dict'])

            # # Load optimizer parameters
            # optim = getattr(torch.optim, cfgs.OPT)
            # optimizer = optim(model.parameters(), lr=cfgs.LR,
            #                   **cfgs.OPT_PARAMS)
            # optimizer.load_state_dict(ckpt['optimizer'])

            # start_epoch = cfgs.CKPT_EPOCH

        else:
            path = None

            optim_a = getattr(torch.optim, cfgs.OPT)
            optimizer_a = optim(model_a.parameters(), lr=cfgs.LR,
                              **cfgs.OPT_PARAMS)
            
            optim_b = getattr(torch.optim, cfgs.OPT)
            optimizer_b = optim(model_b.parameters(), lr=cfgs.LR,
                              **cfgs.OPT_PARAMS)

            start_epoch_a = 0
            start_epoch_b = 0
        
        # Dataloader
        data_bowaddupdate_a = BOWAddUpdateData(cfgs, split='a')
        data_bowaddupdate_b = BOWAddUpdateData(cfgs, split='b')

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

        # Train drawer A
        for epoch in range(start_epoch_a, cfgs.MAX_EPOCH):
            for step, sample_iter in enumerate(dataloader_a):
                optimizer_a.zero_grad()

                sample_iter = sample_iter.cuda()
                loss = model_a(sample_iter)
                loss.backward()

                if cfgs.GRAD_CLIP > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model_a.parameters(),
                        cfgs.GRAD_CLIP
                    )

                optimizer_a.step()
            
            # Evaluate for each epoch
            #TODO: implement eval for drawer A
        
        

        # Train drawer B
        for epoch in range(start_epoch_b, cfgs.MAX_EPOCH):
            for step, sample_iter in enumerate(dataloader_b):
                optimizer_b.zero_grad()

                sample_iter = sample_iter.cuda()
                loss = model_b(sample_iter)
                loss.backward()

                if cfgs.GRAD_CLIP > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model_b.parameters(),
                        cfgs.GRAD_CLIP
                    )

                optimizer_b.step()
            
            # Evaluate for each epoch
            # TODO: implement eval for drawer B



        # Save states
        state_a = {
            'state_dict': model_a.state_dict(),
            'optimizer': optimizer_a.state_dict()
        }

        state_b = {
            'state_dict': model_b.state_dict(),
            'optimizer': optimizer_b.state_dict()
        }

        model_states = {
            'drawer_a': state_a,
            'drawer_b': state_b
        }

        # TODO: Currently does not accommodate saving in the middle of training.
        torch.save(
            model_states,
            cfgs.CKPTS_PATH + 
            'ckpt_' + cfgs.CKPT_VERSION + 
            'epoch' + str(cfgs.MAX_EPOCH) + '.pkl'
        )


        # include logic if resuming training
        # also func to select device
        # set model to train mode
        # possible gradient accumulation?
        # can use key: 'dev' or 'test' instead of 'a' or 'b' to access other splits
    elif run_mode == 'val':
        pass
        
        # model_a.eval()
        # model_b.eval()
        # data_bowaddupdate_dev = BOWAddUpdateData(cfgs, split='dev')
        # data_bowaddupdate_test = BOWAddUpdateData(cfgs, split='test')

        # print("len_data_dev", len(data_bowaddupdate_dev))
        # print("len_data_test", len(data_bowaddupdate_test))
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
