import argparse
import yaml
import torch
import pdb
import os
import numpy as np
import datasets.codraw_data as codraw_data

from cfgs.base_cfgs import ExpConfig
from datasets.datagen import BOWAddUpdateData, custom_collate
from torch.utils.data import DataLoader
from models.model import LSTMAddOnlyDrawer
from evaluation.eval import make_fns, eval_fns, scripted_tell, \
                            scripted_tell_before_peek, scripted_tell_after_peek, ComponentEvaluator
from utils.abs_metric import scene_similarity


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
        data_bowaddupdate_a = BOWAddUpdateData(cfgs, split='a')
        data_bowaddupdate_b = BOWAddUpdateData(cfgs, split='b')
        
        # Define model
        # Custom loss is defined within the model itself
        model_a = LSTMAddOnlyDrawer(
            cfgs, datagen=data_bowaddupdate_a, d_embeddings=cfgs.EMBEDDING_DIM,
            d_hidden=cfgs.HIDDEN_DIM, d_lstm=cfgs.LSTM_DIM, num_lstm_layers=cfgs.LAYER_SIZE,
            pre_lstm_dropout=cfgs.PRELSTM_DROPOUT, lstm_dropout=cfgs.LSTM_DROPOUT
        )
        model_b = LSTMAddOnlyDrawer(
            cfgs, datagen=data_bowaddupdate_b, d_embeddings=cfgs.EMBEDDING_DIM,
            d_hidden=cfgs.HIDDEN_DIM, d_lstm=cfgs.LSTM_DIM, num_lstm_layers=cfgs.LAYER_SIZE,
            pre_lstm_dropout=cfgs.PRELSTM_DROPOUT, lstm_dropout=cfgs.LSTM_DROPOUT
        )


        model_a.cuda()
        model_b.cuda()
        model_a.train()
        model_b.train()

        if cfgs.N_GPU > 1:
            model_a = torch.nn.DataParallel(model_a, device_ids=cfgs.DEVICES)
            model_b = torch.nn.DataParallel(model_b, device_ids=cfgs.DEVICES)

        if cfgs.RESUME:
            pass
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
            optimizer_a = optim_a(model_a.parameters(), lr=cfgs.LR,
                              **cfgs.OPT_PARAMS)
            
            optim_b = getattr(torch.optim, cfgs.OPT)
            optimizer_b = optim_b(model_b.parameters(), lr=cfgs.LR,
                              **cfgs.OPT_PARAMS)

            start_epoch_a = 0
            start_epoch_b = 0
        
        # Dataloader
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

        print("Training drawer A...")
        # Train drawer A
        for epoch in range(start_epoch_a, cfgs.MAX_EPOCH):
            for step, sample_iter in enumerate(dataloader_a):
                optimizer_a.zero_grad()

                sample_iter = {k: v.cuda() for k, v in sample_iter.items()}
                loss = model_a(sample_iter)
                loss.backward()

                if cfgs.GRAD_CLIP > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model_a.parameters(),
                        cfgs.GRAD_CLIP
                    )

                optimizer_a.step()
            
            # Evaluate for each epoch
            model_a.eval()
            print("Done epoch: {} loss: {}".format(epoch, loss.item()))
            if epoch % 1 == 0:
                for split in ('a',):
                    sims = eval_fns(cfgs, make_fns(split, scripted_tell, (model_a, model_b)), limit=100)
                    print('split: ', split, sims.mean())

                    sims = eval_fns(cfgs, make_fns(split, scripted_tell_before_peek, (model_a, model_b)), limit=100)
                    print('split: ', split, 'before', sims.mean())

                    sims = eval_fns(cfgs, make_fns(split, scripted_tell_after_peek, (model_a, model_b)), limit=100)
                    print('split: ', split, 'after', sims.mean())

            model_a.train()   
        
        print("Training drawer B...")
        # Train drawer B
        for epoch in range(start_epoch_b, cfgs.MAX_EPOCH):
            for step, sample_iter in enumerate(dataloader_b):
                optimizer_b.zero_grad()

                sample_iter = {k: v.cuda() for k, v in sample_iter.items()}
                loss = model_b(sample_iter)
                loss.backward()

                if cfgs.GRAD_CLIP > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        model_b.parameters(),
                        cfgs.GRAD_CLIP
                    )

                optimizer_b.step()
            
            # Evaluate for each epoch
            model_b.eval()
            print("Done epoch: {} loss: {}".format(epoch, loss.item()))
            if epoch % 1 == 0:
                for split in ('b',):
                    sims = eval_fns(cfgs, make_fns(split, scripted_tell, (model_a, model_b)), limit=100)
                    print('split: ', split, sims.mean())

                    sims = eval_fns(cfgs, make_fns(split, scripted_tell_before_peek, (model_a, model_b)), limit=100)
                    print('split: ', split, 'before', sims.mean())

                    sims = eval_fns(cfgs, make_fns(split, scripted_tell_after_peek, (model_a, model_b)), limit=100)
                    print('split: ', split, 'after', sims.mean())
            
            model_b.train()


        # Save states
        state_a = {
            'specs': model_a.spec,
            'optimizer': optimizer_a.state_dict()
        }

        state_b = {
            'specs': model_b.spec,
            'optimizer': optimizer_b.state_dict()
        }

        model_states = {
            'drawer_a': state_a,
            'drawer_b': state_b
        }

        # TODO: Currently does not accommodate saving in the middle of training.
        torch.save(
            model_states,
            os.path.join(cfgs.CKPTS_PATH,
            'ckpt_' + cfgs.VERSION + 
            '_epoch' + str(cfgs.MAX_EPOCH) + '.pkl')
        )

    
    # TODO: the original scripts behave differently for training and testing, as BOWAddUpdateData is not used for testing
    # the split for testing is obtained by accessing the keys created in data_for_splits()
    elif run_mode == 'val':
        # model_a.eval()
        # model_b.eval()
        # data_bowaddupdate_dev = BOWAddUpdateData(cfgs, split='dev')
        # data_bowaddupdate_test = BOWAddUpdateData(cfgs, split='test')

        # print("len_data_dev", len(data_bowaddupdate_dev))
        # print("len_data_test", len(data_bowaddupdate_test))

        eval_split = 'dev'
        if cfgs.CKPT_PATH is not None:
            path = cfgs.CKPT_PATH
        else:
            path = os.path.join(
                cfgs.CKPTS_PATH, 
                'ckpt_' + cfgs.CKPT_VERSION + 
                '_epoch' + str(cfgs.CKPT_EPOCH) + '.pkl'
            )

        # Load model
        ckpt = torch.load(path)

        data_bowaddupdate_a = BOWAddUpdateData(cfgs, spec=ckpt['drawer_a']['specs']['datagen_spec'])
        data_bowaddupdate_b = BOWAddUpdateData(cfgs, spec=ckpt['drawer_b']['specs']['datagen_spec'])

        model_a = LSTMAddOnlyDrawer(
            cfgs, datagen=data_bowaddupdate_a, d_embeddings=cfgs.EMBEDDING_DIM,
            d_hidden=cfgs.HIDDEN_DIM, d_lstm=cfgs.LSTM_DIM, num_lstm_layers=cfgs.LAYER_SIZE,
            pre_lstm_dropout=cfgs.PRELSTM_DROPOUT, lstm_dropout=cfgs.LSTM_DROPOUT
        )
        model_b = LSTMAddOnlyDrawer(
            cfgs, datagen=data_bowaddupdate_b, d_embeddings=cfgs.EMBEDDING_DIM,
            d_hidden=cfgs.HIDDEN_DIM, d_lstm=cfgs.LSTM_DIM, num_lstm_layers=cfgs.LAYER_SIZE,
            pre_lstm_dropout=cfgs.PRELSTM_DROPOUT, lstm_dropout=cfgs.LSTM_DROPOUT
        )

        model_a.load_state_dict(ckpt['drawer_a']['specs']['state_dict'])
        model_b.load_state_dict(ckpt['drawer_b']['specs']['state_dict'])

        model_a.cuda()
        model_b.cuda()
        model_a.eval()
        model_b.eval()

        if cfgs.N_GPU > 1:
            model_a = torch.nn.DataParallel(model_a, device_ids=cfgs.DEVICES)
            model_b = torch.nn.DataParallel(model_b, device_ids=cfgs.DEVICES)

        
        lstm_drawer = (model_a, model_b)

        # Evaluate on dev set
        with torch.no_grad():
            print("Evaluating on dev set...")
            limit = None
            component_evaluator = ComponentEvaluator.get(cfgs)

            # Human scene similarity
            human_sims = np.array([
                scene_similarity(human_scene, true_scene)
                for true_scene, human_scene in codraw_data.get_truth_and_human_scenes(cfgs, eval_split)
                ])
            
            print("Human scene similarity: mean={:.6f} std={:.6f} median={:.6f}".format(human_sims.mean(), human_sims.std(), np.median(human_sims)))
            print("")

            # Drawer evaluations against script
            print("Drawer evaluations against script")
            print("Drawer           Scene similarity")

            for split in ('a', 'b'):
                sims = eval_fns(cfgs, make_fns(split, scripted_tell, lstm_drawer), limit=limit, split=eval_split)
                print("LSTM drawer_{}:\t {}".format(split, sims.mean()))

            print("\n")
            print("Drawer evaluations against script")
            print("Drawer            Dir   \t Expr(human)\t Pose(human)\t Depth  \t xy (sq.)\t x-only  \t y-only")
            for split in ('a', 'b'):
                components = component_evaluator.eval_fns(make_fns(split, scripted_tell, lstm_drawer), limit=limit, split=eval_split)
                print("LSTM drawer_" + split, "\t", "\t".join(f"{num: .6f}" for num in components))

    elif run_mode == 'test':
        eval_split = 'test'
        if cfgs.CKPT_PATH is not None:
            path = cfgs.CKPT_PATH
        else:
            path = os.path.join(
                cfgs.CKPTS_PATH, 
                'ckpt_' + cfgs.CKPT_VERSION + 
                '_epoch' + str(cfgs.CKPT_EPOCH) + '.pkl'
            )

        # Load model
        ckpt = torch.load(path)

        data_bowaddupdate_a = BOWAddUpdateData(cfgs, spec=ckpt['drawer_a']['specs']['datagen_spec'])
        data_bowaddupdate_b = BOWAddUpdateData(cfgs, spec=ckpt['drawer_b']['specs']['datagen_spec'])

        model_a = LSTMAddOnlyDrawer(
            cfgs, datagen=data_bowaddupdate_a, d_embeddings=cfgs.EMBEDDING_DIM,
            d_hidden=cfgs.HIDDEN_DIM, d_lstm=cfgs.LSTM_DIM, num_lstm_layers=cfgs.LAYER_SIZE,
            pre_lstm_dropout=cfgs.PRELSTM_DROPOUT, lstm_dropout=cfgs.LSTM_DROPOUT
        )
        model_b = LSTMAddOnlyDrawer(
            cfgs, datagen=data_bowaddupdate_b, d_embeddings=cfgs.EMBEDDING_DIM,
            d_hidden=cfgs.HIDDEN_DIM, d_lstm=cfgs.LSTM_DIM, num_lstm_layers=cfgs.LAYER_SIZE,
            pre_lstm_dropout=cfgs.PRELSTM_DROPOUT, lstm_dropout=cfgs.LSTM_DROPOUT
        )

        model_a.load_state_dict(ckpt['drawer_a']['specs']['state_dict'])
        model_b.load_state_dict(ckpt['drawer_b']['specs']['state_dict'])

        model_a.cuda()
        model_b.cuda()
        model_a.eval()
        model_b.eval()

        if cfgs.N_GPU > 1:
            model_a = torch.nn.DataParallel(model_a, device_ids=cfgs.DEVICES)
            model_b = torch.nn.DataParallel(model_b, device_ids=cfgs.DEVICES)

        lstm_drawer = (model_a, model_b)

        # Evaluate on test set
        with torch.no_grad():
            print("Evaluating on test set...")
            limit = None
            component_evaluator = ComponentEvaluator.get(cfgs)

            # Human scene similarity
            human_sims = np.array([
                scene_similarity(human_scene, true_scene)
                for true_scene, human_scene in codraw_data.get_truth_and_human_scenes(cfgs, eval_split)
                ])
            
            print("Human scene similarity: mean={:.6f} std={:.6f} median={:.6f}".format(human_sims.mean(), human_sims.std(), np.median(human_sims)))
            print("")

            # Drawer evaluations against script
            # eval_automatic.py use split 'b' for drawer officially.
            print("Drawer evaluations against script")
            print("Drawer           Scene similarity")

            for split in ('a', 'b'):
                sims = eval_fns(cfgs, make_fns(split, scripted_tell, lstm_drawer), limit=limit, split=eval_split)
                print("LSTM drawer_{}:\t {}".format(split, sims.mean()))
            
            print("\n")
            print("Drawer evaluations against script")
            print("Drawer            Dir   \t Expr(human)\t Pose(human)\t Depth  \t xy (sq.)\t x-only  \t y-only")
            for split in ('a', 'b'):
                components = component_evaluator.eval_fns(make_fns(split, scripted_tell, lstm_drawer), limit=limit, split=eval_split)
                print("LSTM drawer_" + split, "\t", "\t".join(f"{num: .6f}" for num in components))

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
