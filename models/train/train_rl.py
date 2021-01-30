import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

import os
import torch
import torch.multiprocessing as mp
import pprint
import json
from data.preprocess import Dataset
from importlib import import_module
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models.utils.helper_utils import optimizer_to


if __name__ == '__main__':
    # multiprocessing settings
    mp.set_start_method('spawn')
    manager = mp.Manager()

    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--seed', help='random seed', default=123, type=int)
    parser.add_argument('--data', help='dataset folder', default='data/json_feat_2.1.0')
    parser.add_argument('--splits', help='json file containing train/dev/test splits', default='data/splits/oct21.json')
    parser.add_argument('--preprocess', help='store preprocessed data to json files', action='store_true')
    parser.add_argument('--pp_folder', help='folder name for preprocessed data', default='pp')
    parser.add_argument('--save_every_epoch', help='save model after every epoch (warning: consumes a lot of space)', action='store_true')
    parser.add_argument('--model', help='model to use', default='seq2seq_im_mask')
    parser.add_argument('--gpu', help='use gpu', action='store_true')
    parser.add_argument('--dout', help='where to save model', default='exp/model:{model}')
    parser.add_argument('--use_templated_goals', help='use templated goals instead of human-annotated goal descriptions', action='store_true')
    parser.add_argument('--resume', help='load a checkpoint')

    # hyper parameters
    parser.add_argument('--batch', help='batch size', default=8, type=int)
    parser.add_argument('--epoch', help='number of epochs', default=1000000, type=int)
    parser.add_argument('--lr', help='optimizer learning rate', default=1e-4, type=float)
    parser.add_argument('--decay_epoch', help='num epoch to adjust learning rate', default=10, type=int)
    parser.add_argument('--dhid', help='hidden layer size', default=512, type=int)
    parser.add_argument('--dframe', help='image feature vec size', default=2500, type=int)
    parser.add_argument('--demb', help='language embedding size', default=100, type=int)
    parser.add_argument('--pframe', help='image pixel size (assuming square shape eg: 300x300)', default=300, type=int)
    parser.add_argument('--mask_loss_wt', help='weight of mask loss', default=1., type=float)
    parser.add_argument('--action_loss_wt', help='weight of action loss', default=1., type=float)
    parser.add_argument('--subgoal_aux_loss_wt', help='weight of subgoal completion predictor', default=0., type=float)
    parser.add_argument('--pm_aux_loss_wt', help='weight of progress monitor', default=0., type=float)

    # rl settings
    parser.add_argument('--reward_config', default='models/config/rewards.json')
    parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true', help='smooth nav actions (might be required based on training data)')
    parser.add_argument('--num_rollout_processes', type=int, default=4, help='number of processes collecting rollouts')
    parser.add_argument('--num_valid_processes', type=int, default=4, help='number of processes doing validation')

    # rl parameters
    parser.add_argument('--max_steps', type=int, default=100, help='max steps before episode termination')
    parser.add_argument('--max_fails', type=int, default=10, help='max API execution failures before episode termination')
    parser.add_argument('--episodes_per_epoch', type=int, default=32, help='number of episodes to gather each epoch for reinforcement learning')
    parser.add_argument('--batches_per_epoch', type=int, default=32, help='max number of immitation learning batches for each epoch')
    parser.add_argument('--ppo_batch', type=int, default=4, help='number of rollouts in each ppo batch')
    parser.add_argument('--gamma', type=float, default=0.99, help='return discount factor')
    parser.add_argument('--epsilon', type=float, default=0.2, help='PPO trust region parameter')
    parser.add_argument('--value_constant', type=float, default=1, help='PPO value loss scalar')
    parser.add_argument('--policy_constant', type=float, default=1, help='PPO policy loss scalar')
    parser.add_argument('--ppo_epochs', type=int, default=4, help='number of epochs to run on PPO step')

    # dropouts
    parser.add_argument('--zero_goal', help='zero out goal language', action='store_true')
    parser.add_argument('--zero_instr', help='zero out step-by-step instr language', action='store_true')
    parser.add_argument('--lang_dropout', help='dropout rate for language (goal + instr)', default=0., type=float)
    parser.add_argument('--input_dropout', help='dropout rate for concatted input feats', default=0., type=float)
    parser.add_argument('--vis_dropout', help='dropout rate for Resnet feats', default=0.3, type=float)
    parser.add_argument('--hstate_dropout', help='dropout rate for LSTM hidden states during unrolling', default=0.3, type=float)
    parser.add_argument('--attn_dropout', help='dropout rate for attention', default=0., type=float)
    parser.add_argument('--actor_dropout', help='dropout rate for actor fc', default=0., type=float)

    # other settings
    parser.add_argument('--dec_teacher_forcing', help='use gpu', action='store_true')
    parser.add_argument('--temp_no_history', help='use gpu', action='store_true')

    # debugging
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--small_train', help='fast epoch during debugging', action='store_true')
    parser.add_argument('--small_valid', help='fast valid during debugging', action='store_true')
    parser.add_argument('--single_task', help='train and validate on a single task', action='store_true')
    parser.add_argument('--dataset_fraction', help='use fraction of the dataset for debugging (0 indicates full size)', default=0, type=int)
    parser.add_argument('--video_output_path', help='path for validation video files', default='data/validation_video/videos/', type=str)
    parser.add_argument('--video_fps', help='path for validation video fps', default=5, type=int)

    # args and init
    args = parser.parse_args()
    args.dout = args.dout.format(**vars(args))
    torch.manual_seed(args.seed)

    # check if dataset has been preprocessed
    if not os.path.exists(os.path.join(args.data, "%s.vocab" % args.pp_folder)) and not args.preprocess:
        raise Exception("Dataset not processed; run with --preprocess")

    # make output dir
    pprint.pprint(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)

    # load train/valid/tests splits
    with open(args.splits) as f:
        splits = json.load(f)
        pprint.pprint({k: len(v) for k, v in splits.items()})

    # preprocess and save
    if args.preprocess:
        print("\nPreprocessing dataset and saving to %s folders ... This will take a while. Do this once as required." % args.pp_folder)
        dataset = Dataset(args, None)
        dataset.preprocess_splits(splits)
        vocab = torch.load(os.path.join(args.dout, "%s.vocab" % args.pp_folder))
    else:
        vocab = torch.load(os.path.join(args.data, "%s.vocab" % args.pp_folder))

    # load model
    M = import_module('model.{}'.format(args.model))
    if args.resume:
        print("Loading: " + args.resume)
        model, optimizer = M.Module.load(args.resume)
    else:
        model = M.Module(args, vocab, manager)
        optimizer = None

    # to gpu
    if args.gpu:
        model = model.to(torch.device('cuda'))
        if not optimizer is None:
            optimizer_to(optimizer, torch.device('cuda'))

    # start train loop
    model.run_train(splits, optimizer=optimizer)