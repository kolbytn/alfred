import os
import random
import json
import torch
import pprint
import collections
import numpy as np
from torch import nn
from tensorboardX import SummaryWriter
from tqdm import trange

from models.nn.resnet import Resnet


class SplitPolicyModule(nn.Module):

    def __init__(self, args, vocab):
        super().__init__()

        # args and vocab
        self.args = args
        self.vocab = vocab
        
        # internal states
        self.test_mode = False

        # load resnet
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

        # gpu
        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

    def run_train(self, splits):

        # splits
        train = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=self.args.dout)
        
        # dump config
        fconfig = os.path.join(self.args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(self.args), f, indent=2)

        # optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)

        # environment
        env = ThorEnv()
        
        print("Saving to: %s" % self.args.dout)
        for epoch in trange(0, self.args.epoch, desc='epoch'):

            # reset model
            self.reset()

            # setup scene
            task = random.sample(train, 1)[0]
            traj_data = model.load_task_json(task)
            r_idx = task['repeat_idx']
            self.setup_scene(env, traj_data, r_idx, self.args)

            done = False
            step = 0
            while not done and step < self.args.max_steps:

                # extract visual features
                curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                feat['frames'] = resnet.featurize([curr_image], batch=1).unsqueeze(0)

                # forward model
                m_out = self.step(feat)
                m_pred = self.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
                m_pred = list(m_pred.values())[0]

                # get action and mask
                action, mask = m_pred['action_low'], m_pred['action_low_mask'][0]
                mask = np.squeeze(mask, axis=0) if self.has_interaction(action) else None

                # use predicted action and mask (if available) to interact with the env
                t_success, _, _, err, api_action = env.va_interact(action, interact_mask=mask, smooth_nav=False)

                step += 1

    def step(self, feat, prev_action=None):
        # output formatting
        feat['out_action_low'] = np.ones((1, 1, len(self.vocab['action_low']))) / len(self.vocab['action_low'])
        feat['out_action_low_mask'] = np.ones((1, 1, self.args.pframe, self.args.pframe))
        return feat

    def extract_preds(self, out, batch, feat, clean_special_tokens=True):
        pred = {}

        # index to API actions
        words = self.vocab['action_low'].index2word(feat['out_action_low'][0])

        # sigmoid preds to binary mask
        alow_mask = F.sigmoid(feat['out_action_low_mask'][0])
        p_mask = [(alow_mask[t] > 0.5).cpu().numpy() for t in range(alow_mask.shape[0])]

        task_id_ann = self.get_task_and_ann_id(batch[0])
        pred[task_id_ann] = {
            'action_low': ' '.join(words),
            'action_low_mask': p_mask,
        }

        return pred

    def forward(self, feat, max_decode=100):
        pass

    def featurize(self, batch):
        pass

    def reset(self):
        pass

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def setup_scene(self, env, traj_data, r_idx, args, reward_type='dense'):
        '''
        intialize the scene and agent from the task info
        '''
        # scene setup
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        env.reset(scene_name)
        env.restore_scene(object_poses, object_toggles, dirty_and_empty)

        # initialize to start position
        env.step(dict(traj_data['scene']['init_action']))

        # print goal instr
        print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    def has_interaction(self, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True
