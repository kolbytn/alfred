import os
import random
import json
import torch
import pprint
import collections
from PIL import Image
import numpy as np
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from tqdm import trange
import time
from importlib import import_module

from env.thor_env import ThorEnv
from models.nn.resnet import Resnet

# video recorder helper class
from datetime import datetime
import cv2
class VideoRecord:
    def __init__(self, path, name, fps=5):
        """
        param:
            path: video save path (str)
            name: video name (str)
            fps: frames per second (int) (default=5)
        example usage:
            rec = VideoRecord('path/to/', 'filename', 10)
        """
        self.path = path
        self.name = name
        self.fps = fps
        self.frames = []
    def record_frame(self, env_frame):
        """
            records video frame in this object
        param:
            env_frame: a frame from thor environment (ThorEnv().last_event.frame)
        example usage:
            env = Thorenv()
            lastframe = env.last_event.frame
            rec.record_frame(lastframes)
        """
        curr_image = Image.fromarray(np.uint8(env_frame))
        img = cv2.cvtColor(np.asarray(curr_image), cv2.COLOR_RGB2BGR)
        self.frames.append(img)
    def savemp4(self):
        """
            writes video to file at specified location, finalize video file
        example usage:
            rec.savemp4()
        """
        if len(self.frames) == 0:
            raise Exception("Can't write video file with no frames recorded")
        height, width, layers = self.frames[0].shape
        size = (width,height)
        out = cv2.VideoWriter(f"{self.path}{self.name}.mp4", 0x7634706d, self.fps, size)
        for i in range(len(self.frames)):
            out.write(self.frames[i])
        out.release()


class Module(nn.Module):

    def __init__(self, args, vocab, manager=None):
        '''
        Base Seq2Seq agent with common train and val loops
        '''
        super().__init__()

        # sentinel tokens
        self.pad = 0
        self.seg = 1

        # args and vocab
        self.args = args
        self.vocab = vocab

        # emb modules
        self.emb_word = nn.Embedding(len(vocab['word']), args.demb)
        self.emb_action_low = nn.Embedding(len(vocab['action_low']), args.demb)

        # end tokens
        self.stop_token = self.vocab['action_low'].word2index("<<stop>>", train=False)
        self.seg_token = self.vocab['action_low'].word2index("<<seg>>", train=False)

        # set random seed (Note: this is not the seed used to initialize THOR object locations)
        random.seed(a=args.seed)

        # summary self.writer
        self.summary_writer = None

        # load resnet
        args.visual_model = 'resnet18'
        self.resnet = Resnet(args, eval=True, share_memory=True, use_conv_feat=True)

        # multiprocessing
        self.manager = manager

    def run_train(self, splits, optimizer=None):
        '''
        training loop
        '''

        device = torch.device('cuda') if self.args.gpu else torch.device('cpu')

        # splits
        train = splits['train']
        valid_seen = splits['valid_seen']
        valid_unseen = splits['valid_unseen']

        # debugging: chose a small fraction of the dataset
        if self.args.dataset_fraction > 0:
            small_train_size = int(self.args.dataset_fraction * 0.7)
            small_valid_size = int((self.args.dataset_fraction * 0.3) / 2)
            train = train[:small_train_size]
            valid_seen = valid_seen[:small_valid_size]
            valid_unseen = valid_unseen[:small_valid_size]

        # debugging: use to check if training loop works without waiting for full epoch
        if self.args.fast_epoch:
            train = train[:16]
            valid_seen = valid_seen[:16]
            valid_unseen = valid_unseen[:16]

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=self.args.dout)

        # dump config
        fconfig = os.path.join(self.args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(self.args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=self.args.lr)

        # rollout thread model
        if self.args.episodes_per_epoch > 0:
            M = import_module('model.{}'.format(self.args.model))
            thread_model = M.Module(self.args, self.vocab)
            thread_model.share_memory()
            thread_model.to(device)
            task_queue = self.manager.Queue()
            rollouts = self.manager.Queue()
            threads = []
            for _ in range(self.args.num_rollout_threads):
                thread = mp.Process(target=Module.run_rollouts, args=(thread_model, task_queue, rollouts, self.args))
                thread.start()
                threads.append(thread)

        fsave = os.path.join(self.args.dout, 'latest.pth')
        torch.save({
            'metric': dict(),
            'model': self.state_dict(),
            'optim': optimizer.state_dict(),
            'args': self.args,
            'vocab': self.vocab,
        }, fsave)

        # display dout
        print("Saving to: %s" % self.args.dout)
        # best_loss = {'train': 1e10, 'valid_seen': 1e10, 'valid_unseen': 1e10}
        mean_reward = {'train': 0, 'valid_seen': 0, 'valid_unseen': 0}

        train_iter, valid_seen_iter, valid_unseen_iter = 0, 0, 0
        mean_valid_seen_reward = None
        mean_valid_unseen_reward = None

        for epoch in trange(0, self.args.epoch, desc='epoch'):
            self.train()

            ##################################################
            ###           Reinforcement Learning           ###
            ##################################################
            # print("==========================REINFORCEMENT LEARNING==========================")
            if self.args.episodes_per_epoch > 0:

                ############### Collect Rollouts #################
                thread_model.load_state_dict(self.state_dict())
                
                for traj in random.sample(train, self.args.episodes_per_epoch):
                    task_queue.put(traj)

                completed_rollouts = []
                while len(completed_rollouts) < self.args.episodes_per_epoch:
                    rollout = rollouts.get()

                    discounted = 0
                    for i in reversed(range(len(rollout))):
                        discounted = self.args.gamma * discounted + rollout[i]['reward']
                        rollout[i]['ret'] = discounted
                    completed_rollouts.append(rollout)

                for _ in range(self.args.ppo_epochs):

                    random.shuffle(completed_rollouts)
                    batch_idx = 0
                    while batch_idx < len(completed_rollouts):

                        ############### Prepare Batch #################
                        if batch_idx + self.args.ppo_batch < len(completed_rollouts):
                            batch_rollouts = completed_rollouts[batch_idx:batch_idx+self.args.ppo_batch] 
                        else:
                            batch_rollouts = completed_rollouts[batch_idx:]
                        batch_idx += len(batch_rollouts)

                        batch_size = sum(len(rollout) for rollout in batch_rollouts)

                        ret = torch.empty((batch_size, 1), dtype=torch.float64, device=device)
                        out_value = torch.empty((batch_size, 1), dtype=torch.float64, device=device)
                        prev_action_dist = torch.empty((batch_size, 15), dtype=torch.float64, device=device)
                        prev_action_mask_dist = torch.empty((batch_size, 300, 300, 2), dtype=torch.float64, device=device)
                        curr_action_dist = torch.empty((batch_size, 15), dtype=torch.float64, device=device)
                        curr_action_mask_dist = torch.empty((batch_size, 300, 300, 2), dtype=torch.float64, device=device)
                        action_idx = torch.empty((batch_size, 1), dtype=torch.long, device=device)
                        action_mask_idx = torch.empty((batch_size, 300, 300), dtype=torch.long, device=device)

                        step_idx = 0
                        for i, rollout in enumerate(batch_rollouts):
                            self.reset()

                            for j, step in enumerate(rollout):
                                feat = {
                                    'frames': torch.from_numpy(step['frames']).detach().to(device), 
                                    'lang_goal_instr': nn.utils.rnn.PackedSequence(torch.from_numpy(step['lang_goal_instr_data']).detach(),
                                        torch.from_numpy(step['lang_goal_instr_batch']),
                                        torch.from_numpy(step['lang_goal_instr_sorted']) if step['lang_goal_instr_sorted'] is not None else None,
                                        torch.from_numpy(step['lang_goal_instr_unsorted']) if step['lang_goal_instr_unsorted'] is not None else None).to(device)
                                }
                                out = self.step(feat)
                                pred = self.sample_pred(out)

                                out_value[step_idx] = out['out_value'][0][0]
                                curr_action_dist[step_idx] = pred['action_low_dist']
                                curr_action_mask_dist[step_idx] = pred['action_low_mask_dist']

                                ret[step_idx] = torch.from_numpy(step['ret'])
                                prev_action_dist[step_idx] = torch.from_numpy(step['action_dist'])
                                prev_action_mask_dist[step_idx] = torch.from_numpy(step['action_mask_dist'])
                                action_idx[step_idx] = torch.from_numpy(step['action_idx'])
                                action_mask_idx[step_idx] = torch.from_numpy(step['action_mask_idx'])

                                step_idx += 1

                        ############### Update Policy ###############
                        optimizer.zero_grad()

                        advantage = ret - out_value
                        value_loss = self.args.value_constant * torch.mean((ret - out_value) ** 2)
                        advantage.detach_()

                        curr_prob = torch.log(curr_action_dist[range(curr_action_dist.shape[0]), action_idx.squeeze(1)].unsqueeze(1))
                        prev_prob = torch.log(prev_action_dist[range(prev_action_dist.shape[0]), action_idx.squeeze(1)].unsqueeze(1))
                        curr_mask_prob = torch.log(curr_action_mask_dist.view(-1, 2)[range(curr_action_mask_dist.view(-1, 2).shape[0]), action_mask_idx.view(-1)].view(-1, 300, 300))
                        prev_mask_prob = torch.log(prev_action_mask_dist.view(-1, 2)[range(prev_action_mask_dist.view(-1, 2).shape[0]), action_mask_idx.view(-1)].view(-1, 300, 300))

                        curr_total_prob = curr_prob + torch.sum(torch.sum(curr_mask_prob, 1, keepdim=False), 1, keepdim=True)
                        prev_total_prob = prev_prob + torch.sum(torch.sum(prev_mask_prob, 1, keepdim=False), 1, keepdim=True)
                        ratio = curr_total_prob - prev_total_prob
                        left = torch.exp(ratio) * advantage
                        right = torch.clamp(ratio, 1 - self.args.epsilon, 1 + self.args.epsilon) * advantage
                        policy_loss = self.args.policy_constant * -torch.mean(torch.min(left, right))

                        loss = value_loss + policy_loss
                        loss.backward()
                        optimizer.step()
                        print("PPO Step... Value Loss: {:.2f}, Polcy Loss: {:.2f}".format(
                              value_loss, policy_loss))

            ##################################################
            ###            Immitation Learning             ###
            ##################################################

            # print("==========================IMITATION LEARNING==========================")

            m_train = collections.defaultdict(list)
            self.adjust_lr(optimizer, self.args.lr, epoch, decay_epoch=self.args.decay_epoch)
            # p_train = {}
            total_train_loss = list()
            random.shuffle(train) # shuffle every epoch
            sampled_train = train[:self.args.batch * self.args.batches_per_epoch] if self.args.batch * self.args.batches_per_epoch < len(train) else train

            for batch, feat in self.iterate(sampled_train, self.args.batch):
                out = self.forward(feat)
                preds = self.extract_preds(out, batch, feat)
                # p_train.update(preds)
                loss = self.compute_loss(out, batch, feat)
                for k, v in loss.items():
                    ln = 'loss_' + k
                    m_train[ln].append(v.item())
                    self.summary_writer.add_scalar('train/' + ln, v.item(), train_iter)

                # optimizer backward pass
                optimizer.zero_grad()
                sum_loss = sum(loss.values())
                sum_loss.backward()
                optimizer.step()

                self.summary_writer.add_scalar('train/loss', sum_loss, train_iter)
                sum_loss = sum_loss.detach().cpu()
                total_train_loss.append(float(sum_loss))
                train_iter += self.args.batch

            ##################################################
            ###                 Validation                 ###
            ##################################################

            # # NOTE: Original Implementation: predict action and compute loss
            # # compute metrics for train (too memory heavy!)
            # m_train = {k: sum(v) / len(v) for k, v in m_train.items()}
            # m_train.update(self.compute_metric(p_train, train))
            # m_train['total_loss'] = sum(total_train_loss) / len(total_train_loss)
            # self.summary_writer.add_scalar('train/total_loss', m_train['total_loss'], train_iter)

            # # compute metrics for valid_seen
            # p_valid_seen, valid_seen_iter, total_valid_seen_loss, m_valid_seen = self.run_pred(valid_seen, args=self.args, name='valid_seen', iter=valid_seen_iter)
            # m_valid_seen.update(self.compute_metric(p_valid_seen, valid_seen))
            # m_valid_seen['total_loss'] = float(total_valid_seen_loss)
            # self.summary_writer.add_scalar('valid_seen/total_loss', m_valid_seen['total_loss'], valid_seen_iter)

            # # compute metrics for valid_unseen
            # p_valid_unseen, valid_unseen_iter, total_valid_unseen_loss, m_valid_unseen = self.run_pred(valid_unseen, args=self.args, name='valid_unseen', iter=valid_unseen_iter)
            # m_valid_unseen.update(self.compute_metric(p_valid_unseen, valid_unseen))
            # m_valid_unseen['total_loss'] = float(total_valid_unseen_loss)
            # self.summary_writer.add_scalar('valid_unseen/total_loss', m_valid_unseen['total_loss'], valid_unseen_iter)


            # # NOTE: RL implementation: rollout and record trajectory
            if (epoch + 1) % self.args.validation_frequency == 0:
                print("==========================VALIDATION==========================")
                
                # sampled = random.sample(valid_seen, self.args.validation_episodes // 2)              # seen envs
                # sampled.extend(random.sample(valid_unseen, self.args.validation_episodes // 2))      # unseen envs
                sampled = valid_seen + valid_unseen
                print(sampled)
                total_rewards = [] # 1st half: seen, 2nd half: unseen
                
                for i, task in enumerate(sampled):

                    # reset model
                    self.reset()

                    # setup scene
                    traj_data = self.load_task_json(task)
                    r_idx = task['repeat_idx']
                    

                    self.setup_scene(env, traj_data, r_idx, self.args)

                    feat = self.featurize([traj_data], load_frames=False, load_mask=False)

                    done = False
                    fails = 0
                    total_reward = 0
                    num_steps = 0

                    # initialize video recording
                    task_name = '_'.join(('_'.join(str(datetime.now()).split(':')) + '_' + task['task']).split('/'))
                    print("video recording: (", task_name, ") created at:", self.args.video_output_path)
                    video_recording = VideoRecord(self.args.video_output_path, task_name, self.args.video_fps)

                    while not done and num_steps < self.args.max_steps:

                        # extract visual features
                        curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                        feat['frames'] = self.resnet.featurize([curr_image], batch=1).unsqueeze(0)

                        # record video using last observation
                        video_recording.record_frame(env.last_event.frame)

                        # forward model
                        m_out = self.step(feat)
                        m_pred = self.extract_preds(m_out, [traj_data], feat, clean_special_tokens=False)
                        m_pred = list(m_pred.values())[0]

                        # # check if <<stop>> was predicted
                        # if m_pred['action_low'] == "<<stop>>":
                        #     print("\tpredicted STOP")
                        #     break

                        # get action and mask
                        action, mask = m_pred['action_low'], m_pred['action_low_mask'][0]
                        mask = np.squeeze(mask, axis=0) if self.has_interaction(action) else None

                        # use predicted action and mask (if available) to interact with the env
                        t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=self.args.smooth_nav, debug=self.args.debug)

                        # if not t_success:
                        #     fails += 1
                        #     if fails >= self.args.max_fails:
                        #         print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                        #         break

                        # next time-step
                        reward, done = env.get_transition_reward()
                        total_reward += reward
                        num_steps += 1
                    
                    total_rewards.append(total_reward)
                    video_recording.savemp4()
                
                mean_valid_seen_reward = sum(total_rewards[:(len(total_rewards)//2)]) / (len(total_rewards) // 2)
                mean_valid_unseen_reward = sum(total_rewards[(len(total_rewards)//2):]) / (len(total_rewards) // 2)
            
                ##################################################
                ###                   Logging                  ###
                ##################################################

                # print("==========================LOGGING==========================")

                stats = {'epoch': epoch,
                        'valid_seen': (mean_valid_seen_reward),
                        'valid_unseen': (mean_valid_unseen_reward)}

                # check reward if better then save
                # new best valid_seen loss
                if mean_valid_seen_reward is not None and mean_valid_seen_reward > mean_reward['valid_seen']:
                    print('\nFound new best valid_seen!! Saving...')
                    fsave = os.path.join(self.args.dout, 'best_seen.pth')
                    torch.save({
                        'metric': stats,
                        'model': self.state_dict(),
                        'optim': optimizer.state_dict(),
                        'args': self.args,
                        'vocab': self.vocab,
                    }, fsave)
                    fbest = os.path.join(self.args.dout, 'best_seen.json')
                    with open(fbest, 'wt') as f:
                        json.dump(stats, f, indent=2)

                    # fpred = os.path.join(args.dout, 'valid_seen.debug.preds.json')
                    # with open(fpred, 'wt') as f:
                    #     json.dump(self.make_debug(p_valid_seen, valid_seen), f, indent=2)
                    mean_reward['valid_seen'] = mean_valid_seen_reward

                # new best valid_unseen loss
                if mean_valid_unseen_reward is not None and mean_valid_unseen_reward < mean_reward['valid_unseen']:
                    print('Found new best valid_unseen!! Saving...')
                    fsave = os.path.join(self.args.dout, 'best_unseen.pth')
                    torch.save({
                        'metric': stats,
                        'model': self.state_dict(),
                        'optim': optimizer.state_dict(),
                        'args': self.args,
                        'vocab': self.vocab,
                    }, fsave)
                    fbest = os.path.join(self.args.dout, 'best_unseen.json')
                    with open(fbest, 'wt') as f:
                        json.dump(stats, f, indent=2)

                    # fpred = os.path.join(self.args.dout, 'valid_unseen.debug.preds.json')
                    # with open(fpred, 'wt') as f:
                    #     json.dump(self.make_debug(p_valid_unseen, valid_unseen), f, indent=2)
                    mean_reward['valid_unseen'] = mean_valid_unseen_reward



                # save the latest checkpoint
                if self.args.save_every_epoch:
                    fsave = os.path.join(self.args.dout, 'net_epoch_%d.pth' % epoch)
                else:
                    fsave = os.path.join(self.args.dout, 'latest.pth')
                torch.save({
                    'metric': stats,
                    'model': self.state_dict(),
                    'optim': optimizer.state_dict(),
                    'args': self.args,
                    'vocab': self.vocab,
                }, fsave)

                ## debug action output json for train
                # fpred = os.path.join(self.args.dout, 'train.debug.preds.json')
                # with open(fpred, 'wt') as f:
                #     json.dump(self.make_debug(p_train, train), f, indent=2)

                # write stats
                for split in stats.keys():
                    if isinstance(stats[split], dict):
                        for k, v in stats[split].items():
                            self.summary_writer.add_scalar(split + '/' + k, v, train_iter)
                pprint.pprint(stats)

        for _ in range(len(threads)):
            task_queue.put(None)
        for t in threads:
            t.join()

    @classmethod
    def run_rollouts(cls, model, task_queue, rollouts, args):
        env = ThorEnv()

        while True:
            task = task_queue.get()
            if task is None:
                break

            # reset model
            model.reset()

            # setup scene
            traj_data = model.load_task_json(task)
            r_idx = task['repeat_idx']
            cls.setup_scene(env, traj_data, r_idx, args)

            feat = model.featurize([traj_data], load_frames=False, load_mask=False)

            curr_rollout = []
            done = False
            fails = 0
            total_reward = 0
            num_steps = 0
            while not done and num_steps < args.max_steps:

                # extract visual features
                curr_image = Image.fromarray(np.uint8(env.last_event.frame))
                feat['frames'] = model.resnet.featurize([curr_image], batch=1).unsqueeze(0)

                # forward model
                out = model.step(feat)
                pred = model.sample_pred(out)

                # # check if <<stop>> was predicted
                # if pred['action_low'] == "<<stop>>":
                #     print("\tpredicted STOP")
                #     break

                # get action and mask
                action = pred['action_low']
                mask = pred['action_low_mask'] if cls.has_interaction(action) else None

                # use predicted action and mask (if available) to interact with the env
                t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)

                # if not t_success:
                #     fails += 1
                #     if fails >= args.max_fails:
                #         print("Interact API failed %d times" % fails + "; latest error '%s'" % err)
                #         break

                # next time-step
                reward, done = env.get_transition_reward()
                total_reward += reward
                num_steps += 1

                curr_rollout.append({
                    'frames': feat['frames'].cpu().detach().numpy(),
                    'lang_goal_instr_data': feat['lang_goal_instr'].data.cpu().detach().numpy(),
                    'lang_goal_instr_batch': feat['lang_goal_instr'].batch_sizes.cpu().detach().numpy(),
                    'lang_goal_instr_sorted': feat['lang_goal_instr'].sorted_indices.cpu().detach().numpy() if feat['lang_goal_instr'].sorted_indices is not None else None,
                    'lang_goal_instr_unsorted': feat['lang_goal_instr'].unsorted_indices.cpu().detach().numpy() if feat['lang_goal_instr'].unsorted_indices is not None else None,
                    'action_dist': pred['action_low_dist'].cpu().detach().numpy(),
                    'action_mask_dist': pred['action_low_mask_dist'].cpu().detach().numpy(),
                    'action_idx': pred['action_low_idx'].cpu().detach().numpy(),
                    'action_mask_idx': pred['action_low_mask_idx'].cpu().detach().numpy(),
                    'reward': np.array([reward])
                })

            rollouts.put(curr_rollout)
        env.stop()

    def run_pred(self, dev, args=None, name='dev', iter=0):
        '''
        validation loop
        '''
        args = args or self.args
        m_dev = collections.defaultdict(list)
        p_dev = {}
        self.eval()
        total_loss = list()
        dev_iter = iter
        for batch, feat in self.iterate(dev, args.batch):
            out = self.forward(feat)
            preds = self.extract_preds(out, batch, feat)
            p_dev.update(preds)
            loss = self.compute_loss(out, batch, feat)
            for k, v in loss.items():
                ln = 'loss_' + k
                m_dev[ln].append(v.item())
                self.summary_writer.add_scalar("%s/%s" % (name, ln), v.item(), dev_iter)
            sum_loss = sum(loss.values())
            self.summary_writer.add_scalar("%s/loss" % (name), sum_loss, dev_iter)
            total_loss.append(float(sum_loss.detach().cpu()))
            dev_iter += len(batch)

        m_dev = {k: sum(v) / len(v) for k, v in m_dev.items()}
        total_loss = sum(total_loss) / len(total_loss)
        return p_dev, dev_iter, total_loss, m_dev

    def featurize(self, batch):
        raise NotImplementedError()

    def forward(self, feat, max_decode=100):
        raise NotImplementedError()

    def extract_preds(self, out, batch, feat):
        raise NotImplementedError()

    def sample_pred(self, feat):
        raise NotImplementedError()

    def compute_loss(self, out, batch, feat):
        raise NotImplementedError()

    def compute_metric(self, preds, data):
        raise NotImplementedError()

    def rl_update(self, rollouts):
        raise NotImplementedError()

    def get_task_and_ann_id(self, ex):
        '''
        single string for task_id and annotation repeat idx
        '''
        return "%s_%s" % (ex['task_id'], str(ex['repeat_idx']))

    def make_debug(self, preds, data):
        '''
        readable output generator for debugging
        '''
        debug = {}
        for task in data:
            ex = self.load_task_json(task)
            i = self.get_task_and_ann_id(ex)
            debug[i] = {
                'lang_goal': ex['turk_annotations']['anns'][ex['ann']['repeat_idx']]['task_desc'],
                'action_low': [a['discrete_action']['action'] for a in ex['plan']['low_actions']],
                'p_action_low': preds[i]['action_low'].split(),
            }
        return debug

    def load_task_json(self, task):
        '''
        load preprocessed json from disk
        '''
        json_path = os.path.join(self.args.data, task['task'], '%s' % self.args.pp_folder, 'ann_%d.json' % task['repeat_idx'])
        with open(json_path) as f:
            data = json.load(f)
        return data

    def get_task_root(self, ex):
        '''
        returns the folder path of a trajectory
        '''
        return os.path.join(self.args.data, ex['split'], *(ex['root'].split('/')[-2:]))

    def iterate(self, data, batch_size):
        '''
        breaks dataset into batch_size chunks for training
        '''
        for i in trange(0, len(data), batch_size, desc='batch'):
            tasks = data[i:i+batch_size]
            batch = [self.load_task_json(task) for task in tasks]
            feat = self.featurize(batch)
            yield batch, feat

    def zero_input(self, x, keep_end_token=True):
        '''
        pad input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        return list(np.full_like(x[:-1], self.pad)) + end_token

    def zero_input_list(self, x, keep_end_token=True):
        '''
        pad a list of input with zeros (used for ablations)
        '''
        end_token = [x[-1]] if keep_end_token else [self.pad]
        lz = [list(np.full_like(i, self.pad)) for i in x[:-1]] + end_token
        return lz

    @staticmethod
    def adjust_lr(optimizer, init_lr, epoch, decay_epoch=5):
        '''
        decay learning rate every decay_epoch
        '''
        lr = init_lr * (0.1 ** (epoch // decay_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    @classmethod
    def load(cls, fsave):
        '''
        load pth model from disk
        '''
        save = torch.load(fsave)
        model = cls(save['args'], save['vocab'])
        model.load_state_dict(save['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(save['optim'])
        return model, optimizer

    @classmethod
    def has_interaction(cls, action):
        '''
        check if low-level action is interactive
        '''
        non_interact_actions = ['MoveAhead', 'Rotate', 'Look', '<<stop>>', '<<pad>>', '<<seg>>']
        if any(a in action for a in non_interact_actions):
            return False
        else:
            return True

    @classmethod
    def setup_scene(cls, env, traj_data, r_idx, args, reward_type='dense'):
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

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)
