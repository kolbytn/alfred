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
import time
from importlib import import_module
import queue
import csv

from env.thor_env import ThorEnv
from models.nn.resnet import Resnet
from models.utils.video_record import VideoRecord
from models.utils.resource_util import start_monitor, stop_monitor

from multiprocessing import Process
import subprocess
class ValidationProcesses:
    """ manage all validation processes """
    
    # evaluation script path
    EVALCUSTOM = os.path.join(os.environ['ALFRED_ROOT'], 'model', 'eval', 'eval_custom.py')

    def __init__(self):
        """ all process can be accessed here """
        self.processes = []
    
    def spawn(self):
        """ spawn a new validation process """
        newprocess = Process(target=subprocess.call, args=(['python', EVALCUSTOM]))
        self.processes.append(newprocess)
        newprocess.start()


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
        if self.args.small_train:
            train = train[:16]
        if self.args.small_valid:
            valid_seen = valid_seen[:8]
            valid_unseen = valid_unseen[:8]

        if self.args.single_task:
            train = [train[0]]
            valid_seen = [train[0]]
            valid_unseen = []

        # initialize summary writer for tensorboardX
        self.summary_writer = SummaryWriter(log_dir=self.args.dout)

        # dump config
        fconfig = os.path.join(self.args.dout, 'config.json')
        with open(fconfig, 'wt') as f:
            json.dump(vars(self.args), f, indent=2)

        # optimizer
        optimizer = optimizer or torch.optim.Adam(self.parameters(), lr=self.args.lr)

        # Rollout processes
        if self.args.episodes_per_epoch > 0:
            M = import_module('model.{}'.format(self.args.model))
            rollout_model = M.Module(self.args, self.vocab)
            rollout_model.share_memory()
            rollout_model.to(device)
            rollout_task_queue = self.manager.Queue()
            rollout_results = self.manager.Queue()
            rollout_processes = []
            for _ in range(self.args.num_rollout_processes):
                p = mp.Process(target=Module.run_rollouts, args=(rollout_model, rollout_task_queue, rollout_results, self.args, False))
                p.start()
                rollout_processes.append(p)

        # Validation processes
        M = import_module('model.{}'.format(self.args.model))
        valid_model = M.Module(self.args, self.vocab)
        valid_model.share_memory()
        valid_model.to(device)
        valid_task_queue = self.manager.Queue()
        valid_results = self.manager.Queue()
        valid_processes = []
        for _ in range(self.args.num_valid_processes):
            p = mp.Process(target=Module.run_rollouts, args=(valid_model, valid_task_queue, valid_results, self.args, True))
            p.start()
            valid_processes.append(p)
        for traj in valid_seen:
            valid_task_queue.put((traj, True))
        for traj in valid_unseen:
            valid_task_queue.put((traj, False))
        valid_completed = []
        train_start_time = time.time()
        valid_time = time.time()
        valid_epoch = 0

        # Create logging
        print("Saving to: %s" % self.args.dout) 
        valid_results_csv = os.path.join(self.args.dout, 'valid_results.csv')
        f = open(valid_results_csv, 'w+')
        f.close()

        fsave = os.path.join(self.args.dout, 'latest.pth')
        torch.save({
            'metric': dict(),
            'model': self.state_dict(),
            'optim': optimizer.state_dict(),
            'args': self.args,
            'vocab': self.vocab,
        }, fsave)

        best_results = {
            'sr_seen': None,
            'plw_sr_seen': None,
            'gc_seen': None,
            'plw_gc_seen': None,
            'sr_unseen': None,
            'plw_sr_unseen': None,
            'gc_unseen': None,
            'plw_gc_unseen': None
        }

        for epoch in range(0, self.args.epoch):
                
            ##################################################
            ###           Reinforcement Learning           ###
            ##################################################
            self.train()
            if self.args.episodes_per_epoch > 0:

                ############### Collect Rollouts #################
                rollout_model.load_state_dict(self.state_dict())
                
                for traj in np.random.choice(train, self.args.episodes_per_epoch):
                    rollout_task_queue.put(traj)

                total_rewards = []
                policy_losses = []
                value_losses = []
                completed_rollouts = []
                while len(completed_rollouts) < self.args.episodes_per_epoch:
                    rollout = rollout_results.get()

                    total_reward = 0
                    discounted = 0
                    for i in reversed(range(len(rollout))):
                        discounted = self.args.gamma * discounted + rollout[i]['reward']
                        rollout[i]['ret'] = discounted
                        total_reward += rollout[i]['reward'].item()
                    completed_rollouts.append(rollout)
                    total_rewards.append(total_reward)

                for _ in range(self.args.ppo_epochs):

                    
                    random.shuffle(completed_rollouts)
                    batch_idx = 0
                    while batch_idx < len(completed_rollouts):

                        ############### Prepare Batch #################

                        # monitor resource usage
                        monitor = start_monitor(path=self.args.dout, note=f"RL prep_batch epoch={_}")

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
                        
                        stop_monitor(monitor)

                        ############### Update Policy ###############
                        optimizer.zero_grad()

                        advantage = ret - out_value
                        value_loss = self.args.value_constant * torch.mean((ret - out_value) ** 2)
                        advantage.detach_()

                        # monitor resource usage
                        monitor = start_monitor(path=self.args.dout, note=f"RL calc_loss epoch={_}")

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

                        value_losses.append(value_loss.item())
                        policy_losses.append(policy_loss.item())

                        loss = value_loss + policy_loss

                        stop_monitor(monitor)

                        # monitor resource usage
                        monitor = start_monitor(path=self.args.dout, note=f"RL policy_update epoch={_}")
                        loss.backward()
                        optimizer.step()

                        stop_monitor(monitor)

                print("PPO Epoch: {} Time: {:.0f} Reward: {:.2f} Value Loss: {:.2f} Polcy Loss: {:.2f}".format(
                        epoch,
                        time.time() - train_start_time,
                        sum(total_rewards) / len(total_rewards), 
                        sum(value_losses) / len(value_losses), 
                        sum(policy_losses) / len(policy_losses)))

            ##################################################
            ###            Immitation Learning             ###
            ##################################################
            self.train()
            if self.args.batches_per_epoch > 0:
                
                

                m_train = collections.defaultdict(list)
                self.adjust_lr(optimizer, self.args.lr, epoch, decay_epoch=self.args.decay_epoch)
                total_train_loss = list()
                sampled_train = np.random.choice(train, self.args.batch * self.args.batches_per_epoch)

                for batch, feat in self.iterate(sampled_train, self.args.batch):
                    # monitor resource usage
                    monitor = start_monitor(path=self.args.dout, note=f"Imitation batch epoch={_}")

                    out = self.forward(feat)
                    preds = self.extract_preds(out, batch, feat)
                    loss = self.compute_loss(out, batch, feat)
                    for k, v in loss.items():
                        ln = 'loss_' + k
                        m_train[ln].append(v.item())
                        self.summary_writer.add_scalar('train/' + ln, v.item(), epoch)

                    # optimizer backward pass
                    optimizer.zero_grad()
                    sum_loss = sum(loss.values())
                    sum_loss.backward()
                    optimizer.step()

                    self.summary_writer.add_scalar('train/loss', sum_loss, epoch)
                    sum_loss = sum_loss.detach().cpu()
                    total_train_loss.append(float(sum_loss))

                    stop_monitor(monitor)

                print("BC Epoch: {} Time: {:.0f} Loss: {:.2f}".format(
                        epoch,
                        time.time() - train_start_time,
                        sum(total_train_loss) / len(total_train_loss)))

            ##################################################
            ###                 Validation                 ###
            ##################################################
            
            while True:
                try:
                    valid_completed.append(valid_results.get_nowait())
                except queue.Empty:
                    break

            if len(valid_completed) == len(valid_seen) + len(valid_unseen):
                # monitor resource usage
                monitor = start_monitor(path=self.args.dout, note=f"Validation train epoch={_}")

                # Count results
                def count_results(results, seen):
                    successes = 0
                    completed_goal_conditions = 0
                    total_goal_conditions = 0
                    path_len_weighted_success_spl = 0
                    path_len_weighted_goal_condition_spl = 0
                    path_len_weight = 0
                    for log_result in valid_completed:
                        if log_result['seen'] == seen:
                            successes += log_result['goal_satisfied']
                            completed_goal_conditions += log_result['completed_goal_conditions']
                            total_goal_conditions += log_result['total_goal_conditions']
                            path_len_weighted_success_spl += log_result['path_len_weighted_success_spl']
                            path_len_weighted_goal_condition_spl += log_result['path_len_weighted_goal_condition_spl']
                            path_len_weight += log_result['path_len_weight']
                    if seen and len(valid_seen) > 0 or not seen and len(valid_unseen) > 0:
                        sr = successes / len(valid_seen) if seen else successes / len(valid_unseen)
                        plw_sr = path_len_weighted_success_spl / path_len_weight
                        gc = completed_goal_conditions / total_goal_conditions
                        plw_gc = path_len_weighted_goal_condition_spl / path_len_weight
                    else:
                        sr = -1
                        plw_sr = -1
                        gc = -1
                        plw_gc = -1
                    return sr, plw_sr, gc, plw_gc

                sr_seen, plw_sr_seen, gc_seen, plw_gc_seen = count_results(valid_completed, True)
                sr_unseen, plw_sr_unseen, gc_unseen, plw_gc_unseen = count_results(valid_completed, False)
                valid_summary_dict = {
                    'epoch': valid_epoch, 
                    'time': valid_time - train_start_time,
                    'sr_seen': sr_seen,
                    'plw_sr_seen': plw_sr_seen,
                    'gc_seen': gc_seen,
                    'plw_gc_seen': plw_gc_seen,
                    'sr_unseen': sr_unseen,
                    'plw_sr_unseen': plw_sr_unseen,
                    'gc_unseen': gc_unseen,
                    'plw_gc_unseen': plw_gc_unseen,
                }

                stop_monitor(monitor)

                valid_summary_list = [valid_epoch, valid_time - train_start_time, sr_seen, plw_sr_seen, gc_seen, 
                                      plw_gc_seen, sr_unseen, plw_sr_unseen, gc_unseen, plw_gc_unseen]
                if self.args.episodes_per_epoch > 0:
                    valid_summary_list += [
                        sum(total_rewards) / len(total_rewards), 
                        sum(value_losses) / len(value_losses), 
                        sum(policy_losses) / len(policy_losses)
                    ]
                else:
                    valid_summary_list += [None, None, None]
                if self.args.batches_per_epoch > 0:
                    valid_summary_list += [sum(total_train_loss) / len(total_train_loss)]
                else:
                    valid_summary_list += [None]

                # Log results
                with open(valid_results_csv, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(valid_summary_list)
                print()
                print("Validation Epoch: {}-{} Time: {:.0f}-{:.0f}".format(valid_epoch, epoch, valid_time - train_start_time, time.time() - train_start_time))
                print("\tsr_seen: {:.2f} plw_sr_seen: {:.2f} gc_seen: {:.2f} plw_gc_seen: {:.2f}".format(sr_seen, plw_sr_seen, gc_seen, plw_gc_seen))
                print("\tsr_unseen: {:.2f} plw_sr_unseen: {:.2f} gc_unseen: {:.2f} plw_gc_unseen: {:.2f}".format(sr_unseen, plw_sr_unseen, gc_unseen, plw_gc_unseen))
                print()

                new_bests = []
                new_bests.append('sr_seen' if best_results['sr_seen'] is None or sr_seen > best_results['sr_seen'] else None)
                new_bests.append('plw_sr_seen' if best_results['plw_sr_seen'] is None or plw_sr_seen > best_results['plw_sr_seen'] else None)
                new_bests.append('gc_seen' if best_results['gc_seen'] is None or gc_seen > best_results['gc_seen'] else None)
                new_bests.append('plw_gc_seen' if best_results['plw_gc_seen'] is None or plw_gc_seen > best_results['plw_gc_seen'] else None)
                new_bests.append('sr_unseen' if best_results['sr_unseen'] is None or sr_unseen > best_results['sr_unseen'] else None)
                new_bests.append('plw_sr_unseen' if best_results['plw_sr_unseen'] is None or plw_sr_unseen > best_results['plw_sr_unseen'] else None)
                new_bests.append('gc_unseen' if best_results['gc_unseen'] is None or gc_unseen > best_results['gc_unseen'] else None)
                new_bests.append('plw_gc_unseen' if best_results['plw_gc_unseen'] is None or plw_gc_unseen > best_results['plw_gc_unseen'] else None)

                for filename in new_bests:
                    if filename is None:
                        continue
                    fsave = os.path.join(self.args.dout, 'best_{}.pth'.format(filename))
                    torch.save({
                        'metric': valid_summary_dict,
                        'model': self.state_dict(),
                        'optim': optimizer.state_dict(),
                        'args': self.args,
                        'vocab': self.vocab,
                    }, fsave)
                    fbest = os.path.join(self.args.dout, 'best_{}.json'.format(filename))
                    with open(fbest, 'wt') as f:
                        json.dump(valid_completed, f, indent=2)
                    best_results[filename] = valid_summary_dict[filename]
                
                # monitor resource usage
                monitor = start_monitor(path=self.args.dout, note=f"Validation restart epoch={_}")

                # Restart validation
                valid_model.load_state_dict(self.state_dict())
                for traj in valid_seen:
                    valid_task_queue.put((traj, True))
                for traj in valid_unseen:
                    valid_task_queue.put((traj, False))
                valid_completed = []
                valid_time = time.time()
                valid_epoch = epoch

                stop_monitor(monitor)

            # Save the latest checkpoint
            if self.args.save_every_epoch:
                fsave = os.path.join(self.args.dout, 'epoch_{}.pth'.format(epoch))
            else:
                fsave = os.path.join(self.args.dout, 'latest.pth')
            torch.save({
                'metric': dict(),
                'model': self.state_dict(),
                'optim': optimizer.state_dict(),
                'args': self.args,
                'vocab': self.vocab,
            }, fsave)

        for _ in range(len(rollout_processes)):
            rollout_task_queue.put(None)
        for _ in range(len(valid_processes) // 2):
            valid_task_queue.put(None)
        for p in rollout_processes:
            p.join()
        for p in valid_processes:
            p.join()
    

    @classmethod
    def run_rollouts(cls, model, task_queue, results, args, validation=False):
        env = ThorEnv()

        while True:
            if validation:
                task, seen = task_queue.get()
            else:
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
                pred = model.sample_pred(out, greedy=validation)

                # monitor resource usage
                monitor = start_monitor(path=args.dout, note="validation" if validation else "rollout" + f" step={num_steps}")


                # # check if <<stop>> was predicted
                # if pred['action_low'] == "<<stop>>":
                #     print("\tpredicted STOP")
                #     break

                # get action and mask
                action = pred['action_low']
                mask = pred['action_low_mask'] if cls.has_interaction(action) else None

                # use predicted action and mask (if available) to interact with the env
                t_success, _, _, err, _ = env.va_interact(action, interact_mask=mask, smooth_nav=args.smooth_nav, debug=args.debug)

                if not t_success:
                    fails += 1
                    if fails >= args.max_fails:
                        break

                # next time-step
                reward, done = env.get_transition_reward()
                total_reward += reward
                num_steps += 1

                if not validation:
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

                stop_monitor(monitor)

            if validation:
                # check if goal was satisfied
                goal_satisfied = env.get_goal_satisfied()

                # goal_conditions
                pcs = env.get_goal_conditions_met()
                goal_condition_success_rate = pcs[0] / float(pcs[1])

                # SPL
                path_len_weight = len(traj_data['plan']['low_actions'])
                s_spl = (1 if goal_satisfied else 0) * min(1., path_len_weight / float(num_steps))
                pc_spl = goal_condition_success_rate * min(1., path_len_weight / float(num_steps))

                # path length weighted SPL
                plw_s_spl = s_spl * path_len_weight
                plw_pc_spl = pc_spl * path_len_weight

                # log success/fails
                log_entry = {'trial': traj_data['task_id'],
                            'type': traj_data['task_type'],
                            'repeat_idx': int(r_idx),
                            'seen': seen,
                            'goal_instr': traj_data['turk_annotations']['anns'][r_idx]['task_desc'],
                            'goal_satisfied': goal_satisfied,
                            'completed_goal_conditions': int(pcs[0]),
                            'total_goal_conditions': int(pcs[1]),
                            'goal_condition_success': float(goal_condition_success_rate),
                            'success_spl': float(s_spl),
                            'path_len_weighted_success_spl': float(plw_s_spl),
                            'goal_condition_spl': float(pc_spl),
                            'path_len_weighted_goal_condition_spl': float(plw_pc_spl),
                            'path_len_weight': int(path_len_weight),
                            'reward': float(total_reward)}
                results.put(log_entry)
            else:
                results.put(curr_rollout)
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
        for i in range(0, len(data), batch_size):
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
        # print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

        # setup task for reward
        env.set_task(traj_data, args, reward_type=reward_type)
