import warnings
from collections import OrderedDict

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
import torch.nn.functional as F
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

import logging
from collections import defaultdict


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class EvalAction:

    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        self.eval_env = dmc.make(cfg.task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.eval_env.observation_spec(),
                                self.eval_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.eval_env.observation_spec(),
                      self.eval_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self):
        # evaluate every skill several episodes within different env (reset), eval_mode=True
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        # for skill_num in range(5):
        for skill_num in range(1):
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            meta = OrderedDict()
            step, episode, total_reward = 1, 0, 0
            skill = np.zeros(self.cfg.skill_dim, dtype=np.float32)

            skill[skill_num] = 1
            meta["skill"] = skill

            while eval_until_episode(episode):
                self.eval_env = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                         self.cfg.action_repeat, episode)
                time_step = self.eval_env.reset()
                self.video_recorder.init(self.eval_env, enabled=True)

                qs = []
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                                meta,
                                                step,
                                                eval_mode=True)
                    qs.append(time_step['observation'])
                    time_step = self.eval_env.step(action)
                    self.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    step += 1
                episode += 1

                self.video_recorder.save(f'{episode}epo_{skill_num}th-skill_{total_reward}_reward.mp4')

                self._summary(qs, skill_num, episode)

    def _summary(self, qs, skill_num, episode):
        # logging.info(f'the {episode}epo_{skill_num}th Q(z| tau) is: \n ')
        # for value in qs:
        #     logging.info(f"{value} .\n")
        # logging.info("\n")
        np.save(f"{self.cfg.agent.name}_{episode}_data", qs)

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = '../..' / snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            logging.info("loading model :{},cwd is {}".format(str(snapshot), str(Path.cwd())))
            if not snapshot.exists():
                logging.error("no such a pretrain model")
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload

        raise FileNotFoundError


@hydra.main(config_path='.', config_name='eval_action')
def main(cfg):
    from eval_action import EvalAction as ek
    root_dir = Path.cwd()
    workspace = ek(cfg)
    logging.basicConfig(filename="eval_q.log", encoding="utf-8", level=logging.DEBUG, format='')
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.eval()


if __name__ == '__main__':
    main()
