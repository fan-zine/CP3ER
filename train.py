import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
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
from wandblogger import WandbLogger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from datetime import datetime

from omegaconf import OmegaConf

import numpy as np

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)

        if self.cfg.use_wb:
            wandb_config = OmegaConf.to_container(self.cfg, resolve=True)
            self.wandblogger = WandbLogger(self.cfg.task_name,config=wandb_config, seed=self.cfg.seed)
            # self.wandblogger = WandbLogger(self.cfg.task_name,self.cfg.algo,config=wandb_config, seed=self.cfg.seed)

        # create envs
        # dmc.make函数创建训练环境和评估环境，task_name指定任务名称，frame_stack指定连续帧数，action_repeat指定动作重复次数，seed用于设置随机种子
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)

        # 获取训练环境的动作空间规范，提取动作的最小值和最大值
        action_spec = self.train_env.action_spec()

        self.action_low = action_spec.minimum
        self.action_high = action_spec.maximum

        # create replay buffer
        # 定义回放缓冲区的数据规范，包括观测数据、动作数据、奖励数据和折扣因子数据
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # 初始化回放缓冲区存储
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        # 创建回放缓冲区加载器，用于从回放缓冲区中采样
        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount,
            task_name=self.cfg.task_name, sample_alpha=self.cfg.sample_alpha)
        self._replay_iter = None

        # 初始化评估和训练视频记录器
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        # 初始化执行步数、完成的回合数、累计奖励
        step, episode, total_reward = 0, 0, 0
        # 当评估回合数达到10次时终止评估
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            # 重置评估环境，获取初始状态(包含观测、奖励等信息)
            time_step = self.eval_env.reset()
            # 仅在第一个评估回合启用视频录制，初始化录制器
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            # 在单个回合内逐步执行，直到环境返回终止信号
            while not time_step.last():
                # 禁用梯度计算，将Agent切换到评估模式，根据当前观测选择动作，无探索
                with torch.no_grad(), utils.eval_mode(self.agent):
                    # 调用Agent的act方法，根据当前观测生成一个动作
                    action = self.agent.act(time_step.observation)

                # 执行生成的动作，获取下一个时间步，这里包含了新的观测、奖励等信息
                time_step = self.eval_env.step(action)
                # 存储时间步数据到回放缓冲区
                self.replay_storage.add(time_step)
                # 记录视频
                self.video_recorder.record(self.eval_env)
                # 累计奖励和步数
                total_reward += time_step.reward
                step += 1

            # episode结束，回合加一
            episode += 1

            # 如果全局帧数超过30000000，则保存视频到文件
            if self.global_frame > 30000000:
                self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            # 记录平均回合奖励、平均回合长度、当前全局回合数、当前全局步数
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

        if self.cfg.use_wb:
            # 如果启用了wandb，则记录以下评估指标到wandb平台
            self.wandblogger.scalar_summary("eval/episode_reward",total_reward / episode, self.global_frame)
            self.wandblogger.scalar_summary('eval/episode_length', step * self.cfg.action_repeat / episode, self.global_frame)
            self.wandblogger.scalar_summary("eval/episode",self.global_episode, self.global_frame)


    def train(self):
        # print(self.cfg.num_train_frames)
        # assert self.cfg.num_train_frames != 3100000
        # predicates
        # 当全局步数乘以动作重复次数达到指定的训练帧数时停止训练
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        # 当全局步数乘以动作重复次数达到指定的预热帧数时，停止随机探索
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        # 当全局步数乘以动作重复次数达到指定评估间隔帧数时，进行评估
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        # 初始化当前回合的步数和累积奖励
        episode_step, episode_reward = 0, 0
        # 重置训练环境，获取初始时间步
        time_step = self.train_env.reset()
        # 将初始时间步数据添加到回放缓冲区
        self.replay_storage.add(time_step)
        # 初始化视频训练记录器
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        # 启动主循环训练，循环条件是当前全局步数是否达到指定的训练步数
        while train_until_step(self.global_step):
            # 如果当前时间步是episode的最后一步
            if time_step.last():
                # episode加一步
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                # 如果metrics不是None，也就是Agent已经经过更新，则记录训练指标
                if metrics is not None:
                    # log stats
                    # 计算并记录每秒帧数、总时间、回合奖励、回合长度、当前回合数、回放缓冲区大小和全局步数
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                    # 记录日志
                    if self.global_step % 100 == 0 and self.cfg.use_wb:
                        self.wandblogger.scalar_summary("train/fps", episode_frame / elapsed_time, self.global_frame)
                        self.wandblogger.scalar_summary('train/total_time', total_time, self.global_frame)
                        self.wandblogger.scalar_summary('train/episode_reward', episode_reward, self.global_frame)
                        self.wandblogger.scalar_summary('train/episode_length', episode_frame, self.global_frame)
                        self.wandblogger.scalar_summary('train/episode', self.global_episode, self.global_frame)
                        self.wandblogger.scalar_summary('train/buffer_size', len(self.replay_storage), self.global_frame)
                        self.wandblogger.scalar_summary('train/step', self.global_step, self.global_frame)

                # reset env
                # 重置训练环境，获取新的初始时间步
                time_step = self.train_env.reset()
                # 将新的初始时间步数据添加到回放缓冲区
                self.replay_storage.add(time_step)
                # 初始化训练视频记录器
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                # 如果配置中启用了保存快照，则保存当前训练状态的快照
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                # 重置当前回合的步数和累计奖励
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            # 检查是否达到评估间隔
            if eval_every_step(self.global_step):
                # 记录评估的总时间
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.cfg.use_wb:
                    self.wandblogger.scalar_summary("eval/eval_total_time", self.timer.total_time(), self.global_frame)
                # 进行评估
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                # 策略网络生成动作
                action = self.agent.act(time_step.observation)
                # 初始探索阶段使用随机动作，1万步内
                if self.global_frame < self.agent.num_expl_steps: #  default is 10k
                    # 用均匀随机动作替代策略动作，促进环境探索
                    action = np.random.uniform(self.action_low, self.action_high, size=action.shape)
                    action = action.astype(np.float32)
                    # action = self.train_env.action_spec.sample()
                    
                # action = self.agent.ou_noise.get_action(action, self.global_frame)  # exporation
                

            # try to update the agent
            # 如果已经过了纯探索阶段，可以开始正式训练了
            if not seed_until_step(self.global_step):
                # 就从replay buffer中采样数据并更新agent
                # replay_iter是一个迭代器，不断从replay buffer中采样batch数据
                metrics = self.agent.update(self.replay_iter, self.global_step)
                # 每隔100步，记录agent更新返回的性能指标
                if self.global_step % 100 == 0:
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')
                    if self.cfg.use_wb:
                        self.wandblogger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            # 执行动作，获取下一个时间步的信息
            time_step = self.train_env.step(action)
            # 累积奖励
            episode_reward += time_step.reward
            # 将时间步加入经验回放缓冲区
            self.replay_storage.add(time_step)
            # 记录当前观测到视频中
            self.train_video_recorder.record(time_step.observation)
            # 增加当前回合步数和全局步数
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()