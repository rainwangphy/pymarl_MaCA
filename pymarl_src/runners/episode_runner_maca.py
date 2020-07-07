from functools import partial

import numpy as np

from pymarl_src.components.episode_buffer import EpisodeBatch
# from pymarl_src.envs import REGISTRY as env_REGISTRY

import copy

from agent.fix_rule.agent import Agent
from interface import Environment

RENDER = True
MAP_PATH = 'maps/1000_1000_fighter10v10.map'
DETECTOR_NUM = 0
FIGHTER_NUM = 10
COURSE_NUM = 16
ATTACK_IND_NUM = (DETECTOR_NUM + FIGHTER_NUM) * 2 + 1  # long missile attack + short missile attack + no attack
ACTION_NUM = COURSE_NUM * ATTACK_IND_NUM  # plus one action: no attack


# LEARN_INTERVAL = 100


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # create blue agent
        self.blue_agent = Agent()
        # get agent obs type
        red_agent_obs_ind = 'simple'
        blue_agent_obs_ind = self.blue_agent.get_obs_ind()
        # make env
        self.env = Environment(MAP_PATH, red_agent_obs_ind, blue_agent_obs_ind, render=RENDER)

        # get map info
        size_x, size_y = self.env.get_map_size()
        red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = self.env.get_unit_num()
        # set map info to blue agent
        self.blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)

        # self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = 5000  # Following the number used in MaCA
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):

        # env_info = {"state_shape": self.get_state_size(),
        #             "obs_shape": self.get_obs_size(),
        #             "n_actions": self.get_total_actions(),
        #             "n_agents": self.n_agents,
        #             "episode_limit": self.episode_limit}

        env_info = {
            "state_shape": 1000 * 1000 * 3,
            "obs_shape": 100 * 100 * 3,
            "n_actions": ACTION_NUM,
            "n_agents": FIGHTER_NUM,
            "episode_limit": self.episode_limit

        }
        return env_info

    def save_replay(self):
        # self.env.save_replay()
        return

    def close_env(self):
        # self.env.close()
        return

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def env_get_state(self):
        return

    # def env_get_avail_actions(self):
    #     red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = self.env.get_unit_num()
    #     red_obs_dict, blue_obs_dict = self.env.get_obs()
    #     available_action = np.zeros(red_fighter_num, ACTION_NUM)
    #
    #     return

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        while not terminated:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = self.env.get_unit_num()
            red_obs_dict, blue_obs_dict = self.env.get_obs()
            # get action
            # get blue action
            blue_detector_action, blue_fighter_action = self.blue_agent.get_action(blue_obs_dict, step_cnt=self.t)
            red_detector_action = []

            red_fighter_action = []
            obs_list = []
            avail_actions = []
            action_list = []  # TODO: obtained based on the alive or not alive
            # get red action
            obs_got_ind = [False] * red_fighter_num
            for y in range(red_fighter_num):
                true_action = np.array([0, 1, 0, 0], dtype=np.int32)
                tmp_img_obs = red_obs_dict['fighter'][y]['screen']
                tmp_img_obs = tmp_img_obs.transpose(2, 0, 1)
                tmp_info_obs = red_obs_dict['fighter'][y]['info']
                obs_list.append({'screen': copy.deepcopy(tmp_img_obs), 'info': copy.deepcopy(tmp_info_obs)})
                # print(tmp_img_obs)
                if red_obs_dict['fighter'][y]['alive']:
                    obs_got_ind[y] = True
                    # TODO: obtain action from MAC
                    tmp_action = fighter_model.choose_action(tmp_img_obs)
                    action_list.append(tmp_action)
                    # action formation
                    true_action[0] = int(360 / COURSE_NUM * int(tmp_action[0] / ATTACK_IND_NUM))
                    true_action[3] = int(tmp_action[0] % ATTACK_IND_NUM)
                else:
                    tmp_action = ACTION_NUM - 1  # if the fighter is dead, it cannot execute any action, so no attack
                    action_list.append(tmp_action)

                red_fighter_action.append(true_action)
            red_fighter_action = np.array(red_fighter_action)

            pre_transition_data = {
                "state": [self.env_get_state()],
                "avail_actions": [avail_actions],
                "obs": [obs_list]
            }

            self.batch.update(pre_transition_data, ts=self.t)
            # step
            # reward, terminated, env_info = self.env.step(actions[0])
            self.env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
            # get reward
            red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = self.env.get_reward()
            # detector_reward = red_detector_reward + red_game_reward
            # fighter_reward = red_fighter_reward + red_game_reward
            episode_return += red_fighter_reward

            post_transition_data = {
                "actions": action_list,
                "reward": [(red_fighter_reward,)],
                "terminated": [(self.env.get_done(),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        # last_data = {
        #     "state": [self.env_get_state()],
        #     "avail_actions": [],
        #     "obs": [self.env.get_obs()]
        # }
        # self.batch.update(last_data, ts=self.t)
        #
        # # Select actions in the last stored state
        # actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # self.batch.update({"actions": actions}, ts=self.t)

        # cur_stats = self.test_stats if test_mode else self.train_stats
        # cur_returns = self.test_returns if test_mode else self.train_returns
        # log_prefix = "test_" if test_mode else ""
        # cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        # cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        # cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        # cur_returns.append(episode_return)
        #
        # if test_mode and (len(self.test_returns) == self.args.test_nepisode):
        #     self._log(cur_returns, cur_stats, log_prefix)
        # elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
        #     self._log(cur_returns, cur_stats, log_prefix)
        #     if hasattr(self.mac.action_selector, "epsilon"):
        #         self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
        #     self.log_train_stats_t = self.t_env

        return self.batch
    #
    # def _log(self, returns, stats, prefix):
    #     self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
    #     self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
    #     returns.clear()
    #
    #     for k, v in stats.items():
    #         if k != "n_episodes":
    #             self.logger.log_stat(prefix + k + "_mean", v / stats["n_episodes"], self.t_env)
    #     stats.clear()
