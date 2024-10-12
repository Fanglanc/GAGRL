import logging
import math
import copy
import pickle
from pprint import pprint
from typing import Tuple, Dict, List, Text, Callable
from functools import partial

import numpy as np
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from transit.envs.transit import Transit
from transit.utils.config import Config

import time


class InfeasibleActionError(ValueError):
    """An infeasible action were passed to the env."""

    def __init__(self, action, mask):
        """Initialize an infeasible action error.

        Args:
          action: Infeasible action that was performed.
          mask: The mask associated with the current observation. mask[action] is
            `0` for infeasible actions.
        """
        super().__init__(self, action, mask)
        self.action = action
        self.mask = mask

    def __str__(self):
        return 'Infeasible action ({}) when the mask is ({})'.format(
            self.action, self.mask)


def reward_info_function(transit: Transit, stage) -> Tuple[float, Dict]:
    totoal_cost = transit.get_cost()
    reward = transit.get_reward()
    add_od = 0
    total_od = transit.get_od()
    return reward, {'reward': reward, 'cost': totoal_cost, 'od': total_od,'add_od': add_od}


class TransitEnv:
    """ Environment for urban planning."""
    FAILURE_REWARD = -4.0
    INTERMEDIATE_REWARD = -4.0

    def __init__(self,
                 cfg: Config,
                 is_eval: bool = False,
                 reward_info_fn=reward_info_function):

        self.cfg = cfg
        self._is_eval = is_eval
        self._frozen = False
        self._action_history = []
        self._transit =  self.load_graph(cfg)
        self._copy_transit = copy.deepcopy(self._transit)
        self._reward_info_fn = partial(reward_info_fn)

        self._done = False

    def load_graph(self,cfg):
        origin = cfg.build.get('origin')
        destination = cfg.build.get('destination')
        shape_param = cfg.build.get('shape')
        cost = cfg.build.get('cost_per')
        corridor = cfg.build.get('corridor')
        max_line = cfg.build.get('max_line')
        pre_line = cfg.build.get('pre_line')
        expand = cfg.build.get('expand')
        min_num = cfg.build.get('min_num')
        self.budget = cfg.build.get('budget')
        self.ax0 = None
        self.ax1 = None

        file_data = []
        if corridor != None:
            all_files = ['city_graph', 'region_processed', 'od_pair', 'line_info']
            for file_name in all_files:
                with open('/ProjectFolder/GAGRL/data/{}/{}lines/{}.pickle'.format(cfg.city_name,pre_line,file_name), 'rb') as f:
                    file_data.append(pickle.load(f))

        m = Transit(file_data,self.budget,max_line,min_num,expand,cost,origin,destination,shape_param)

        return m

    def _set_cached_reward_info(self):
        """
        Set the cached reward.
        """
        if not self._frozen:
            self._cached_life_circle_reward = -1.0
            self._cached_greeness_reward = -1.0
            self._cached_concept_reward = -1.0

            self._cached_life_circle_info = dict()
            self._cached_concept_info = dict()

            self._cached_land_use_reward = -1.0
            self._cached_land_use_gdf = self.snapshot_land_use()

    def get_reward_info(self) -> Tuple[float, Dict]:
        return self._reward_info_fn(self._transit, self._stage)

    def eval(self):
        self._is_eval = True

    def train(self):
        self._is_eval = False

    def get_numerical_feature_size(self):
        return self._transit.get_numerical_dim()

    def get_node_dim(self):
        return self._transit.get_node_dim()
    
    def get_stage(self):
        if self._stage == 'build':
            return [1,0]
        elif self._stage == 'done':
            return [0,1]

    def _get_obs(self) -> List:
        numerical, node_feature, edge_distance, edge_od, node_mask = self._transit.get_obs()
        stage = self.get_stage()

        return [numerical, node_feature, edge_distance, edge_od, node_mask, stage]

    def add_station(self, action):
        self._transit.add_station_from_action(int(action))

    def snapshot_land_use(self):

        return self._transit.snapshot()
       
    def save_step_data(self):
        return
        self._transit.save_step_data()

    def failure_step(self, logging_str, logger):
        """
        Logging and reset after a failure step.
        """
        logger.info('{}: {}'.format(logging_str, self._action_history))
        info = {
            'road_network': -1.0,
            'life_circle': -1.0,
            'greeness': -1.0,
        }
        return self._get_obs(), self.FAILURE_REWARD, True, info

    def build(self,action):
        self._transit.add_station_from_action(int(action))

    def get_cost(self):
        return self._transit.get_cost()
    
    def fake_cost(self,action):
        return self._transit.fake_cost(int(action))

    def step(self, action,logger: logging.Logger) -> Tuple[List, float, bool, Dict]:
        
        if self._done:
            raise RuntimeError('Action taken after episode is done.')

        else:
            if self._stage == 'build':
                fake_cost = self.fake_cost(action)

                if self.get_cost() + fake_cost > self.budget:
                    self.transition_stage()
                else:
                    self.build(action)
                    self._action_history.append(int(action))

            if self._transit.get_done():
                self.transition_stage()

            reward, info = self.get_reward_info()
            if self._stage == 'done':
                self.save_step_data()

        return self._get_obs(), reward, self._done, info

    def reset(self,eval=False):
        t1 = time.time()
        self._transit.reset(eval)
        self._action_history = []
        self._set_stage()
        self._done = False

        return self._get_obs()

    def _set_stage(self):
        self._stage = 'build'

    def transition_stage(self):
        if self._stage == 'build':
            self._stage = 'done'
            self._done = True
        else:
            raise RuntimeError('Error stage!')
        
    def plot_and_save(self,
                          save_fig: bool = False,
                          path: Text = None,
                          show=False, final = None) -> None:
        """
        Plot and save the gdf.
        """
        self._transit.plot_transit(self.ax0, self.ax1, final)
        if save_fig:
            assert path is not None
            plt.savefig(path, format='svg', transparent=True)
        if show:
            plt.show()

        plt.cla()
        plt.close('all')

    def visualize(self,
                  save_fig: bool = False,
                  path: Text = None,
                  show=False, final=None) -> None:
        """
        Visualize the city plan.
        """
        self.plot_and_save(save_fig, path, show, final)
