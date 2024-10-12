import os
from pprint import pprint

import pygad
import setproctitle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app
from absl import flags

from khrylib.utils import *
from transit.utils.config import Config
from transit.agents.transit_agent import TransitExpansionAgent


flags.DEFINE_string('root_dir', '/ProjectFolder/GAGRL/transit/result/', 'Root directory for writing '
                                                                      'logs/summaries/checkpoints.')
flags.DEFINE_string('cfg', '4linesml3', 'Configuration file of rl training.')
flags.DEFINE_string('city_name', 'beijing', 'City name.')
flags.DEFINE_bool('tmp', False,  'Whether to use temporary storage.')
flags.DEFINE_enum('agent', 'gagrl',
                  ['gagrl','rl'],
                  'Agent type.')
flags.DEFINE_bool('mean_action', True, 'Whether to use greedy strategy.')
flags.DEFINE_bool('visualize', False, 'Whether to visualize the planning process.')
flags.DEFINE_bool('only_road', False, 'Whether to only visualize road planning.')
flags.DEFINE_bool('save_video', False, 'Whether to save a video of the planning process.')
flags.DEFINE_integer('global_seed', 1, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('iteration', 'best', 'The start iteration. Can be number or best. If not 0, the agent will load from '
                                      'a saved checkpoint.')

FLAGS = flags.FLAGS


def main_loop(_):

    cfg = Config(FLAGS.cfg, FLAGS.city_name, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    """create agent"""
    agent = TransitExpansionAgent(cfg=cfg, dtype=dtype, device=device, num_threads=1,
                               training=False, checkpoint=checkpoint, restore_best_rewards=True)

    if FLAGS.only_road:
        agent.freeze_land_use()

    if FLAGS.agent != 'ga':
        agent.infer(num_samples=1, mean_action=FLAGS.mean_action, visualize=FLAGS.visualize,
                    save_video=FLAGS.save_video, only_road=FLAGS.only_road)
    else:

        best_solution, _ = agent.load_ga()
        _, plan = agent.fitness_ga(best_solution, num_samples=1, mean_action=FLAGS.mean_action, visualize=FLAGS.visualize)
        pprint(plan, indent=4, sort_dicts=False)


if __name__ == '__main__':
    flags.mark_flags_as_required([
        'cfg',
        'global_seed'
    ])
    app.run(main_loop)
