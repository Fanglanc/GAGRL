import os
import setproctitle
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app
from absl import flags

from khrylib.utils import *
from transit.utils.config import Config
from transit.agents.transit_agent import TransitExpansionAgent

flags.DEFINE_string('root_dir', 'result' , 'Root directory for writing '
                                                                      'logs/summaries/checkpoints.')
flags.DEFINE_string('city_name', 'beijing', 'data_dir')
flags.DEFINE_string('cfg', '4linesml3', 'Configuration file of rl training.')
flags.DEFINE_bool('tmp', False, 'Whether to use temporary storage.')
flags.DEFINE_bool('infer', False, 'Train or Infer.')
flags.DEFINE_bool('visualize', False, 'visualize plan.')
flags.DEFINE_enum('agent', 'gagrl', ['gagrl','rl'],'Agent type.')
flags.DEFINE_integer('num_threads', 1, 'The number of threads for sampling trajectories.')
flags.DEFINE_bool('use_nvidia_gpu', True, 'Whether to use Nvidia GPU for acceleration.')
flags.DEFINE_integer('gpu_index', 0,'GPU ID.')
flags.DEFINE_integer('global_seed', 1, 'Used in env and weight initialization, does not impact action sampling.')
flags.DEFINE_string('iteration', '0', 'The start iteration. Can be number or best. If not 0, the agent will load from '
                                      'a saved checkpoint.')
flags.DEFINE_bool('restore_best_rewards', True, 'Whether to restore the best rewards from a saved checkpoint. '
                                                'True for resume training. False for finetune with new reward.')

FLAGS = flags.FLAGS


def train_one_iteration(agent: TransitExpansionAgent, iteration: int) -> None:
    """Train one iteration"""
    agent.optimize(iteration)
    if agent.cfg.agent not in ['random','greedy']:
        agent.save_checkpoint(iteration)

    """clean up gpu memory"""
    torch.cuda.empty_cache()


def main_loop(_):

    setproctitle.setproctitle(f'metro_{FLAGS.cfg}_{FLAGS.city_name}_{FLAGS.global_seed}')

    cfg = Config(FLAGS.cfg, FLAGS.city_name, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    if FLAGS.use_nvidia_gpu and torch.cuda.is_available():
        device = torch.device('cuda', index=FLAGS.gpu_index)
    else:
        raise RuntimeError('cpu')
        device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(FLAGS.gpu_index)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration

    """create agent"""
    agent = TransitExpansionAgent(cfg=cfg, dtype=dtype, device=device, num_threads=FLAGS.num_threads,
                               training=True, checkpoint=checkpoint, restore_best_rewards=FLAGS.restore_best_rewards)
    
    if FLAGS.infer:
        agent.infer(visualize=FLAGS.visualize)
    else:
        start_iteration = agent.start_iteration
        for iteration in range(start_iteration, cfg.max_num_iterations):
            train_one_iteration(agent, iteration)

    agent.logger.info('training done!')


if __name__ == '__main__':

    app.run(main_loop)
    # Empty cache
    torch.cuda.empty_cache()
