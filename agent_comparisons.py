import os
from rl_baselines_zoo.utils import create_test_env, ALGOS


def load_agents(params, agents_list):
    agents = []
    environment = create_test_env(params.env, n_envs=params.n_envs, is_atari=params.is_atari,
                                  stats_path=params.stats_path, seed=0, log_dir=params.log_dir,
                                  should_render=not params.no_render, hyperparams=params.hyperparams)
    for a in agents_list:
        model_path = os.path.join(params.agents_dir, a, params.env + ".pkl")
        agents.append(ALGOS[a].load(model_path, env=environment))
    return agents


def get_agents_comparison(params, method):
    pass


def compare_agents(args):
    agent_names = ['a2c', 'ppo2', 'acktr', 'dqn']
    agents = load_agents(args, agent_names)
    get_agents_comparison(agents)
