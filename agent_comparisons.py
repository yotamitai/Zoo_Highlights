import os
import pandas as pd
import numpy as np
import copy
from itertools import combinations
from highlights_state_selection import compute_states_importance
from rl_baselines_zoo.utils import create_test_env, ALGOS
from tools import pickle_load, ACTION_MEANING


def load_agents_params(params, agents_list):
    agents, states, traces, trajectories, dataframes = {}, {}, {}, {}, {}
    # states['all_states'] = {}

    environment = create_test_env(params.env, n_envs=params.n_envs, is_atari=params.is_atari,
                                  stats_path=params.stats_path, seed=0, log_dir=params.log_dir,
                                  should_render=not params.no_render, hyperparams=params.hyperparams)
    for a in agents_list:
        """load agent model"""
        model_path = os.path.join(params.agents_dir, a, params.env + ".pkl")
        agents[a] = ALGOS[a].load(model_path, env=environment)
        """load agent states traces and trajectories"""
        params_path = os.path.join(params.stt_dir, a)
        for file in os.listdir(params_path):
            if file.startswith('States'):
                states[a] = pickle_load(os.path.join(params_path, file))
            elif file.startswith('Traces'):
                traces[a] = pickle_load(os.path.join(params_path, file))
            else:
                trajectories[a] = pickle_load(os.path.join(params_path, file))
        # states['all_states'] = {**states['all_states'], **states[a]}
        # The number of "same" states between algos is very small because the id of a state is based on the
        # full observation, therefore taking into account the current score.

        """importance dataframes"""
        data = {
            'state': list(states[a].keys()),
            'q_values': [x.observed_actions for x in states[a].values()]
        }
        df = pd.DataFrame(data)
        # top "important_state_percentage" of the states sorted by importance
        dataframes[a] = compute_states_importance(df, compare_to=params.state_importance).sort_values(
            ['importance'], ascending=False).head(int(df.shape[0] * params.important_state_percentage))

    return agents, states, traces, trajectories, dataframes


def get_differences(a1_q_values, a2_q_values):
    # importance by sum of diff in q-values per action - normalized by 2 (max disagreement), values between 0-1
    diff_all_actions = sum(
        [abs(a2_q_values[i] - a1_q_values[i]) for i in range(len(a2_q_values))]) / 2

    # importance by chosen action:
    a1_chose_action = np.argmax(a1_q_values)
    a2_chose_action = np.argmax(a2_q_values)
    diff_chosen_actions = (max(a1_q_values) - a1_q_values[a2_chose_action] \
                           + max(a2_q_values) - a2_q_values[a1_chose_action]) / 2

    # importance by diff in sorted order:
    a1_i_vals = sorted((list(enumerate(a1_q_values))), key=lambda x: x[1], reverse=True)
    a2_i_vals = sorted((list(enumerate(a2_q_values))), key=lambda x: x[1], reverse=True)
    top = 5
    a1_top_actions = [x[0] for x in a1_i_vals[:top]]
    a2_top_actions = [x[0] for x in a2_i_vals[:top]]
    diff_sorted_actions = 1 - (sum([1 for x in a1_top_actions if x in a2_top_actions])) / top

    # diff by action description word context
    a1_action_name = ACTION_MEANING[a1_chose_action]
    a2_action_name = ACTION_MEANING[a2_chose_action]
    words = np.zeros(6)
    word_dict = {0: 0, 1: 0, 2: 0}
    for a in [a1_action_name, a2_action_name]:
        word_list = np.zeros(6)
        word_list[0] = 1 if "FIRE" in a else 0
        word_list[1] = 1 if "UP" in a else 0
        word_list[2] = 1 if "DOWN" in a else 0
        word_list[3] = 1 if "LEFT" in a else 0
        word_list[4] = 1 if "RIGHT" in a else 0
        word_list[5] = 1 if "NOOP" in a else 0
        words += word_list
    unique, counts = np.unique(words, return_counts=True)
    word_dict = {**word_dict, **dict(zip(unique, counts))}
    disagrees = word_dict[1] / 2  # disagreements are always both ways, so we divide it by 2
    agrees = word_dict[2]
    diff_action_name = disagrees / (agrees + disagrees)

    return diff_all_actions, diff_chosen_actions, diff_sorted_actions, diff_action_name


def same_state_comparison(models, agent1, agent2, importance, states):
    """compare agent predictions for the same state"""

    for a1,a2 in [(agent1,agent2),(agent2,agent1)]:
        avg_diff = []
        for _, row in importance[a1].iterrows():
            a1_state_q_values = row['q_values']
            observation = states[a1][row['state']].observation
            a2_state_q_values = models[a2].action_probability(observation)[0]

            """Difference"""
            differences = get_differences(a1_state_q_values, a2_state_q_values)
            avg_diff.append(np.average(np.array(differences)))  # average between all diff measures
        importance[a1]['avg_diff:'+a2] = avg_diff
    return





def compare_agents(args):
    agent_names = ['a2c', 'ppo2' , 'acktr', 'dqn']
    agent_models, agent_states, agent_traces, agent_trajectories, importance_dfs = load_agents_params(args, agent_names)

    """compare by disagreement on action prediction for the same state"""
    for a1, a2 in combinations(agent_names, 2):
        same_state_comparison(agent_models, a1, a2, importance_dfs, agent_states)

    print()
