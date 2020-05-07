import random
import xxhash
from rl_baselines_zoo.utils import ALGOS, create_test_env
from tools import Trace, State, pickle_save, pickle_load, mapped_actions, find_features_layer
import os


def load_agent(params):
    environment = create_test_env(params.env, n_envs=params.n_envs, is_atari=params.is_atari,
                                  stats_path=params.stats_path,
                                  seed=0, log_dir=params.log_dir, should_render=not params.no_render,
                                  hyperparams=params.hyperparams)
    agent = ALGOS[params.algo].load(params.model_path, env=environment)
    return environment, agent


def get_traces(args):
    execution_traces, states_dictionary = [], {}
    shuffled_list = list(range(0, args.random_noop_range))
    random.shuffle(shuffled_list)
    environment, agent = load_agent(args)
    features_layer = find_features_layer(environment, agent)
    random_noop = 0
    for i in range(args.num_traces):
        if args.random_noop_init:
            random_noop = shuffled_list[i] + args.built_in_noop
        get_single_trace(agent, environment, execution_traces, states_dictionary, features_layer, args, random_noop)
        environment = create_test_env(args.env, n_envs=args.n_envs, is_atari=args.is_atari,
                                      stats_path=args.stats_path,
                                      seed=0, log_dir=args.log_dir, should_render=not args.no_render,
                                      hyperparams=args.hyperparams)
        environment.close()
    """save"""
    pickle_save(execution_traces, args.traces_file)
    if args.verbose: print(f"Traces Saved")
    pickle_save(states_dictionary, args.state_file)
    if args.verbose: print(f"States Saved")
    return execution_traces, states_dictionary


def get_single_trace(model, env, agent_traces, states_dict, features_layer_name, args, n_noop_steps):
    """run agent until a life is lost"""

    trace = Trace()
    obs = env.reset()
    for j in range(args.max_trace_timesteps):
        state_img = env.render(mode='rgb_array')[args.crop_top:args.crop_bottom]  # crop top to remove "additional life" section
        state_id = xxhash.xxh64(state_img, seed=0).hexdigest()

        if state_id not in states_dict.keys():
            """get current state action probabilities == q-values"""
            state_q_values = model.action_probability(obs)[0]
            """get current state features from agent model"""
            features = model.get_parameters()[features_layer_name]
            """create State object"""
            states_dict[state_id] = State(state_id, obs, state_q_values, features, state_img)

        """get next action from model"""
        if args.random_noop_init and j < n_noop_steps:
            action = [0]
        elif args.rand_step and j % args.rand_step == 0:
            # Random action
            action = [env.action_space.sample()]
        else:
            action, _ = model.predict(obs, deterministic=False)

        if args.map_action_reduction:
            action = mapped_actions[action[0]]

        obs, reward, done, infos = env.step(action)
        if not j:
            # first action - get the max num of lives
            start_n_lives = infos[0]["ale.lives"]
        if not args.no_render:
            env.render('human')

        """Check if trace is done"""
        if infos is not None:
            if args.single_life_trace:
                if infos[0].get('ale.lives') < start_n_lives:
                    """a life has ended -> end of trace"""
                    agent_traces.append(trace)
                    return
            else:
                if done:
                    """The game has ended"""
                    agent_traces.append(trace)
                    return

        """Add step to trace"""
        trace.reward_sum += reward[0]
        trace.length += 1
        trace.obs.append(obs), trace.rewards.append(reward)
        trace.dones.append(done), trace.infos.append(infos)
        trace.actions.append(action), trace.states.append(state_id)

        # TODO 1 - how are the rewards calculated?
        # TODO 2 - why is there a difference between the number of steps and episode_infos['l']?
        # For atari the return reward is not the atari score so we have to get it from the infos dict
        # episode_infos = infos[0].get('episode')
        # if episode_infos is not None:
        #     print(f"Atari Episode Score: {episode_infos['r']:10.2f}")
        #     print(f"Atari Episode Length: {episode_infos['l']:9}")
        #     trace.game_score = episode_infos['r']
    return


def load_traces(args):
    t = pickle_load(args.traces_file)
    if args.verbose: print(f"Traces Loaded")
    s = pickle_load(args.state_file)
    if args.verbose: print(f"States Loaded")
    return t, s
