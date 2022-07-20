import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parameter.Parameter import Parameter
from parameter.private_config import *
from models.policy import Policy
from agent.Agent import EnvRemoteArray
from envs.nonstationary_env import NonstationaryEnv
import gym
from log_util.logger import Logger
import torch
from utils.replay_memory import MemoryNp
import numpy as np
import pickle

task_name = 'Hopper-v2-use_vrdc-1'


def inference(log_relative_path):
    parameter = Parameter(config_path=os.path.join(get_base_path(), 'log_file', log_relative_path))
    parameter.load_config()
    policy_config = Policy.make_config_from_param(parameter)
    # Define the env to test
    env = NonstationaryEnv(gym.make(parameter.env_name), log_scale_limit=ENV_DEFAULT_CHANGE)
    env_tasks = env.sample_tasks(0)
    deterministic = True
    training_agent = EnvRemoteArray(parameter=parameter, env_name=parameter.env_name,
                                    worker_num=1, seed=parameter.seed,
                                    deterministic=deterministic, use_remote=False, policy_type=Policy,
                                    history_len=parameter.history_length, env_decoration=NonstationaryEnv,
                                    env_tasks=env_tasks,
                                    use_true_parameter=parameter.use_true_parameter,
                                    non_stationary=False)
    if parameter.use_true_parameter:
        policy_config['ep_dim'] = training_agent.env_parameter_len
    policy = Policy(training_agent.obs_dim, training_agent.act_dim, **policy_config)
    policy.load(Logger.get_model_output_path(parameter), map_location=torch.device('cpu'))
    # parameter.load_config()
    print(policy.sample_hidden_state)
    finals = []
    replay_buffer = MemoryNp()
    for i in range(1000):
        '''
        Interact in the env for 1000 steps
        '''
        mem, logs = training_agent.sample1step(policy, random=False)
        replay_buffer.mem_push(mem)
        # print(i, mem.memory[0].done,
        #           [round(item, 3) for item in policy.ep_tensor.detach().cpu().numpy().tolist()[0]], policy.ep_tensor.abs().mean().item())
        if mem.memory[0].done[0]:
            finals.append([round(item, 3) for item in policy.ep_tensor.detach().cpu().numpy().tolist()[0]])
            print(i, mem.memory[0].done,
                  [round(item, 3) for item in policy.ep_tensor.detach().cpu().numpy().tolist()[0]])
            print('\n\n\n')
            replay_buffer.save_to_disk(os.path.join(get_base_path(), 'replay.pkl'))
            # np.save('1233.npy', replay_buffer, allow_pickle=True)
    print(*finals, sep='\n')


if __name__ == '__main__':
    inference(task_name)
