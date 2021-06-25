import argparse

from Environment import Environment
from Parameters import *
from Monte_carlo_control import Monte_carlo
from SARSA import SARSA
from Q_learning import Q_learning

parser = argparse.ArgumentParser(description="Parameters need to be input for training")
parser.add_argument('--agent', default=0)
parser.add_argument('--grid_size', default=4)
parser.add_argument('--num_epoch', default='10000')
args = parser.parse_args()

env = Environment(grid_size=args.grid_size)

if args.agent == 'mc':
    # Create a monte carlo agent
    monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)

    # Learning and updating Q table
    Q = monte_carlo.fv_mc_prediction(num_epoch=NUM_EPISODES)

    # write_Q_table(file_name='./Q_table/monte_carlo', Q = Q)

    # Test after training
    monte_carlo.test()

    # Remain visualization
    env.mainloop()

elif args.agent == 'sarsa':
    # Create a SARSA agent
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # write_Q_table(file_name='./Q_table/SARSA', Q = Q)

    # Learning and updating
    SARSA.train(num_epoch=NUM_EPISODES)

    # Test after training
    SARSA.test()

    # Remain visualization
    env.mainloop()

elif args.agent == 'ql':
    # Create a q learning agent
    Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # Learning and updating
    Q_table = Q_learning.train(num_epoch=NUM_EPISODES)

    # Test after training
    Q_learning.test()

    # Remain visualization
    env.mainloop()


