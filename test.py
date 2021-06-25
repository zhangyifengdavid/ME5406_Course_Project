import numpy as np
import matplotlib.pyplot as plt
import argparse

from Environment import Environment
from Parameters import *
from Monte_carlo_control import Monte_carlo
from SARSA import SARSA
from Q_learning import Q_learning


# Define line plot functions
# Episodes via steps
def plot_steps(steps, label):
    plt.figure()
    plt.plot(np.arange(len(steps[0])), steps[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(steps[1])), steps[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(steps[2])), steps[2], 'b', label=label[2], linewidth=1)
    plt.title('Episode vias Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend(loc='best')
    plt.show()


# Episodes via Costs
def plot_all_cost(all_cost, label):
    plt.figure()
    plt.plot(np.arange(len(all_cost[0])), all_cost[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(all_cost[1])), all_cost[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(all_cost[2])), all_cost[2], 'b', label=label[2], linewidth=1)
    plt.title('Episode via Cost')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.legend(loc='best')
    plt.show()


# Episodes via Accuracy
def plot_accuracy(accuracy, label):
    plt.figure()
    plt.plot(np.arange(len(accuracy[0])), accuracy[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(accuracy[1])), accuracy[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(accuracy[2])), accuracy[2], 'b', label=label[2], linewidth=1)
    plt.title('Episode via Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()


# Episodes via Average Rewards
def plot_average_rewards(Reward_list, label):
    plt.figure()
    plt.plot(np.arange(len(Reward_list[0])), Reward_list[0], 'r', label=label[0], linewidth=1)
    plt.plot(np.arange(len(Reward_list[1])), Reward_list[1], 'g', label=label[1], linewidth=1)
    plt.plot(np.arange(len(Reward_list[2])), Reward_list[2], 'b', label=label[2], linewidth=1)
    plt.title('Episode via Average rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average rewards')
    plt.legend(loc='best')
    plt.show()


# Define scatter plot functions
def plot_steps_scatter(steps, label):
    plt.figure()
    plt.scatter(np.arange(len(steps[0])), steps[0], alpha=0.8, s=1.5, c='r', label=label[0])
    plt.scatter(np.arange(len(steps[1])), steps[1], alpha=0.8, s=1.5, c='g', label=label[1])
    plt.scatter(np.arange(len(steps[2])), steps[2], alpha=0.8, s=1.5, c='b', label=label[2])
    plt.title('Episode via Steps')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend(loc='best')
    plt.show()


def plot_all_cost_scatter(all_cost, label):
    plt.figure()
    plt.scatter(np.arange(len(all_cost[0])), all_cost[0], label=label[0], alpha=0.8, s=2, c='r')
    plt.scatter(np.arange(len(all_cost[1])), all_cost[1], label=label[1], alpha=0.8, s=2, c='g')
    plt.scatter(np.arange(len(all_cost[2])), all_cost[2], label=label[2], alpha=0.8, s=2, c='b')
    plt.title('Episode via Cost')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.legend(loc='best')
    plt.show()


def plot_accuracy_scatter(accuracy, label):
    plt.figure()
    plt.scatter(np.arange(len(accuracy[0])), accuracy[0], alpha=0.8, s=1.5, c='r', label=label[0])
    plt.scatter(np.arange(len(accuracy[1])), accuracy[1], alpha=0.8, s=1.5, c='g', label=label[1])
    plt.scatter(np.arange(len(accuracy[2])), accuracy[2], alpha=0.8, s=1.5, c='b', label=label[2])
    plt.title('Episode via Accuracy')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.show()


def plot_average_rewards_scatter(Reward_list, label):
    plt.figure()
    plt.scatter(np.arange(len(Reward_list[0])), Reward_list[0], alpha=0.8, s=1.5, c='r', label=label[0])
    plt.scatter(np.arange(len(Reward_list[1])), Reward_list[1], alpha=0.8, s=1.5, c='g', label=label[1])
    plt.scatter(np.arange(len(Reward_list[2])), Reward_list[2], alpha=0.8, s=1.5, c='b', label=label[2])
    plt.title('Episode via Average rewards')
    plt.xlabel('Episode')
    plt.ylabel('Average rewards')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)

    parser = argparse.ArgumentParser(description="Parameters need to be input for testing")
    parser.add_argument('--job', default=0)
    parser.add_argument('--grid_size', default=4)
    parser.add_argument('--num_epoch', default='10000')
    args = parser.parse_args()

    # Job 0, 4x4 frozen lake environment training, correctness test, and comparison test
    if args.job == 0:
        env = Environment(grid_size=4)
        # Create three agents corresponding to three algorithms
        Monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)

        SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        label_1 = ['Monte_carlo', 'SARSA', 'Q_learning']

        Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1 = Monte_carlo.fv_mc_prediction(num_epoch=args.num_epoch)

        Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2 = SARSA.train(num_epoch=args.num_epoch)

        Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3 = Q_learning.train(num_epoch=args.num_epoch)

        steps = [steps_1, steps_2, steps_3]

        all_cost = [all_cost_1, all_cost_2, all_cost_3]

        accuracy = [accuracy_1, accuracy_2, accuracy_3]

        Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

        plot_steps(steps, label_1)

        plot_all_cost_scatter(all_cost, label_1)

        plot_accuracy(accuracy, label_1)

        plot_average_rewards(Rewards_list, label_1)

    # Job 1, 10X10 frozen lake environment training, correctness test, and comparison test
    elif args.job == 1:
        NUM_EPISODES = 100000

        GRID_SIZE = 10

        env = Environment(grid_size=10)
        # Create three agents corresponding to three algorithms
        Monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)

        SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        label_1 = ['Monte_carlo', 'SARSA', 'Q_learning']

        Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1 = Monte_carlo.fv_mc_prediction(num_epoch=args.num_epoch)

        Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2 = SARSA.train(num_epoch=args.num_epoch)

        Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3 = Q_learning.train(num_epoch=args.num_epoch)

        steps = [steps_1, steps_2, steps_3]

        all_cost = [all_cost_1, all_cost_2, all_cost_3]

        accuracy = [accuracy_1, accuracy_2, accuracy_3]

        Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

        plot_steps(steps, label_1)

        plot_all_cost_scatter(all_cost, label_1)

        plot_accuracy(accuracy, label_1)

        plot_average_rewards(Rewards_list, label_1)

    # Job 2, comparison test for different learning rate value settings
    elif args.job == 2:
        env = Environment(grid_size=args.grid_size)

        label_2 = ['lr:0.01', 'lr:0.001', 'lr:0.0001']

        Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        Q_learning.lr = 0.01

        Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1 = Q_learning.train(num_epoch=args.num_epoch)

        Q_learning.lr = 0.001

        Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2 = Q_learning.train(num_epoch=args.num_epoch)

        Q_learning.lr = 0.0001

        Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3 = Q_learning.train(num_epoch=args.num_epoch)

        steps = [steps_1, steps_2, steps_3]

        all_cost = [all_cost_1, all_cost_2, all_cost_3]

        accuracy = [accuracy_1, accuracy_2, accuracy_3]

        Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

        plot_steps(steps, label_2)

        plot_all_cost_scatter(all_cost, label_2)

        plot_accuracy(accuracy, label_2)

        plot_average_rewards(Rewards_list, label_2)

    # Job 3, comparison test for different gamma value settings
    elif args.job == 3:
        env = Environment(grid_size=args.grid_size)

        label_3 = ['gamma:0.8', 'gamma:0.9', 'gamma:0.99']

        Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        Q_learning.gamma = 0.8
        Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1 = Q_learning.train(num_epoch=args.num_epoch)

        Q_learning.gamma = 0.9
        Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2 = Q_learning.train(num_epoch=args.num_epoch)

        Q_learning.gamma = 0.99
        Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3 = Q_learning.train(num_epoch=args.num_epoch)

        steps = [steps_1, steps_2, steps_3]

        all_cost = [all_cost_1, all_cost_2, all_cost_3]

        accuracy = [accuracy_1, accuracy_2, accuracy_3]

        Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

        plot_steps(steps, label_3)

        plot_all_cost_scatter(all_cost, label_3)

        plot_accuracy(accuracy, label_3)

        plot_average_rewards(Rewards_list, label_3)

    # Job 4, comparison test for different epsilon value settings
    elif args.job == 4:
        env = Environment(grid_size=args.grid_size)

        label_4 = ['epsilon:0.7', 'epsilon:0.8', 'epsilon:0.9']

        Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

        Q_learning.epsilon = 0.7
        Q_1, steps_1, all_cost_1, accuracy_1, all_cost_bar_1, Rewards_list_1 = Q_learning.train(num_epoch=args.num_epoch)

        Q_learning.epsilon = 0.8
        Q_2, steps_2, all_cost_2, accuracy_2, all_cost_bar_2, Rewards_list_2 = Q_learning.train(num_epoch=args.num_epoch)

        Q_learning.epsilon = 0.9
        Q_3, steps_3, all_cost_3, accuracy_3, all_cost_bar_3, Rewards_list_3 = Q_learning.train(num_epoch=args.num_epoch)

        steps = [steps_1, steps_2, steps_3]

        all_cost = [all_cost_1, all_cost_2, all_cost_3]

        accuracy = [accuracy_1, accuracy_2, accuracy_3]

        Rewards_list = [Rewards_list_1, Rewards_list_2, Rewards_list_3]

        plot_steps(steps, label_4)

        plot_all_cost_scatter(all_cost, label_4)

        plot_accuracy(accuracy, label_4)

        plot_average_rewards(Rewards_list, label_4)
