import random
import numpy as np
import math
import matplotlib.pyplot as plt

from Environment import Environment
from Parameters import *

# set the constant random seed
np.random.seed(1)


class SARSA(object):
    def __init__(self, env, learning_rate, gamma, epsilon):
        self.env = env
        self.n_obs = self.env.n_states
        self.n_a = self.env.n_actions

        # Hyper parameters
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = self.create_Q_table()

    # Create a Q table
    def create_Q_table(self):
        self.Q = {}
        for s in range(self.n_obs):
            for a in range(self.n_a):
                self.Q[(s, a)] = 0.0

        return self.Q

    # Choose actions based on epsilon greedy policy
    def epsilon_greedy(self, state):
        """
        Define a epsilon greedy policy based on epsilon soft policy:
        If a random number larger than epsilon, choose a random action.
        Otherwise, choose an action with maximum Q value.

        Argument: input observation, Q table
        Return: Action
        """
        # Set the epsilon value(hyper-parameter)
        if random.uniform(0, 1) > self.epsilon:
            return random.randint(0, 3)
        else:
            return max(list(range(self.n_a)), key=lambda x: self.Q[(state, x)])

    # Choose actions based on optimal greedy policy
    def optimal_policy(self, observation):
        """
        Define the optimal policy, choosing the best action which
        has the maximum Q value with the input observation

        Argument: input observation, Q table

        Return: Action
        """

        return max(list(range(self.n_a)), key=lambda x: self.Q[(observation, x)])

    # Learning and updating the Q table using the SARSA update rules as :
    # Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))
    def train(self, num_epoch):
        # Resulted list for the plotting Episodes via Steps
        steps = []
        all_costs = []
        accuracy = []
        Reward_list = []
        Q_value = {}

        # Initialize the counts
        goal_count = 0
        rewards = 0
        positive_count = 0
        negative_count = 0

        # for each episode
        for i in range(num_epoch):
            # reset the environment and get initial observation
            observation = self.env.reset()

            # select a action according to the epsilon greedy policy
            action = self.epsilon_greedy(observation)

            step = 0
            cost = 0

            # Calculate the accuracy rate for every 50 steps
            if i != 0 and i % 50 == 0:
                goal_count = goal_count / 50
                accuracy += [goal_count]
                goal_count = 0

            # Record Q value for specific grid for checking converging
            if i != 0 and i % 1000 == 0:
                Q_value[i] = []
                for j in range(self.env.n_actions):
                    Q_value[i].append(self.Q[((self.env.n_states - 2), j)])

            while True:
                # perform the selected action and get the tuple
                next_observation, reward, done, info = self.env.step(action)

                # choose the next step action using the epsilon greedy policy
                next_action = self.epsilon_greedy(next_observation)

                # Calculate the Q value of the state-action pair
                # SARSA specifies unique next action based on epsilon greedy policy(different from q-learning)
                self.Q[(observation, action)] += self.lr * (
                            reward + self.gamma * self.Q[(next_observation, next_action)] - self.Q[
                        (observation, action)])

                # calculating the cost
                cost += self.Q[(observation, action)]

                step += 1

                if done:
                    if reward > 0:
                        positive_count += 1
                    else:
                        negative_count += 1

                    steps += [step]
                    all_costs += [cost]
                    if reward == 1:
                        goal_count += 1

                    # Record average rewards
                    rewards += reward
                    Reward_list += [rewards / (i + 1)]

                    break

                # Update observation and action
                observation = next_observation
                action = next_action

            print("episodes:{}".format(i))

        # See if converge
        print("Q_value：{}".format(Q_value))

        # Record the data to the list
        all_cost_bar = [positive_count, negative_count]

        # Print final route
        # env.final()

        # Print results
        self.plot_results(steps, all_costs, accuracy, all_cost_bar, Reward_list)

        return self.Q, steps, all_costs, accuracy, all_cost_bar, Reward_list

    # Plotting the training results
    def plot_results(self, steps, cost, accuracy, all_cost_bar, Reward_list):
        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost)
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        #
        plt.figure()
        plt.plot(np.arange(len(accuracy)), accuracy, 'b')
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')

        #
        plt.figure()
        list = ['Success', 'Fail']
        color_list = ['blue', 'red']
        plt.bar(np.arange(len(all_cost_bar)), all_cost_bar, tick_label=list, color=color_list)
        plt.title('Bar/Success and Fail')
        plt.ylabel('Number')

        plt.figure()
        plt.plot(np.arange(len(Reward_list)), Reward_list, 'b')
        plt.title('Episode via Average rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average rewards')

        # Showing the plots
        plt.show()

    # Test after training
    def test(self):
        # Test for 100 episodes
        num_test = 100

        # Print route
        f = {}

        # Initialize count, and data store lists
        num_find_goal = 0
        reward_list = []
        steps_list = []

        # run 100 episode to test the correctness of the method
        for i in range(num_test):
            # resert the environment
            observation = self.env.reset()

            for j in range(NUM_STEPS):
                # # render the environment
                # env.render()

                # Choose the best action based on the optimal_policy
                action = self.optimal_policy(observation)

                # perform action and get a tuple
                next_observation, reward, done, info = self.env.step(action)

                # Coordinate transformation
                y = int(math.floor(next_observation / GRID_SIZE)) * PIXELS
                x = int(next_observation % GRID_SIZE) * PIXELS
                f[j] = [x, y]

                if done:
                    # Record the number of goal reaching
                    if reward == 1:
                        num_find_goal += 1
                    # While a episode terminates, record the total reward, step
                    # Then add to the list
                    r = reward
                    step = j + 1
                    reward_list += [r]
                    steps_list += [step]

                    break

                observation = next_observation

        # Print final route
        self.env.f = f
        self.env.final()

        print("correctness:{}".format(num_find_goal / num_test))

        #
        plt.figure()
        plt.plot(np.arange(len(steps_list)), steps_list, 'r')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(reward_list)), reward_list, 'r')
        plt.title('Episode via Success Rate')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')

        # Showing the plots
        plt.show()


# Store the final Q table values
def write_Q_table(file_name, Q):
    # open data file
    filename = open(file_name, 'w')
    # write data
    for k, v in Q.items():
        filename.write(str(k) + ':' + str(v))
        filename.write('\n')
    # close file
    filename.close()


if __name__ == '__main__':
    # create a FrozenLake environment
    env = Environment(grid_size=GRID_SIZE)

    # Create a SARSA agent
    SARSA = SARSA(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # write_Q_table(file_name='./Q_table/SARSA', Q = Q)

    # Learning and updating
    SARSA.train(num_epoch=NUM_EPISODES)

    # Test after training
    SARSA.test()

    # Remain visualization
    env.mainloop()
