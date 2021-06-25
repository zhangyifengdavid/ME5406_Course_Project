import random
import numpy as np
import math
import matplotlib.pyplot as plt

from Environment import Environment
from collections import defaultdict
from Parameters import *

np.random.seed(10)


class Monte_carlo(object):
    def __init__(self, env, epsilon, gamma):
        # Variable initialization
        self.env = env
        self.n_obs = self.env.n_states
        self.n_a = self.env.n_actions

        self.epsilon = epsilon
        self.gamma = gamma
        # Variables for metrics
        self.steps = []
        self.all_cost = []
        self.accuracy = []
        self.Rewards_list = []
        self.rewards = 0
        self.positive_count = 0
        self.negative_count = 0
        self.goal_count = 0

        self.Q, self.Total_return, self.N = self.create_q_table()

    # Create a Q table
    def create_q_table(self):
        """
        Initialization
        :return
        1. Q[(s,a)]:initialize the dictionary for storing the Q values;
        2. Total_return[(s,a)]: initialize the dictionary for storing the total return of the state-action pair;
        3. N[(s,a)]: initialize the dictionary for storing the count of the number of times a state-action pair is visited
        """
        Q = defaultdict(float)
        N = defaultdict(int)
        Total_return = defaultdict(float)

        for s in range(self.n_obs):
            for a in range(self.n_a):
                Q[(s, a)] = 0.0
                Total_return[(s, a)] = 0
                N[(s, a)] = 0
        return Q, Total_return, N

    # Choose actions based on epsilon greedy policy
    def epsilon_greedy_policy(self, observation):
        """
        Define a epsilon greedy policy based on epsilon soft policy

        If a random number larger than epsilon, choose a random action.
        Otherwise, choose an action with maximum Q value.

        Argument: input observation, Q table
        Return: Action
        """
        # Sample a random number from a uniform distribution, if the number is less than
        # the value of epsilon then choose a random action, else choose the best action
        # which has the maximum Q value based on the current state
        if random.uniform(0, 1) > self.epsilon:
            return np.random.randint(0, 3)
        else:
            return max(list(range(self.n_a)), key=lambda x: self.Q[(observation, x)])

    # Choose actions based on optimal policy
    def optimal_policy(self, observation):
        """
        Define the optimal policy, choosing the best action which
        has the maximum Q value with the input observation

        Argument: input observation, Q table

        Return: Action
        """
        return max(list(range(self.n_a)), key=lambda x: self.Q[(observation, x)])

    # Generate a list of data for a episode
    def generate_episode(self):
        """
        To generate episodes/experience and store for updating Q table
        """

        # initialize a list for storing the episode
        episode = []

        # Reset the environment and get the initial observation
        observation = self.env.reset()

        steps = 0

        # Loop for each time step
        for t in range(NUM_STEPS):
            # Choose the action according to the epsilon greedy policy
            action = self.epsilon_greedy_policy(observation)

            # Perform the action the get the [next_obs, reward, info] tuple
            next_observation, reward, done, info = self.env.step(action)

            # Store the transition tuple
            episode.append((observation, action, reward))

            steps += 1

            # if the next state is the terminal, then break the loop
            if done:
                # Record the positive cost and negative cost
                if reward > 0:
                    self.positive_count += 1

                    self.goal_count += 1
                else:
                    self.negative_count += 1

                # Record the step
                self.steps += [steps]

                self.rewards += reward

                print("Episode finished after {} time steps".format(t + 1))
                break

            # update time
            # else update the next observation to the current state
            observation = next_observation

        return episode

    # Learning and updating Q table based on the First visit Monte Carlo method
    def fv_mc_prediction(self, num_epoch):
        """
        Updating process for first visit MC method

        Step 1: Calculate the total return value for each step/(s,a) pair of each generated episode
                G <- gamma* G + R(t+1)

        Step 2: If (s,a) pair is first visited(not occur before in this episode), then add G to the Returns(s,a)

        Step 3: Calculate the average Returns(s,a) as the Q(s,a)

        """

        # For each iteration
        for i in range(num_epoch):
            cost = 0

            # Using the initialized Q table to generate an episode
            episode = self.generate_episode()

            # Get all state-action pairs in the episodes
            state_action_pairs = [(observation, action) for (observation, action, reward) in episode]

            # Initialize the G value
            G = 0

            # Calculate the accuracy rate for every 50 steps
            if i != 0 and i % 50 == 0:
                self.goal_count = self.goal_count / 50
                self.accuracy += [self.goal_count]
                self.goal_count = 0

            # Record average rewards
            self.Rewards_list += [self.rewards / (i + 1)]

            # for each state-action pairs
            for i in range(len(episode)):
                # Calculate the return G from the end, T-1, T-2...... by G = gamma* G + R(t+1)
                observation, action, reward = episode[len(episode) - (i + 1)]

                G = reward + self.gamma * G

                # Check if the state-action pair is occurring for the first time in the episode
                # #(limited by first visit MC method)
                if not (observation, reward) in state_action_pairs[:i]:
                    # update the total return of the state-action pair
                    self.Total_return[(observation, action)] += G

                    # update the number of times the state-action pair is visited
                    self.N[(observation, action)] += 1

                    # calculate the Q value for each state-action pair by taking the average
                    self.Q[(observation, action)] = self.Total_return[(observation, action)] / self.N[
                        (observation, action)]

                # Record total cost
                cost += self.Q[(observation, action)]

            self.all_cost += [cost]

        all_cost_bar = [self.positive_count, self.negative_count]
        # Print final route
        # env.final()
        # Plot training results
        self.plot_results(self.steps, self.all_cost, self.accuracy, all_cost_bar, self.Rewards_list)

        return self.Q, self.steps, self.all_cost, self.accuracy, all_cost_bar, self.Rewards_list

    # Plot training results
    @ staticmethod
    def plot_results(steps, all_cost, accuracy, all_cost_bar, Reward_list):
        # Plot Episodes vis steps
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        # Plot Episodes via Cost
        plt.figure()
        plt.plot(np.arange(len(all_cost)), all_cost)
        plt.title('Episode via Cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        # Plot Episodes via Accuracy
        plt.figure()
        plt.plot(np.arange(len(accuracy)), accuracy, 'b')
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')

        # Plot Bar of Success and failure rate
        plt.figure()
        list = ['Success', 'Fail']
        color_list = ['blue', 'red']
        plt.bar(np.arange(len(all_cost_bar)), all_cost_bar, tick_label=list, color=color_list)
        plt.title('Bar/Success and Fail')
        plt.ylabel('Number')

        # Plot Episode via Average rewards
        plt.figure()
        plt.plot(np.arange(len(Reward_list)), Reward_list, 'b')
        plt.title('Episode via Average rewards')
        plt.xlabel('Episode')
        plt.ylabel('Average rewards')

        # Showing the plots
        plt.show()

    # Test after training
    def test(self):
        # run a set of episode to test the correctness of the method
        num_test = 100

        # Print route
        f = {}

        # Initialize count, and data store lists
        num_reach_goal = 0
        reward_list = []
        steps_list = []

        for i in range(num_test):
            # reset the environment
            observation = self.env.reset()

            # render the environment
            # env.render()

            for j in range(NUM_STEPS):
                # # render the environment
                # self.env.render()

                # Choose the best action based on the optimal_policy
                action = self.optimal_policy(observation)

                # perform action and get a tuple
                next_observation, reward, done, info = self.env.step(action)

                # Coordinate transformation
                y = int(math.floor(next_observation / GRID_SIZE)) * PIXELS
                x = int(next_observation % GRID_SIZE) * PIXELS
                f[j] = [x, y]

                if done:
                    if reward == 1:
                        num_reach_goal += 1
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

        print("correctness:{}".format(num_reach_goal / num_test))

        # Plot the test results
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


if __name__ == "__main__":
    # create a FrozenLake environment
    env = Environment(grid_size=GRID_SIZE)

    # Create a monte carlo agent
    monte_carlo = Monte_carlo(env, epsilon=EPSILON, gamma=GAMMA)

    # Learning and updating Q table
    Q = monte_carlo.fv_mc_prediction(num_epoch=NUM_EPISODES)

    # write_Q_table(file_name='./Q_table/monte_carlo', Q = Q)

    # Test after training
    monte_carlo.test()

    # Remain visualization
    env.mainloop()
