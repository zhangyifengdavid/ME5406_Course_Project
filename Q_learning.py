import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from Environment import Environment
from Parameters import *

np.random.seed(1)


class Q_learning(object):
    def __init__(self, env, learning_rate, gamma, epsilon):
        # Class environment
        self.env = env
        # List of actions
        self.actions = list(range(self.env.n_actions))
        # Learning rate
        self.lr = learning_rate
        # Value of gamma
        self.gamma = gamma
        # Value of epsilon
        self.epsilon = epsilon
        # Creating full Q-table for all cells
        self.q_table = pd.DataFrame(columns=self.actions)
        # Creating Q-table for cells of the final route
        self.q_table_final = pd.DataFrame(columns=self.actions)

    # Adding to the Q-table new states
    def check_state_validation(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

        # Function for choosing the action for the agent

    # Choose valid actoins
    def epsilon_greedy_policy(self, observation):

        self.check_state_validation(observation)
        # action selection
        if np.random.uniform() > self.epsilon:
            # choose random action
            action = np.random.choice(self.actions)
        else:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    # Learning and updating the Q table using the Q learning update rules as :
    # Q(s,a) = Q(s,a) + alpha *(r + gamma * max[Q(s',a)] - Q(s,a))
    def learn(self, state, action, reward, next_state):
        # Checking if the next step exists in the Q-table
        self.check_state_validation(next_state)

        # Current state in the current position
        q_predict = self.q_table.loc[state, action]

        # Calculate the q target value according to update rules
        q_target = reward + self.gamma * self.q_table.loc[next_state, :].max()

        # Updating Q-table
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

        return self.q_table.loc[state, action]

    # Train for updating the Q table
    def train(self, num_epoch):
        # Resulted list for the plotting Episodes via Steps
        steps = []
        # Resulted list for the plotting Episodes via cost
        all_costs = []
        # Resulted list for the plotting Episodes via average accuracy
        accuracy = []
        # List for average rewards
        Reward_list = []
        # List for Q value
        Q_value = {}

        # Initialize variable
        goal_count = 0
        rewards = 0
        positive_count = 0
        negative_count = 0

        for i in range(num_epoch):
            # Initial Observation
            observation = self.env.reset()

            # Initialize step count
            step = 0

            # Initialize cost count
            cost = 0

            # Calculate the accuracy for every 50 steps
            if i != 0 and i % 50 == 0:
                goal_count = goal_count / 50
                accuracy += [goal_count]
                goal_count = 0

            # Record Q value for specific grid for checking converging
            if i != 0 and i % 1000 == 0:
                Q_value[i] = []
                for j in range(self.env.n_actions):
                    Q_value[i].append(self.q_table.loc[str(14), j])

            while True:
                # Render environment
                # self.env.render()

                # RL chooses action based on epsilon greedy policy
                action = self.epsilon_greedy_policy(str(observation))

                # Takes an action and get the next observation and reward
                observation_, reward, done, info = self.env.step(action)

                # learns from this transition and calculating the cost
                cost += self.learn(str(observation), action, reward, str(observation_))

                # Swapping the observations - current and next
                observation = observation_

                # Count the number of Steps in the current Episode
                step += 1

                # Break while loop when it is the end of current Episode
                # When agent reached the goal or obstacle
                if done:
                    # Record the positive cost and negative cost
                    if reward > 0:
                        positive_count += 1
                    else:
                        negative_count += 1

                    # Record the step
                    steps += [step]

                    # Record the cost
                    all_costs += [cost]

                    # goal count +1, if reaching the goal
                    if reward == 1:
                        goal_count += 1

                    # Record total rewards to calculate average rewards
                    rewards += reward
                    Reward_list += [rewards / (i + 1)]

                    break

            print('episode:{}'.format(i))

        # See if converge
        print("Q_value：{}".format(Q_value))

        # Record the data to the list
        all_cost_bar = [positive_count, negative_count]

        # Showing the final route
        # self.env.final()

        # # Showing the Q-table with values for each action
        self.print_q_table()

        # # Plotting the results
        self.plot_results(steps, all_costs, accuracy, all_cost_bar, Reward_list)

        return self.q_table, steps, all_costs, accuracy, all_cost_bar, Reward_list

    # Printing the Q-table with states
    def print_q_table(self):
        # Getting the coordinates of final route from env.py
        e = self.env.final_states()

        # Comparing the indexes with coordinates and writing in the new Q-table values
        for i in range(len(e)):
            state = str(e[i])
            # Going through all indexes and checking
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.q_table_final.loc[state, :] = self.q_table.loc[state, :]

        print()
        print('Length of full Q-table =', len(self.q_table.index))
        print('Full Q-table:')
        print(self.q_table)

    # plot training results
    def plot_results(self, steps, cost, accuracy, all_cost_bar, Reward_list):

        #
        f, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
        #
        ax1.plot(np.arange(len(steps)), steps, 'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Episode via steps')

        #
        ax2.plot(np.arange(len(cost)), cost, 'b')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Episode via cost')

        #
        ax3.plot(np.arange(len(accuracy)), accuracy, 'b')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Episode via Accuracy')

        plt.tight_layout()  # Function to make distance between figures

        #
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Episode via steps')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        #
        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'r')
        plt.title('Episode via cost')
        plt.xlabel('Episode')
        plt.ylabel('Cost')

        #
        plt.figure()
        plt.plot(np.arange(len(accuracy)), accuracy, 'r')
        plt.title('Episode via Accuracy')
        plt.xlabel('Episode')
        plt.ylabel('Accuracy')

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
                # render the environment
                # self.env.render()

                # Choose the best action based on the optimal_policy
                state_action = self.q_table.loc[str(observation), :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)

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

        # Plot results
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


# Commands to be implemented after running this file
if __name__ == "__main__":
    # Create an environment
    env = Environment(grid_size=GRID_SIZE)

    # Create a q learning agent
    Q_learning = Q_learning(env, learning_rate=LEARNING_RATE, gamma=GAMMA, epsilon=EPSILON)

    # Learning and updating
    Q_table = Q_learning.train(num_epoch=NUM_EPISODES)

    # Test after training
    Q_learning.test()

    # Remain visualization
    env.mainloop()
