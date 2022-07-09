import gym
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from os.path import exists, expanduser
from pynput import keyboard
from threading import Thread

env = gym.make('CartPole-v1')
filepath = expanduser("~/Documents/Random Coding Projects/MachineLearningExperiments/OpenAI-Gym-Cartpole/Saved Networks/CP_gens-10_children-10_layers-1_layerHeight-4_networkTests-10_wMax-5_bMax-0.txt")

#open file:
file = open(filepath, "r")
lines = file.readlines()
file.close()

#get constants:
constants = lines[0].split(",")
neurons_per_layer = int(constants[0])
num_layers = int(constants[1])
output_space = int(constants[2])
NN = np.zeros((neurons_per_layer*num_layers+output_space, neurons_per_layer, 2))

#get Neural network:
for l in range(neurons_per_layer*num_layers+output_space):
    sets = lines[l+1].split(";")
    for i in range(neurons_per_layer):
        nums = sets[i].split(",")
        NN[l][i][0] = float(nums[0])
        NN[l][i][1] = float(nums[1])

#get old performances
if neurons_per_layer*num_layers+output_space + 1 < len(lines):
    print("\nPrevious performances:")
    num = 1
    for p in range(neurons_per_layer*num_layers+output_space+1, len(lines)):
        performance = lines[p].split(",")
        old_success_rate = float(performance[0])
        old_avg_reward = float(performance[1])
        old_tests_run = int(performance[2])
        print("- Saved Performance #", num, "- total tests =", old_tests_run, "success rate =", old_success_rate, ", average reward =", old_avg_reward)
        num+=1
    print()

#environment constants:
max_episode_length = 500
continuing = True

# DEFINING FUNCTIONS:
#Get output of neuronal network:
def neuralNetworkOutput(NN, input):
    last_layer = input
    for l in range(num_layers):
        next_layer = np.zeros(neurons_per_layer)
        for i in range(neurons_per_layer):
            value = 0
            for j in range(len(last_layer)):
                value += NN[l*neurons_per_layer + i][j][0] * last_layer[j] +  NN[l*neurons_per_layer + i][j][1]
            next_layer[i] = value
        last_layer = next_layer
    output = 0
    for i in range(len(last_layer)):
        output += last_layer[i] * NN[num_layers*neurons_per_layer][i][0] + NN[num_layers*neurons_per_layer][i][1]
    return output

def displayModel():
    obs = env.reset()
    for step in range(max_episode_length):
        obsArray = np.array(obs)
        action = 0
        if neuralNetworkOutput(NN, obsArray) > 0:
            action = 1
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

def key_capture():
    global continuing
    input()
    print("Terminating Display...")
    continuing = False

#Displaying Final Model
print("Press the enter key in the terminal to end display")
print("Displaying Sample Runs...")

Thread(target=key_capture, args=(), name='key_capture', daemon=True).start()

while continuing:
    displayModel()

print("Display Terminated")
env.close()