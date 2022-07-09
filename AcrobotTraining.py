import gym
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from os.path import exists

env = gym.make('Acrobot-v1')

#CONSTANTS:
num_generations = 40
num_children = 150
num_layers = 2
input_space = 6
neurons_per_layer = 6
output_space = 1
action_threshold = 2
weight_max_magnitude = 10
bias_max_magnitude = 10
#min_exploration = 0.2
tests_per_network = 3
max_episode_length = 500
full_randomization = False
next_full_randomization_setting = False
partial_randomization = True
next_partial_randomization_setting = True
start_evolving_gen = -1
render_demos = True
render_bad_demos = True
saving = True
final_testing_iterations = 1000
final_display_iterations = 5
success = False

#Setting up neural network:
best_NN = np.zeros((neurons_per_layer*num_layers+output_space, neurons_per_layer, 2))
#best_NN = np.zeros((neurons_per_layer, input_space))
#for l in range(num_layers-1):
#    np.append(best_NN, np.zeros((neurons_per_layer, neurons_per_layer)))
#np.append(best_NN,np.zeros(neurons_per_layer))

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

# Mutate the neural network to make a child:
def mutateNeuralNetwork(NN, randValue): #randValue is between 0 and 1
    newNN = np.zeros((neurons_per_layer*num_layers+output_space, neurons_per_layer, 2))
    for l in range(num_layers*neurons_per_layer+1):
        for i in range(neurons_per_layer):
            oldValue1 = NN[l][i][0]
            newValue1 = oldValue1 + (1-2*rand.random())*randValue*weight_max_magnitude
            if newValue1 > weight_max_magnitude:
                newValue1 = weight_max_magnitude
            if newValue1 < -1*weight_max_magnitude:
                newValue1 = -1 * weight_max_magnitude
            newNN[l][i][0] = newValue1

            oldValue2 = NN[l][i][1]
            newValue2 = oldValue2 + (1-2*rand.random())*randValue*bias_max_magnitude
            if newValue2 > bias_max_magnitude:
                newValue2 = bias_max_magnitude
            if newValue2 < -1*bias_max_magnitude:
                newValue2 = -1 * bias_max_magnitude
            newNN[l][i][1] = newValue2
    return newNN

for gen in range(num_generations):
    successes = 0
    bestReward = -10000
    bestIndex = -1
    best_NN_in_generation = np.zeros((neurons_per_layer*num_layers+output_space, neurons_per_layer, 2))
    #maxRValue =  1 - (1-min_exploration)*(gen-start_evolving_gen)/(num_generations-start_evolving_gen)
    maxRValue =  (1 - gen/num_generations)*(1-gen/num_generations) #Squared for faster decrease
    if full_randomization:
        maxRValue = 1
    elif partial_randomization:
        maxRValue = (1-gen/num_generations)
    print("max rValue =", maxRValue)
    for child in range(num_children):

        # Mutate nueral network
        #rValue = 1 - (1-min_exploration)*(gen-start_evolving_gen)/(num_generations-start_evolving_gen)
        rValue = maxRValue * child/num_children
        if full_randomization:
            rValue = 1
        newNN = mutateNeuralNetwork(best_NN, rValue)
        #print(newNN)

        #Make variables:
        total_good_rewards = 0

        # Simulate:
        for test in range(tests_per_network):
            #Make iteration of the environment
            totalReward = 0
            max_y_val = -10
            obs = env.reset()
            for step in range(max_episode_length):
                obsArray = np.array(obs)
                height = -obsArray[0] - (obsArray[0]*obsArray[2] - obsArray[1]*obsArray[3])
                if height > max_y_val:
                    max_y_val = height
                action = neuralNetworkOutput(newNN, obsArray)
                if action > action_threshold:
                    action = 1
                elif action < -1 * action_threshold:
                    action = -1
                else:
                    action = 0
                obs, reward, done, info = env.step(action)
                #env.render()
                totalReward += reward
                if done:
                    if step < max_episode_length - 1:
                        next_full_randomization_setting = False 
                        next_partial_randomization_setting = False
                        #print('success!')
                        successes += 1
                        if not success:
                            start_evolving_gen = gen
                            print('first success!')
                            success = True
                    break
            if not success:
                total_good_rewards += max_y_val - 500
            else:
                total_good_rewards += totalReward

        #Evaluate/save
        # print("Reward ", child, " = ", total_good_rewards)
        if total_good_rewards > bestReward:
            bestReward = total_good_rewards
            best_NN_in_generation = newNN
            bestIndex = child
    print("Best reward = ", (bestReward/tests_per_network))
    print("Generation", gen, "success rate =", successes, "/", (num_children*tests_per_network))
    #print("Best reward index = ", bestIndex)
    print()

    best_NN = best_NN_in_generation
    full_randomization = next_full_randomization_setting
    partial_randomization = next_partial_randomization_setting
    #print("Best NN = ", best_NN)
    #Run demo:
    if render_demos and (render_bad_demos or success):
        obs = env.reset()
        for step in range(max_episode_length):
            obsArray = np.array(obs)
            action = neuralNetworkOutput(best_NN, obsArray)
            if action > action_threshold:
                action = 1
            elif action < -1 * action_threshold:
                action = -1
            else:
                action = 0
            obs, reward, done, info = env.step(action)
            env.render()
            totalReward += reward
            if done:
                break


#Final Model Testing:
print()
print("Testing Final Model...")
print()
final_successes = 0
avg_reward = 0
for iteration in range(final_testing_iterations):
    obs = env.reset()
    total_reward = 0
    for step in range(max_episode_length):
        obsArray = np.array(obs)
        action = neuralNetworkOutput(best_NN, obsArray)
        if action > action_threshold:
            action = 1
        elif action < -1 * action_threshold:
            action = -1
        else:
            action = 0
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            if step < max_episode_length - 1:
                final_successes +=1
            break
    avg_reward += total_reward

avg_reward /= final_testing_iterations
print("Final Model Results:")
print()
print("Final Success rate =", final_successes, "/", final_testing_iterations, "=", (final_successes/final_testing_iterations*100),"%")
print("Average Reward =", avg_reward)
print()

#Displaying Final Model
print("Displaying sample runs of final model....")
for iteration in range(final_display_iterations):
    obs = env.reset()
    for step in range(max_episode_length):
        obsArray = np.array(obs)
        action = neuralNetworkOutput(best_NN, obsArray)
        if action > action_threshold:
            action = 1
        elif action < -1 * action_threshold:
            action = -1
        else:
            action = 0
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

#Saving Final Model
if saving:
    #Creating file:
    print()
    folder = "/Users/henrydemarest/Documents/Random Coding Experiments/MachineLearningExperiments/Classic Control/Acrobot/Saved Networks/"
    filename = "AB"
    filename = filename + "_gens-" + str(num_generations)
    filename = filename + "_children-"+str(num_children) 
    filename = filename + "_layers-" + str(num_layers)
    filename = filename +"_layerHeight-"+str(neurons_per_layer)
    filename = filename + "_networkTests-"+str(tests_per_network)
    filename = filename + "_wMax-"+str(weight_max_magnitude)
    filename = filename + "_bMax-"+str(bias_max_magnitude)
    filename = filename + "_aThreshold-"+str(action_threshold)
    if exists(folder + filename + ".txt"):
        value = 1
        while (exists(folder +filename + "_"+str(value))):
            value+=1
        filename = filename + "_" + str(value)
    filename = filename +".txt"
    file = open(folder+filename, "w")
    print("Creating file....")
    print("Filename =", filename)

    #Writing data:
    #First line = network constants:
    print("Saving constants...")
    firstline = []
    firstline.append(str(neurons_per_layer)+",")
    firstline.append(str(num_layers)+",")
    firstline.append(str(output_space)+",")
    firstline.append(str(action_threshold+"\n"))
    file.writelines(firstline)
    #Following lines = neural network
    print("Saving final neural network...")
    for l in range(neurons_per_layer*num_layers+output_space):
        line = []
        for i in range(neurons_per_layer):
            line.append(str(best_NN[l][i][0])+",")
            line.append(str(best_NN[l][i][1])+";")
        line.append("\n")
        file.writelines(line)
    #All following lines = training performances
    print("Saving performance statistics...")
    lastline = []
    lastline.append(str(final_successes/final_testing_iterations*100)+",")
    lastline.append(str(avg_reward)+",")
    lastline.append(str(final_testing_iterations)+"\n")
    file.writelines(lastline)

    file.close()
env.close()