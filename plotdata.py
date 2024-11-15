from matplotlib import pyplot as plt
import numpy as np
import json
import os
from spicy import interpolate


def load_data(folder_name):
    data = []
    for file_name in os.listdir(folder_name):
        with open(folder_name+"/"+file_name) as f:
            d = json.load(f)
            d = np.array(d)
            data.append(d)
    data = np.array(data)
    return data

def process_data(data):
    steps = data[:,:,1]
    rewards = data[:,:,2]

    min_length = min(len(reward) for reward in rewards)

    interpolated_rewards = []
    for i, reward in enumerate(rewards):
        # Create an interpolation function for rewards[i]
        interp_func = interpolate.interp1d(steps[i], reward, kind='linear', fill_value="extrapolate")
        
        # Generate the new x values based on the smallest step count
        new_steps = np.linspace(steps[i][0], steps[i][-1], min_length)
        
        # Apply the interpolation function
        new_rewards = interp_func(new_steps)
        interpolated_rewards.append(new_rewards)

    interpolated_rewards = np.array(interpolated_rewards)

    # smooth the data
    smoothed_rewards = []
    pad = 25
    for reward in interpolated_rewards:
        smoothed_reward = np.convolve(reward, np.ones(pad)/pad, mode='same')
        smoothed_rewards.append(smoothed_reward)

    smoothed_rewards = np.array(smoothed_rewards)

    rewardmean = np.mean(smoothed_rewards, axis=0)[:-pad]
    rewardstd = np.std(smoothed_rewards, axis=0)[:-pad]
    return new_steps[:-pad], rewardmean, rewardstd


def plot_data(list_to_plot, labels, title):
    for i, (steps, rewardmean, rewardstd) in enumerate(list_to_plot):
        plt.plot(steps, rewardmean, label=labels[i])
        plt.fill_between(steps, rewardmean - rewardstd, rewardmean + rewardstd, alpha=0.3)

    plt.xlabel('Steps')
    plt.ylabel('Reward')

    plt.title(title)
    plt.legend()
    plt.show()




folderAllArchers = "DataPlot/AllArchers"
folderGlobalReward = "DataPlot/GlobalReward"
folderArrowPenalty = "DataPlot/ArrowPenalty"

dataAllArchers = load_data(folderAllArchers)
dataGlobalReward = load_data(folderGlobalReward)
dataArrowPenalty = load_data(folderArrowPenalty)

processedDataAllArchers = process_data(dataAllArchers)
processedDataGlobal = process_data(dataGlobalReward)
processedDataArrowPenalty = process_data(dataArrowPenalty)

plot_data([processedDataAllArchers,processedDataGlobal, processedDataArrowPenalty], ["AllArchers", "Reward Global", "Arrow Penalty"], "All Archers vs Reward Global vs Arrow Penalty")