import torch
import matplotlib.pyplot as plt
import argparse

from agent import Agent

parser = argparse.ArgumentParser(description='This script runs the highway agent with the Deep Q-learning model.')
parser.add_argument("--mode", default="train", help="Mode of the script to run - Either 'train' or 'test'")
parser.add_argument("--episodes", default=1000, help="Number of road runs to run - default: 1000")
parser.add_argument("--difficulty", default=2, help="Difficulty of the road (number of surrounding cars) - default: 2")
parser.add_argument("--weights", default=None, help="Path to pre-trained weights. When not specified, the model learns from scratch.")
parser.add_argument("--cpu_only", action="store_true", help="Whether to use only cpu")
parser.add_argument("--slow", action="store_true", help="Whether to slow down the road simulation")
parser.add_argument("--silent", action="store_true", help="Whether to keep the logs from the progress of agent's road run silent")
args = parser.parse_args()

if args.cpu_only or torch.cuda.is_available() == False:
    device = "cpu"
else:
    device = "cuda"

if args.silent != True:
    print(f"\n\n>>> Running on {device}, buckle up! <<<\n")

if __name__ == "__main__":
    agent = Agent(device=device)
    if args.weights: agent.q_learner.load_pretrained_w(path_to_weights=args.weights,silent=args.silent)

    to_plot = {}
    if args.mode == "train":
        if args.silent != True: print(f"\n... TRAINING TIME ...\n>>> Road difficulty: {args.difficulty} >>> Num of episodes: {args.episodes}\n\n")
        to_plot["Time until crash"],to_plot["Loss"] = agent.train(max_episodes=int(args.episodes),w_exploration=True,road_difficulty=int(args.difficulty),silent=args.silent)
    else:
        if args.silent != True: print(f"\n... TEST RUNS ...\n>>> Road difficulty: {args.difficulty} >>> Num of episodes: {args.episodes}\n\n")
        to_plot["Time until crash"],to_plot["Average speeds"] = agent.test(num_of_episodes=int(args.episodes),slow_simulation=args.slow,road_difficulty=int(args.difficulty),silent=args.silent)
    
    fig,ax = plt.subplots(nrows=2,figsize=(8,8))
    for i,history_type in enumerate(to_plot):
        ax[i].plot(range(len(to_plot[history_type])),to_plot[history_type])
        ax[i].title.set_text(history_type)
        ax[i].set_xlabel('Episode (ID of road run)')
    fig.tight_layout(pad=3.0)
    plt.show()
