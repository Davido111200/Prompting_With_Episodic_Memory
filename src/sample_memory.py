import numpy as np


from src.arguments import get_training_args
import random
import torch
from src.utils import convert_permutations
from sklearn.neighbors import NearestNeighbors
from itertools import permutations
from math import comb
import math
from collections import deque

def normalize_list(data):
    data = np.array(data)
    mean = data.mean()
    std = data.std()
    normalized_data = (data - mean) / std
    return normalized_data.tolist()


class Memory:
    def __init__(self, args, n_states, n_actions):
        self.args = args
        self.n_states = n_states
        self.n_actions = n_actions

        self.states = {i: [] for i in range(self.n_states)}
        self.values = {i: np.zeros(self.n_actions) for i in range(self.n_states)}
        self.occurences = {i: np.zeros(self.n_actions) for i in range(self.n_states)}


    def update(self, state_idx, state, action, value):
        if not self.states[state_idx]:
            self.states[state_idx].append(state.tolist())
            self.values[state_idx][action] = value
            self.occurences[state_idx][action] += 1
        else:
            if self.args.write_type == 'max':
                self.values[state_idx][action] = max(self.values[state_idx][action], value)
            elif self.args.write_type == 'mean':
                self.values[state_idx][action] = (self.values[state_idx][action] * self.occurences[state_idx][action] + value) / (self.occurences[state_idx][action] + 1)
                self.occurences[state_idx][action] += 1



    def _get(self, cur_embd, top_k, weighted=True):
        """
        weighted: if True, return the weighted average of the top k closest states
        """
        neigh = NearestNeighbors(n_neighbors=top_k, metric='cosine')

        if len(list(self.states.values())[0]) == 1:
            new_states = [state[0] for state in list(self.states.values())]
        else:
            new_states = [state for state in list(self.states.values())]

        # print(new_states)

        neigh.fit(new_states)

        distances, indicies = neigh.kneighbors([cur_embd])


        # Access the values of top k indicies
        top_k_values = [self.values[i] for i in indicies[0]]   

        if not weighted:
            # just take the sum of the values for corresponding top k indices
            result = sum(top_k_values)
        else:
            # result = sum(np.array(distances[0]) * np.array(top_k_values))
            weighted_result = []
            
            for dis, top in zip(distances[0], top_k_values):
                weighted_result.append(top*dis)
            result = sum(weighted_result)

        return result
