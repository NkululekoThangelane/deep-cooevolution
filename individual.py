import torch
from random import random
import numpy as np
from Net import Net
from operator import attrgetter
from itertools import repeat
from collections import Sequence
import math


class Individual(object):
    def __init__(self,generation,genome=None,net=Net()):
        self.net = net
        self.generation = generation
        self.fitness = 0
        self.relative_fitness=0
        self.accuracy = 0
        self.mut_rate = 0.3
        self.data=None
        self.target=None
        self.genome=genome
        if self.genome is None:
            self.genome = net.extract_parameters()


        # self.mutations =["mutGaussian","mutShuffleIndexes","mutRandom","mutZero","mutStd","mutOne","mutMultiplyMean"]
        self.mutations = ["mutGaussian"]



    def get_fitness(self):
        return self.fitness

    def get_net(self):
        return self.net

    def get_generation(self):
        return self.generation
