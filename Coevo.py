import torch
import numpy as np
import random
from Net import Net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from individual import Individual
from operator import attrgetter
from Population import Population

class Coevo(object):

    def __init__(self,generation, train, test,num_cogen ,num_pop=2):
        self.train = train
        self.test = test
        self.num_pop = num_pop
        self.num_cogen= num_cogen
        self.populations = [Population(train_loader=self.train) for i in range(self.num_pop)]

        self.champions = []



    def run(self):

        #TODO START COOEVLOUTOION

        for i in range(self.num_cogen):

            for pop in self.populations:
                best=pop.evolve()
                self.champions.append(best)









        return True




