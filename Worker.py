import math
import ray
from Net import Net
import torch
import torch.nn.functional as F
import random
@ray.remote
class Worker(object):

    def __init__(self, data, target):
        self.data =data
        self.target = target
        self.count =0

    def eval(self,invw):
        """
        Function to evalute neural network performance
        :param invw:
        :return:
        """
        inv = Net()
        inv.inject_parameters(invw)


        #data = torch.from_numpy(self.data)
        #target = torch.from_numpy(self.target)
        with torch.no_grad():
            output = inv(self.data)
            loss = F.nll_loss(output, self.target)
            running_loss = loss.item()
            if math.isnan(running_loss):
                running_loss = random.uniform(100000, 2000000)
                # house keeping
        self.count += 1

        return running_loss

    def mutate(self, indv):
        """
        Remote Mutate
        :param indv:
        :return:
        """
        pass










