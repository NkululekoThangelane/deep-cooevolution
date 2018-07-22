import torch
import torch.multiprocessing as mp
from torch.multiprocessing import SimpleQueue
import numpy as np

import pickle

import time


class Worker(mp.Process):
    def __init__(self, id, param, state_normalizer, task_q, result_q, stop, config):
        mp.Process.__init__(self)
        self.id = id
        self.task_q = task_q
        self.param = param
        self.result_q = result_q
        self.stop = stop
        self.config = config
        #self.evaluator = Evaluator(config, state_normalizer)

    def run(self):
        train_loader, test_loader = get_dataset()
        pop = Population(train_loader=train_loader)

        # pop.indv_batch()


        # TODO RUN COOEVOLTUOION


        best = pop.evolve()
        print("ThE BEST Fitness", best.get_fitness())
        train_fitness = pop.gen_fitness
        plot_pop(train_fitness)

        print("Origial genetation", best.get_generation())

        eval(best.net, test_loader)


