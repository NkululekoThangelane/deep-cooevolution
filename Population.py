import gc
import torch
import numpy as np
import random
from random import randint
# from multiprocess import Pool
from Net import Net
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from individual import Individual
from operator import attrgetter
from itertools import repeat
from collections import Sequence
from tqdm import tqdm
import time
import pandas as pd
from collections import Counter
from Worker import Worker
import torch.multiprocessing as tmp

import matplotlib.pyplot as plt
import math

import ray


@ray.remote
def fitness_eval_dist(invw, data, target):

    inv =Net()
    inv.inject_parameters(invw)



    data=torch.from_numpy(data)
    target=torch.from_numpy(target)
    output = inv(data)
    loss = F.nll_loss(output, target)
    running_loss = loss.item()

    return running_loss


class Population(object):
    def __init__(self, train_loader,workers_num=8, popSize=25, tournament=5, mutation=0.15, cross_over_prob=0.5, generations=10000):

        self.popSize = popSize
        self.train_loader = train_loader
        self.tournament_size = tournament
        self.mutation_rate = mutation
        self.cross_over_prob = cross_over_prob
        self.generations = generations
        self.best_fitness = 0
        self.worst_fitness = 0
        self.selection_rounds = 40
        self.curr_gen_muts = []
        self.start = True
        self.generation_now = 0
        use_cuda = False
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.curPop = self.initil_pop(self.popSize)
        self.prevPop = []
        # if self.mutation_rate + self.cross_over_prob > 1:
        #    raise ValueError(
        #        'The sum of the crossover and mutation probabilities must be <= 1.0.'
        #    )
        self.mutations =["mutGaussian","mutShuffleIndexes","mutRandom","mutZero","mutStd","mutOne","mutMultiplyMean"]
        #self.mutations = ["mutRandom"]

        self.list_data, self.list_target = self.train_batch_list(self.train_loader)
        self.num_batches = len(self.list_data)
        self.workers_num = workers_num

        self.workers =[]

        for d,t in zip(self.list_data,self.list_target):
            self.workers.append(Worker.remote(data=d,target=t))

        self.gen_fitness = []

        self.chek_interval = 50

        self.champ = None

    def initil_pop(self, popSize):

        print("Creating Population  ", popSize)

        return [Individual(generation=self.generation_now, net=Net(init=True).to(self.device)) for i in range(popSize)]

    def train_batch_list(self, train_loader):
        list_data = []
        list_target = []
        for data, target in train_loader:
            list_data.append(data.to(self.device))
            list_target.append(target.to(self.device))

        return list_data, list_target

    def mutRun(self, individual):
        """

        :param individual:
        :return:
        """
        mutation = random.choice(self.mutations)
        self.curr_gen_muts.append(mutation)

        if mutation == "mutGaussian":

            return self.mutGaussian(individual=individual, indpb=self.mutation_rate)
        elif mutation == "mutShuffleIndexes":
            return self.mutShuffleIndexes(individual=individual, indpb=self.mutation_rate)
        elif mutation == "mutZero":
            return self.mutZero(individual=individual)
        elif mutation == "mutOne":
            return self.mutOne(individual=individual)
        elif mutation == "mutStd":
            return self.mutStd(individual=individual)
        elif mutation == "mutMultiplyMean":
            return self.mutMultiplyMean(individual=individual)
        else:
            return self.mutRandom(individual=individual)

    def mutStd(self, individual):

        for pt in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                n = random.randint(1, 100)
                # print("Mutatinggg point", pt, "from ", g[pt])
                stdv = 1. / math.sqrt(n)
                individual[pt] = random.uniform(-stdv, stdv)
                # print("Mutatinggg point",pt,"to ",g[pt])
                # print(individual[pt])
        return individual

    def mutMultiplyMean(self, individual):

        for pt in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                mu = individual.mean()
                # print("Mutatinggg point", pt, "from ", g[pt])
                individual[pt] *= random.uniform(-mu, mu)


                # print("Mutatinggg point",pt,"to ",g[pt])
                # print(individual[pt])
        return individual

    def mutGaussian(self, individual, indpb):
        """This function applies a gaussian mutation of mean *mu* and standard
        deviation *sigma* on the input individual. This mutation expects a
        :term:`sequence` individual composed of real valued attributes.
        The *indpb* argument is the probability of each attribute to be mutated.

        :param individual: Individual to be mutated.
        :param mu: Mean or :term:`python:sequence` of means for the
                   gaussian addition mutation.
        :param sigma: Standard deviation or :term:`python:sequence` of
                      standard deviations for the gaussian addition mutation.
        :param indpb: Independent probability for each attribute to be mutated.
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` and :func:`~random.gauss`
        functions from the python base :mod:`random` module.
        """
        # print("Mutsation Gaussian")
        mu = individual.mean()
        sigma = individual.std()
        size = len(individual)
        if not isinstance(mu, Sequence):
            mu = repeat(mu, size)
        elif len(mu) < size:
            raise IndexError("mu must be at least the size of individual: %d < %d" % (len(mu), size))
        if not isinstance(sigma, Sequence):
            sigma = repeat(sigma, size)
        elif len(sigma) < size:
            raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))

        for i, m, s in zip(range(size), mu, sigma):
            if random.random() < indpb:
                individual[i] += random.gauss(m, s)

        return individual

    def mutZero(self, individual):
        """

        :param individual:
        :return:
        """
        # print("Mutation Zero")

        for pt in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                # print("Mutatinggg point", pt, "from ", g[pt])
                individual[pt] = 0
                # print("Mutatinggg point",pt,"to ",g[pt])
                # print(individual[pt])
        return individual

    def mutOne(self, individual):
        """

        :param individual:
        :return:
        """
        # print("Mutation Zero")

        for pt in range(len(individual)):
            if np.random.random() < self.mutation_rate:
                # print("Mutatinggg point", pt, "from ", g[pt])
                individual[pt] = 1
                # print("Mutatinggg point",pt,"to ",g[pt])
                # print(individual[pt])
        return individual

    def mutRandom(self, individual):
        """

        :param individual:
        :return:
        """

        # print("Mutation Random")

        for pt in range(len(individual)):

            if np.random.random() < self.mutation_rate:
                individual[pt] = random.uniform(-2, 2)

        return individual

    def mutShuffleIndexes(self, individual, indpb):
        """Shuffle the attributes of the input individual and return the mutant.
        The *individual* is expected to be a :term:`sequence`. The *indpb* argument is the
        probability of each attribute to be moved. Usually this mutation is applied on
        vector of indices.

        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be exchanged to
                      another position.
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` and :func:`~random.randint`
        functions from the python base :mod:`random` module.
        """
        # print("index Shuffle")
        size = len(individual)
        for i in range(size):
            if random.random() < indpb:
                swap_indx = random.randint(0, size - 2)
                if swap_indx >= i:
                    swap_indx += 1
                individual[i], individual[swap_indx] = \
                    individual[swap_indx], individual[i]

        return individual

    def mutate_indv(self, indv):
        # print("Mutate")


        mut_genome = self.mutRun(individual=indv.genome)
        new_net = Net().to(self.device)

        new_net.inject_parameters(mut_genome)

        return Individual(generation=self.generation_now, genome=mut_genome, net=new_net)

    def selRandom(self, individuals, k):
        """Select *k* individuals at random from the input *individuals* with
        replacement. The list returned contains references to the input
        *individuals*.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :returns: A list of selected individuals.

        This function uses the :func:`~random.choice` function from the
        python base :mod:`random` module.
        """
        return random.sample(individuals, k)

    def selTournament(self, individuals, k, tournsize, fit_attr="fitness"):
        """Select the best individual among *tournsize* randomly chosen
        individuals, *k* times. The list returned contains
        references to the input *individuals*.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param tournsize: The number of individuals participating in each tournament.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.

        This function uses the :func:`~random.choice` function from the python base
        :mod:`random` module.
        """
        chosen = []
        individuals = [x for x in individuals if ~pd.isnull(x)]
        for i in range(k):
            aspirants = self.selRandom(individuals, tournsize)
            chosen.append(min(aspirants, key=attrgetter(fit_attr)))
        return chosen

    def selRoulette(self, individuals, k, fit_attr="fitness"):
        """Select *k* individuals from the input *individuals* using *k*
        spins of a roulette. The selection is made by looking only at the first
        objective of each individual. The list returned contains references to
        the input *individuals*.

        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list of selected individuals.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.

        .. warning::
           The roulette selection by definition cannot be used for minimization
           or when the fitness can be smaller or equal to 0.
        """

        s_inds = sorted(individuals, key=attrgetter(fit_attr), reverse=False)
        sum_fits = sum(getattr(ind, fit_attr) for ind in individuals)
        chosen = []
        for i in range(k):
            u = random.random() * sum_fits
            sum_ = 0
            for ind in s_inds:
                sum_ += getattr(ind, fit_attr)
                if sum_ < u:
                    chosen.append(ind)
                    break

        return chosen

    def indv_batch(self):

        data_l = []
        target_l = []

        for batch_idx, (data, target) in enumerate(self.train_loader):
            # data,target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target, volatile=True)
            data_l.append(data)
            target_l.append(target)

        for data, target, indv in zip(data_l, target_l, self.curPop):
            indv.data = data
            indv.target = target

        print("done batching")

    def selBest(self, individuals, k, fit_attr="fitness"):
        """Select the *k* best individuals among the input *individuals*. The
        list returned contains references to the input *individuals*.
        :param individuals: A list of individuals to select from.
        :param k: The number of individuals to select.
        :param fit_attr: The attribute of individuals to use as selection criterion
        :returns: A list containing the k best individuals.
        """
        return sorted(individuals, key=attrgetter(fit_attr), reverse=False)[:k]

    def get_fitness(self, individual):

        indx_batch = random.randint(0, self.num_batches - 1)

        data = self.list_data[indx_batch]
        target = self.list_target[indx_batch]
        data, target = Variable(data, volatile=True, requires_grad=False), Variable(target, volatile=True,
                                                                                    requires_grad=False)
        outputs = individual.net(data)

        loss = F.nll_loss(outputs, target, size_average=False).item()
        running_loss = loss.data[0]
        # print("losss")
        individual.fitness = running_loss

    def get_correct_fitness(self, individual):

        indx_batch = random.randint(0, self.num_batches - 1)

        data = self.list_data[indx_batch]
        target = self.list_target[indx_batch]
        data, target = Variable(data, volatile=True), Variable(target, volatile=True)
        output = individual.net(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.data.view_as(pred)).cpu().sum()

        individual.fitness = correct

    def data_target(self):
        indx_batch = random.randint(0, self.num_batches - 1)
        with torch.no_grad():
            data = self.list_data[indx_batch]
            target = self.list_target[indx_batch]
            data, target = data.to(self.device), target.to(self.device)


            data =data.numpy()
            target= target.numpy()
        return data,target

    def fitness_eval(self, inv):
        indx_batch = random.randint(0, self.num_batches - 1)
        with torch.no_grad():
            data = self.list_data[indx_batch]
            target = self.list_target[indx_batch]
            data, target = data.to(self.device), target.to(self.device)

            output = inv.net(data)
            loss = F.nll_loss(output, target)
            running_loss = loss.item()

            inv.fitness = running_loss

    def worker_eval(self, indv):

        w = random.choice(self.workers)
        pvec =indv.net.extract_parameters()
        robj =w.eval.remote(pvec)

        return robj

    def step(self, pop, gen):

        if self.start == True:
            print("Initial Round")
            # mp=map(lambda x: self.fitness_eval(x), pop)
            fitness=[]

            for inv in pop:
                tstart = time.time()
                #self.fitness_eval(inv)  # Evaluate fitness of each indv
                #data,target=self.data_target()
                #fitness.append(fitness_eval_dist.remote(inv.net.extract_parameters(),data,target))
                fitness.append(self.worker_eval(inv))
                e = time.time() - tstart
                print("Take {} second to eval fitness".format(round(e, 2)))
                # self.get_fitness(inv)

                # self.get_correct_fitness(inv)

            fp =[]
            for f in fitness:
                fp.append(ray.get(f))

            for inv,f in zip(pop,fp):
                inv.fitness=f
            print("init pop score", fp)

            print("we got here safely")

            #print("Fittest", min(pop, key=attrgetter("fitness")).get_fitness())

        newpop = []
        counter = 0
        fitness_l =[]
        for i in range(self.selection_rounds):

            parentsa, parentsb = self.selTournament(individuals=pop, k=2, tournsize=self.tournament_size)

            if np.random.random() < self.cross_over_prob:

                childa, childb = self.cross_over_net(parentsa, parentsb)

                childa = self.mutate_indv(childa)
                childb = self.mutate_indv(childb)

                #self.fitness_eval(childa)
                #self.fitness_eval(childb)
                #data, target = self.data_target()
                #fitness_l.append(fitness_eval_dist.remote(childa.net.extract_parameters(), data, target))
                fitness_l.append(self.worker_eval(childa))
                #data, target = self.data_target()
                #fitness_l.append(fitness_eval_dist.remote(childb.net.extract_parameters(), data, target))
                fitness_l.append(self.worker_eval(childb))


                newpop.append(childa)
                newpop.append(childb)


            else:

                indv = parentsa
                indv = self.mutate_indv(indv)
                #self.fitness_eval(indv)
                #data, target = self.data_target()
                #fitness_l.append(fitness_eval_dist.remote(indv.net.extract_parameters(), data, target))
                fitness_l.append(self.worker_eval(indv))
                newpop.append(indv)

            counter += 1
            counter += 1

        for inv,f in zip(newpop,fitness_l):
            inv.fitness=ray.get(f)

        self.start = False
        print("Len pop", len(pop))
        print("Len newpop", len(newpop))

        #over_all = pop + newpop
        next_gen=self.selBest(newpop,self.popSize)
        #next_gen = self.selTournament(individuals=over_all, k=self.popSize, tournsize=self.tournament_size)

        # next_gen = sorted(over_all, key=attrgetter("fitness"))[:self.popSize]

        print("Len overall", len(newpop))
        fscore = [f.get_fitness() for f in next_gen]
        fscore = sorted(fscore)
        print("FSCORE top 10 ", fscore[:5])
        print("FSCORE worst 10 ", fscore[-5:])
        self.best_fitness = min(next_gen, key=attrgetter("fitness")).get_fitness()
        self.worst_fitness = max(next_gen, key=attrgetter("fitness")).get_fitness()
        print("Generation:", gen, " Best Fitness", self.best_fitness)
        self.gen_fitness.append(fscore[0])

        print("Next Gen Size", len(next_gen))
        print("Generation:", gen, "Original Generation", min(next_gen, key=attrgetter("fitness")).get_generation())
        print("NEXT GENl")
        # next_gen = sorted(over_all,key=attrgetter("fitness"))[:self.popSize]
        # print("Generation:", gen, "Fittest",next_gen[0].get_fitness())

        if (gen % self.chek_interval == 0) & (gen > self.chek_interval):
            gc.collect()
            changepct = self.change_ratio(self.chek_interval)
            self.adjust_exp(changepct)

        return next_gen

    def xovere(self, a, b):
        g, h = a.copy(), b.copy()
        for pt in range(len(g)):
            if np.random.random() < 0.5:
                # print("cross over genes parent1",g[pt], "parent2",h[pt])
                g[pt], h[pt] = h[pt], g[pt]
        return g, h

    def xover_frp_linear(self, a, b):
        """
        floating point representation cross over.
        using a Linear approach to crossing over.

        :param a: parent a
        :param b: parent b
        :return:  return two new off spring
        """

        # TODO right linear implemetation


        return True

    def xover_frp_directional(self, a, b):
        """
        floating point representation cross over.
        using a  directional method

        :param a: parent a
        :param b: prent b
        :return:  single off spring
        """

        # TODO write frp directional implmentation

        return True  # sinlge child

    def xover_frp_arithmetic(self, a, b, weight_y=0.5):
        """
        floating point representation cross over.
        using a Arithmetich method.
        Take the weighted average of two parents
        :param a:  parent a
        :param b:  parent b
        :param weight_y: the wieighted value default is 0.5 for getting the average
        :return:  return a single off pring
        """

        # TODO write frp Arithmetic

        return True

    def cxSimulatedBinary(self, ind1, ind2):
        """Executes a simulated binary crossover that modify in-place the input
        individuals. The simulated binary crossover expects :term:`sequence`
        individuals of floating point numbers.
        :param ind1: The first individual participating in the crossover.
        :param ind2: The second individual participating in the crossover.
        :param eta: Crowding degree of the crossover. A high eta will produce
                    children resembling to their parents, while a small eta will
                    produce solutions much more different.
        :returns: A tuple of two individuals.
        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        eta = 1 - self.mutation_rate
        for i, (x1, x2) in enumerate(zip(ind1, ind2)):
            rand = random.random()
            if rand <= 0.5:
                beta = 2. * rand
            else:
                beta = 1. / (2. * (1. - rand))
            beta **= 1. / (eta + 1.)
            ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
            ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))

        return ind1, ind2

    def xover_frp_blx_A(self, inda, indb, alpha=.7):
        """
        floating point representation cross over.
        using a BLX_A method


        :param self:
        :param inda: individual parent a
        :param b: individual parent b
        :param alpha:Extent of the interval in which the new values can be drawn
                    for each attribute on both side of the parents' attributes.
        :return: new off spring
        """

        # done implemtation of BLX_APLHA CROSS OVER
        for i, (x1, x2) in enumerate(zip(inda, indb)):
            gamma = (1. + 2. * alpha) * random.random() - alpha
            inda[i] = (1. - gamma) * x1 + gamma * x2
            indb[i] = gamma * x1 + (1. - gamma) * x2

        return inda, indb

    def cross_over_net(self, indv1, indv2):
        # print("Cross Over")
        inda = Net().to(self.device)
        indb = Net().to(self.device)

        # new_a, new_b = self.xover_frp_blx_A(indv1.genome,indv2.genome)
        new_a, new_b = self.cxSimulatedBinary(indv1.genome, indv2.genome)

        inda.inject_parameters(new_a)
        indb.inject_parameters(new_b)
        childa = Individual(generation=self.generation_now, net=inda, genome=new_a)
        childb = Individual(generation=self.generation_now, net=indb, genome=new_b)
        # self.fitness_eval(childa)
        # self.fitness_eval(childb)


        return childa, childb

    def change_ratio(self, numgens):
        if numgens >= len(self.gen_fitness):

            fitness = self.gen_fitness[-numgens:]

            oldest = fitness[0]
            latest = fitness[-1]

            diff = oldest - latest

            pct = (diff / oldest) * 100
        else:
            pct = 0

        return pct

    def adjust_exp(self, change):
        if change < 1:

            self.mutation_rate += self.mutation_rate * 0.01
            self.cross_over_prob += self.cross_over_prob * 0.01
            print(
                "@@@@@@@@@@@@ little change at {} adjusted mutation rate to {} and crossover rate to {}".format(change,
                                                                                                                self.mutation_rate,
                                                                                                                self.cross_over_prob))
        elif change > 1:
            self.mutation_rate -= self.mutation_rate * 0.01
            self.cross_over_prob -= self.cross_over_prob * 0.01
            print(" *********Good Change change at {} adjusted mutation rate to {} and crossover rate to {}".format(
                change,
                self.mutation_rate,
                self.cross_over_prob))

    def evolve(self):
        best_indv = None

        tstart = time.time()

        fig, ax = plt.subplots(3)
        num_plots = 0

        fitness_gen = []
        worst_fitness = []
        mutation_rates = []

        # TODO PLOT CROSS OVER RATE AND MURATION RATE ON SUBPLOT


        for i in tqdm(range(self.generations)):
            print("Generation----", i)
            self.generation_now = i
            tstartStep = time.time()
            newpop = self.step(self.curPop, i)
            elapsedStep = round(time.time() - tstartStep, 2)
            print("Time it took generatin {} is {} seconds".format(i, elapsedStep))

            self.prevPop = self.curPop
            self.curPop = newpop
            if self.best_fitness == 0.00:
                print("Solution has been found")
                best_indv = min(self.curPop, key=attrgetter("fitness"))
                break
            self.prevPop = self.curPop
            self.curPop = newpop
            mutation_rates.append(self.mutation_rate)
            fitness_gen.append(self.best_fitness)
            worst_fitness.append(self.worst_fitness)
            arr_worst = np.array(worst_fitness)
            arr_f = np.array(fitness_gen)
            arr_muts = np.array(mutation_rates)
            ax[0].clear()
            ax[1].clear()
            ax[2].clear()
            # ax[3].clear()

            ax[0].plot(arr_f, color="green")

            ax[0].set_title("Training loss vs generation:best")
            ax[0].set_ylabel("Log Loss")
            ax[0].set_xlabel("Generation")
            ax[0].legend(["Fitest"])

            ax[1].plot(arr_worst, color="red")
            ax[1].set_title("Training loss vs generation :worst")
            ax[1].set_ylabel("Log Loss")
            ax[1].set_xlabel("Generation")
            ax[1].legend(["Worst"])

            ax[2].plot(arr_muts, color="orange")
            ax[2].set_title("mutation_rate vs generation")
            ax[2].set_ylabel("Mutation rate")
            ax[2].set_xlabel("Generation")

            # count_muts = Counter(self.curr_gen_muts)


            # ax[3].bar(count_muts.keys(),count_muts.values())
            # ax[3].set_title("BAR Count of Mutatations Run")
            # ax[3].set_ylabel("Count")
            # ax[3].set_xlabel("Mutations")
            # ax[3].legend(count_muts.keys())


            elapsed = round(time.time() - tstart, 2)

            print("Time elasped  is {} secounds".format(elapsed))

            plt.pause(0.001)
            self.curr_gen_muts = []
            num_plots += 1

        best_indv = min(self.curPop, key=attrgetter("fitness"))
        plt.savefig("ran.jpg")

        return best_indv
