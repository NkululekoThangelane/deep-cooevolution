import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import datasets, transforms
from Population import  Population
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import ray
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
def get_dataset(splits):
    batchsize =int(60000/ splits)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batchsize, shuffle=True)

    return  train_loader, test_loader
def eval(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item() # sum up batch loss
            #pred=output.data.max(1)
            pred = output.max(1,keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def plot_pop(list_fit):

    tl = [(gen,fit) for gen,fit in zip(range(len(list_fit)),list_fit)]

    labels = ["Generation","Log Loss"]
    df = pd.DataFrame.from_records(tl, columns=labels)
    df.to_csv("train_gen_res.csv", index=False)

    sns.set_style("darkgrid")
    plt.title('Center Title')
    ax = sns.pointplot(x="Generation", y="Log Loss", data=df)
    plt.savefig('traing_gen_plot.jpg')


def save_sol_pcl(model):

    torch.save(model,"model_save_pck")

def save_sol_state(model):
    torch.save(model.state_dict(), "model_save_state")





def main():
    print("GA Evolve a nueral network")
    print("---------------------------")
    ray.init(num_cpus=8)
    train_loader,test_loader =get_dataset(8)
    pop = Population(train_loader=train_loader)

    #pop.indv_batch()



    tstart = time.time()


    best =pop.evolve()
    print("ThE BEST Fitness",best.get_fitness())
    train_fitness =pop.gen_fitness
    plot_pop(train_fitness)

    print("Origial genetation", best.get_generation())

    eval(best.net,test_loader)

if __name__ == "__main__":
    main()
#TODO run COEvoltion idstributed 1 Population per worker
