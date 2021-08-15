
from src.problem.problem_interface import ProblemInterface
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

class TSPProblem(ProblemInterface):

    def __init__(self, fname):
        self.file = fname
        input = pd.read_csv(self.file, header=None, delim_whitespace=True)
        self.cities = input.values

    def distance(self, x, y):
        x_dis = self.cities[x]
        y_dis = self.cities[y]
        return ((x_dis[0] - y_dis[0])**2 + (x_dis[1] - y_dis[1])**2)**0.5 

    def fitness(self, individual):
        d = self.distance(0, individual[0]) #first city
        for i in range(1, len(individual)):
            prev_city = individual[i -1]
            city = individual[i]
            d += self.distance(prev_city,city)
        d += self.distance(individual[-1], 0) #end same city started
        return d

    def new_individual(self):
        individual = list(range(1, len(self.cities)))
        np.random.shuffle(individual)
        return individual

    def mutation(self, individual, mutation_rate):
        if(random.random() < mutation_rate):
            for swapped in range(2):
                swapWith = int(random.random() * len(individual))
                city1 = individual[swapped]
                city2 = individual[swapWith]
                
                individual[swapped] = city2
                individual[swapWith] = city1
        return individual

    def crossover(self, p1, p2):
        child = []
        childP1 = []
        childP2 = []
        
        geneA = int(random.random() * len(p1))
        geneB = int(random.random() * len(p1))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for i in range(startGene, endGene):
            childP1.append(p1[i])
            
        childP2 = [item for item in p2 if item not in childP1]

        child = childP1 + childP2

        return child   

    def selection(self, population):
        fitness = []
        selection_population = []
        for individual in population:
            fit = self.fitness(individual)
            fitness.append(fit)
        index = np.argsort(fitness)
        for i in index:
            #add zero for presetantion
            population[i].insert(0, 0)
            population[i].insert(len(population[i]), 0)

            selection_population.append(population[i])

        return selection_population
          

    def reproduction(self, population, mutation_rate):
        selection = []
        #remove 0 from selection function
        for individual in population:
            selection.append(individual[1:-1])

        eliteSize = len(selection) // 2
        children = []
        length = len(selection) - eliteSize
        pool = random.sample(selection, len(selection)) #create random population

        for i in range(0,eliteSize):
            children.append(selection[i]) #use few of best fitness to be parents
        
        for i in range(0, length):
            child = self.crossover(pool[i], pool[len(selection)-i-1])
            children.append(child)
        
        population = []
        for individual in children:
            population.append(self.mutation(individual,mutation_rate))

        return population    

    def plot(self, individual,save_file=None):
        x = [self.cities[individual[i]][0] for i in range(len(individual))]
        y = [self.cities[individual[i]][1] for i in range(len(individual))]
        plt.clf()
        plt.plot(x,y,'-o',c='blue')               
        if save_file:
            plt.savefig(f"report/{save_file}")

    def plot_fitness(self,fitness,save_file=None):
        plt.clf()
        plt.plot(fitness, linewidth=5, color="black")
        plt.xlabel("Generation", fontsize=20)
        plt.ylabel("Fitness", fontsize=20)

        if save_file:
            plt.savefig(f"report/{save_file}")