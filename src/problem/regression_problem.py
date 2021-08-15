
from src.problem.problem_interface import ProblemInterface
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

class RegressionProblem(ProblemInterface):
    def __init__(self, fname):
        self.file = fname
        input = pd.read_csv(self.file, header=None, delim_whitespace=True)
        self.x = input[0]
        self.y = input[1]
        self.mutation_rate = 0.1

    def fitness(self, individual):
        return mean_squared_error(self.y, individual, squared=False)
             
    def new_individual(self):
        individual = []
        c = [random.uniform(-100,100) for i in range(9)]
        for x in self.x:
            x = c[0] + \
            c[1]*math.sin(1*x) + \
            c[2]*math.cos(1*x) + \
            c[3]*math.sin(2*x) + \
            c[4]*math.cos(2*x) + \
            c[5]*math.sin(3*x) + \
            c[6]*math.cos(3*x) + \
            c[7]*math.sin(4*x) + \
            c[8]*math.cos(4*x)
            individual.append(x)

        return individual

    def crossover(self,p1, p2):
        alpha = np.random.uniform(0,1)
        individual = []
        for i in range(len(p1)):
            individual.append((alpha * p1[i]) + ((1 - alpha) * p2[i]))
        return individual


    def mutation(self, individual, mutation_rate):
        if(random.random() < mutation_rate):
            i = random.randint(0,len(individual)-1)
            if random.randint(0,1):
                new_value = random.uniform(-100,100)             
                individual[i] = new_value
            else:
                new_value = np.random.normal(0,1)
                individual[i] += new_value                
        return individual

    def selection(self, population):
        fitness = []
        selection_population = []
        for individual in population:
            fit = self.fitness(individual)
            fitness.append(fit)
        index = np.argsort(fitness)
        for i in index:
            selection_population.append(population[i])
        return selection_population

    def reproduction(self, population, mutation_rate):
        children = []
        best_fitness = population[0]
        for i in range(len(population)):
            child = self.crossover(best_fitness, self.new_individual())
            children.append(self.mutation(child,mutation_rate))
        return children             

    def plot(self, individual, save_file=None):
        plt.clf()
        plt.scatter(self.x, self.y,s=3, c='red')
        plt.plot(self.x, individual, c='blue')
        if save_file:
            plt.savefig(f"report/{save_file}")

    def plot_fitness(self,fitness,save_file=None):
        plt.clf()
        plt.plot(fitness, linewidth=5, color="black")
        plt.xlabel("Generation", fontsize=20)
        plt.ylabel("Fitness", fontsize=20)

        if save_file:
            plt.savefig(f"report/{save_file}")              
