
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from src.problem.problem_interface import ProblemInterface


class ClassificationProblem(ProblemInterface):
    def __init__(self, fname):
        # load dataset
        with open(fname, "r") as f:
            lines = f.readlines()

        # For each line l, remove the "\n" at the end using
        # rstrip and then split the string the spaces. After
        # this instruction, lines is a list in which each element
        # is a sublist of strings [s1, s2, s3, ..., sn].
        lines = [l.rstrip().rsplit() for l in lines]

        # Convert the list of list into a numpy matrix of integers.
        lines = np.array(lines).astype(np.int32)

        # Split features (x) and labels (y). The notation x[:, i]
        # returns all values of column i. To learn more about numpy indexing
        # see https://numpy.org/doc/stable/reference/arrays.indexing.html .
        x = lines[:, :-1]
        y = lines[:, -1]

        # Split the data in two sets without intersection.
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(x, y, test_size=0.30,
                             stratify=y, random_state=871623)

        # number of features
        self.n_features = self.X_train.shape[1]

        # search space for the values of k and metric
        self.Ks = [1, 3, 5, 7, 9, 11, 13, 15]
        self.metrics = ["euclidean", "hamming", "canberra", "braycurtis"]

    def new_individual(self):
        random_number = np.random.randint(2, size=self.n_features)
        individual = np.append(random_number,[self.Ks[random.randint(0,7)],random.randint(0,3)])
        return individual

    def fitness(self, individual):
        binary_pattern = individual[:-2]
        K = individual[-2]
        metric = self.metrics[individual[-1]]

        # return the indices of the features that are not zero.
        indices = np.nonzero(binary_pattern)[0]

        # check if there is at least one feature available
        if len(indices) == 0:
            return 1e6

        # select a subset of columns given their indices
        x_tr = self.X_train[:, indices]
        x_val = self.X_val[:, indices]

        # build the classifier
        knn = KNeighborsClassifier(n_neighbors=K, metric=metric)
        # train
        knn = knn.fit(x_tr, self.y_train)
        # predict the classes for the validation set
        y_pred = knn.predict(x_val)
        # measure the accuracy
        acc = np.mean(y_pred == self.y_val)

        # since the optimization algorithms minimize,
        # the fitness is defiend as the inverse of the accuracy
        fitness = -acc
 
        return fitness

    def mutation(self, individual):   
        metric = random.randint(0,3)
        while individual[-1] == metric:
            metric = random.randint(0,3)
        individual[-1] = metric
        return individual

    def crossover(self, p1, p2):
        kp1 = p1[-2]
        kp2 = p2[-2]
        metric1 = p1[-1]
        metric2 = p2[-1]
        p1 = p1[:-2]
        p2 = p2[:-2]
        children = []
         
        geneA = int(random.random() * len(p1))
        geneB = int(random.random() * len(p1))
        
        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        for _ in range(2):
            childP1 = []
            childP2 = []            
            for i in range(startGene, endGene):
                childP1.append(p1[i])
                
            childP2 = [p2[i] for i in range(len(p1)-len(childP1))]

            children.append(childP1 + childP2)
        
        children[0] = (np.append(children[0],[kp1,metric2]))
        children[1] = (np.append(children[1],[kp2,metric1]))
         
        return children

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


    def reproduction(self, population, mutation_rate = None):
        new_population = []
        best_fitness = population[0]
        for i in range(len(population) // 2):
            children = self.crossover(best_fitness, self.new_individual())
            for child in children:
                new_population.append(self.mutation(child))

        return new_population    

    def plot(self, individual,save_file=None):
        pass

    def plot_fitness(self,fitness,save_file=None):
        plt.clf()
        plt.plot(fitness, linewidth=5, color="black")
        plt.xlabel("Generation", fontsize=20)
        plt.ylabel("Fitness", fontsize=20)

        if save_file:
            plt.savefig(f"report/{save_file}")     

