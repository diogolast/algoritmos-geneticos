
import argparse
from src.problem.regression_problem import RegressionProblem
from src.problem.classification_problem import ClassificationProblem
from src.problem.tsp_problem import TSPProblem
from src.algorithm.genetic_algorithm import genetic_algorithm
import numpy as np


def generate_report(problem, name, execution, fitness, data):
    min_fitness = np.min(fitness)
    max_fitness = np.max(fitness)
    mean_fitness = np.mean(fitness)
    std_fitness = np.std(fitness)
    print("\n")
    print("Fitness \nMax: {0}\nMin: {1} \nMean: {2} \nStd: {3} \n\nBest Values: {4}"
            .format(max_fitness,min_fitness,mean_fitness,std_fitness, data))

    file_name = f"{name}{execution}"
    problem.plot(data, save_file=f"{file_name}.png")
    problem.plot_fitness(fitness,save_file=f"{file_name}-fitness.png")

    with open(f"report/{file_name}.txt", "w") as f:
        f.write(f"Fitness \nMax: {max_fitness}\nMin: {min_fitness} \nMean: {mean_fitness} \nStd: {std_fitness} \n\nBest Values: {data}")


def build_problem(problem_name):
    if problem_name == "classification":
        return ClassificationProblem("data/german_statlog/german.data-numeric")
    elif problem_name == "regression":
        return RegressionProblem("data/regression/data-3.txt")
    elif problem_name == "tsp":
        return TSPProblem("data/tsp/tsp-30.txt")
    else:
        raise NotImplementedError()


def read_command_line_args():
    parser = argparse.ArgumentParser(
        description='Optimization with genetic algorithms.')

    parser.add_argument('-p', '--problem', default='classification',
                        choices=["classification", "regression", "tsp"])
    parser.add_argument('-n', '--n_generations', type=int,
                        default=1000, help='number of generations.')
    parser.add_argument('-s', '--population_size', type=int,
                        default=200, help='population size.')
    parser.add_argument('-m', '--mutation_rate', type=float,
                        default=0.1, help='mutation rate.')

    args = parser.parse_args()
    return args


def main():
    args = read_command_line_args()

    problem = build_problem(args.problem)

    for execution in range(1,6):
        print(f"Execution {execution}/5")
        fitness = []
        data = []
        fitness, data = genetic_algorithm(
            problem,
            population_size=args.population_size,
            n_generations=args.n_generations)

        generate_report(problem,args.problem,execution,fitness, data)

    print("End of script!")


if __name__ == "__main__":
    main()
