def genetic_algorithm(problem, population_size, n_generations, mutation_rate=0.1):
    fitness = []
    fitness_value = 999999

    #create population
    population = []
    for _ in range(0,population_size):
        population.append(problem.new_individual())
        
    for gen in range(n_generations):
        #order population by fitness
        selection = problem.selection(population)
        if problem.fitness(selection[0]) < fitness_value:
            fitness_value = problem.fitness(selection[0])
            data = selection[0].copy()
        fitness.append(fitness_value)
        
        #reproduct population
        population = problem.reproduction(selection, mutation_rate)

    return fitness, data