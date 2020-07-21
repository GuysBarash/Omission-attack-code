import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def attack_genetic():
    ## parameters of genetic attack
    set_size = 20
    max_value_in_set = 500
    legal_options = np.array(list(range(max_value_in_set)))
    random.seed(64)
    number_of_generations = 1000
    population_size = 100
    chance_to_mutate_idx = 2.0 / set_size
    number_of_offsprings = int(population_size / 2)
    cxpb = 0.5
    mutpb = 0.45
    verbose = True

    def generate_random_create():
        ind = set(np.random.choice(legal_options, size=set_size, replace=False))
        ind = creator.Individual(ind)
        return ind

    # # To assure reproductibility, the RNG seed is set prior to the items
    # random.seed(64)

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", set, n=10, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_item", random.randrange, max_value_in_set)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_item, set_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):
        score = np.abs(np.array(list(individual)) - 100).mean()
        return score,

    def cxSet(ind1, ind2):
        pool = set.union(ind1, ind2)
        ind1 = creator.Individual(set(random.sample(pool, set_size)))
        ind2 = creator.Individual(set(random.sample(pool, set_size)))
        if len(ind1) != set_size or len(ind2) != set_size:
            j = 3
        return ind1, ind2

    def mutSet(individual):
        origin_creature = np.array(list(individual))
        random_creature = np.random.choice(np.setdiff1d(legal_options, origin_creature), size=set_size, replace=False)
        random_map = np.random.choice([0, 1], p=[1 - chance_to_mutate_idx, chance_to_mutate_idx], size=set_size)
        inv_random_map = 1 - random_map

        new_creature = (inv_random_map * origin_creature) + (random_map * random_creature)
        ind = creator.Individual(set(new_creature))
        return ind,

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", cxSet)
    toolbox.register("mutate", mutSet)
    toolbox.register("select", tools.selNSGA2)

    population = [generate_random_create() for _ in range(3)]  # toolbox.population(n=population_size)
    halloffame = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # algorithms.eaMuPlusLambda(pop, toolbox, population_size, LAMBDA, CXPB, MUTPB, number_of_generations, stats,
    #                           halloffame=hof)
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, number_of_generations + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, number_of_offsprings, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, population_size)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)


if __name__ == "__main__":
    attack_genetic()
