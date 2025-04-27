import random
from itertools import combinations
from collections import defaultdict

# Example data
teams = ["A", "B", "C", "D"]
venues = ["Stadium 1", "Stadium 2", "Stadium 3"]
days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
time_slots = [0, 1, 2]  # 3 slots per day

# Generate all unique matches
matches = list(combinations(teams, 2))

# Generate permutation individual
def create_permutation_individual():
    individual = matches.copy()
    random.shuffle(individual)
    return individual

# Penalty Function: Add penalties for violations of constraints
def calculate_penalty(schedule):
    penalty = 0
    team_days_played = defaultdict(list)  # track days each team plays
    used_slots = set()  # track used slots
    
    # Penalty for teams playing on the same day
    for match in schedule:
        team1, team2, day, time_slot, venue = match
        day_index = days.index(day)
        
        # If either team is playing on this day already
        if day_index in team_days_played[team1] or day_index in team_days_played[team2]:
            penalty += 10  # Penalty for same day match

        # Track the days teams played
        team_days_played[team1].append(day_index)
        team_days_played[team2].append(day_index)

        # Penalty for using the same venue at the same time slot (same venue conflict)
        if (day, time_slot, venue) in used_slots:
            penalty += 5  # Penalty for venue conflict
        used_slots.add((day, time_slot, venue))  # Add to used slots

    # Additional penalties for teams playing without sufficient rest
    for team, days_played in team_days_played.items():
        days_played.sort()
        for i in range(1, len(days_played)):
            # If a team played two days in a row without rest, add penalty
            if days_played[i] == days_played[i-1] + 1:
                penalty += 15  # Larger penalty for not having rest day

    return penalty

# Fitness Function: Include penalties in the fitness calculation
def fitness_function(schedule):
    penalty = calculate_penalty(schedule)
    fitness = len(schedule) - penalty  # More matches scheduled and less penalty is better
    return fitness

def assign_schedule_with_rest(permutation):
    schedule = []
    total_slots = len(days) * len(time_slots) * len(venues)
    if len(permutation) > total_slots:
        raise Exception("Too many matches for available slots!")

    team_days_played = defaultdict(list)  # Keep track of days each team plays
    used_slots = set()

    for match in permutation:
        team1, team2 = match
        placed = False

        # Try to place the match in a day and time slot that satisfies the constraints
        for day in days:
            day_index = days.index(day)
            
            if (day_index in team_days_played[team1]) or (day_index in team_days_played[team2]):
                continue
            if (day_index - 1 in team_days_played[team1]) or (day_index - 1 in team_days_played[team2]):
                continue

            # Find first available slot and venue
            for time_slot in time_slots:
                for venue in venues:
                    if (day, time_slot, venue) not in used_slots:
                        schedule.append((team1, team2, day, time_slot, venue))
                        used_slots.add((day, time_slot, venue))
                        team_days_played[team1].append(day_index)
                        team_days_played[team2].append(day_index)
                        placed = True
                        break
                if placed:
                    break
            if placed:
                break

        if not placed:
            raise Exception(f"Couldn't schedule match {team1} vs {team2} with sufficient rest!")

    return schedule

# Create initial population
def create_population(size):
    population = []
    for _ in range(size):
        individual = create_permutation_individual()
        schedule = assign_schedule_with_rest(individual)
        population.append(schedule)
    return population

# Parent Selection: Tournament Selection
def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    tournament_fitness = [fitness_function(schedule) for schedule in tournament]
    winner_index = tournament_fitness.index(max(tournament_fitness))
    return tournament[winner_index]

# Order Crossover
def order_crossover(parent1, parent2):
    # Randomly pick two crossover points
    start, end = sorted(random.sample(range(len(parent1)), 2))
    
    child1 = [None] * len(parent1)
    child2 = [None] * len(parent2)
    
    # Copy the part from parent1 to child1 and parent2 to child2
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]
    
    # Fill the remaining spots in child1 and child2
    p1_idx, p2_idx = 0, 0
    for i in range(len(parent1)):
        if child1[i] is None:
            while parent2[p2_idx] in child1:
                p2_idx += 1
            child1[i] = parent2[p2_idx]
        
        if child2[i] is None:
            while parent1[p1_idx] in child2:
                p1_idx += 1
            child2[i] = parent1[p1_idx]
    
    return child1, child2

# Swap Mutation
def swap_mutation(schedule):
    idx1, idx2 = random.sample(range(len(schedule)), 2)
    schedule[idx1], schedule[idx2] = schedule[idx2], schedule[idx1]
    return schedule

# Survivor Selection (Elitism)
def survivor_selection(population, offspring):
    # Combine population and offspring
    combined = population + offspring
    # Sort by fitness and select top individuals
    sorted_combined = sorted(combined, key=fitness_function, reverse=True)
    return sorted_combined[:len(population)]  # Return the top individuals

# Termination Condition Check
def run_genetic_algorithm(population_size=10, max_generations=100, convergence_threshold=1e-3, max_no_improvement=10):
    population = create_population(population_size)
    best_fitness = float('-inf')
    generations_without_improvement = 0
    
    for generation in range(max_generations):
        print(f"\nGeneration {generation+1}:")
        
        # Evaluate the fitness of the current population
        population_fitness = [fitness_function(schedule) for schedule in population]
        best_current_fitness = max(population_fitness)
        
        # Check for convergence (if the best fitness has not improved enough)
        if best_current_fitness - best_fitness < convergence_threshold:
            generations_without_improvement += 1
        else:
            best_fitness = best_current_fitness
            generations_without_improvement = 0
        
        # Terminate if there has been no improvement for 'max_no_improvement' generations
        if generations_without_improvement >= max_no_improvement:
            print("Termination condition reached: No improvement after several generations.")
            break
        
        # Create offspring through crossover and mutation
        offspring = []
        for _ in range(population_size // 2):  # Generate half the population size in offspring
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = order_crossover(parent1, parent2)
            mutated_child1 = swap_mutation(child1)
            mutated_child2 = swap_mutation(child2)
            offspring.extend([mutated_child1, mutated_child2])
        
        # Survivor selection to determine the next generation
        population = survivor_selection(population, offspring)
        
        # Print the fitness of the best individual
        best_individual = population[0]
        print(f"Best individual fitness: {fitness_function(best_individual)}")
        for match in best_individual:
            print(f"{match[0]} vs {match[1]} on {match[2]} - Slot {match[3]} at {match[4]}")

# Run the genetic algorithm
run_genetic_algorithm()
