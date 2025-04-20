import tkinter as tk
from tkinter import ttk
import random
from itertools import combinations

# ========== بيانات وهمية ==========
teams = ["Team A", "Team B", "Team C", "Team D"]
venues = ["Stadium 1", "Stadium 2"]
days = ["Day 1", "Day 2", "Day 3", "Day 4"]

# ========== GA FUNCTIONS ==========
def generate_matches(teams):
    return list(combinations(teams, 2))

def create_individual(matches):
    individual = []
    for match in matches:
        day = random.choice(days)
        venue = random.choice(venues)
        individual.append((match[0], match[1], day, venue))
    return individual

def fitness(individual):
    score = 0
    schedule = {}
    for match in individual:
        t1, t2, day, _ = match
        if day not in schedule:
            schedule[day] = []
        schedule[day].append((t1, t2))

    for day_matches in schedule.values():
        played = set()
        for t1, t2 in day_matches:
            if t1 in played or t2 in played:
                score -= 1
            else:
                played.update([t1, t2])
    return score

def crossover(parent1, parent2):
    point = len(parent1) // 2
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, mutation_rate=0.1):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            t1, t2, _, _ = individual[i]
            new_day = random.choice(days)
            new_venue = random.choice(venues)
            individual[i] = (t1, t2, new_day, new_venue)
    return individual

def create_population(size, matches):
    return [create_individual(matches) for _ in range(size)]

def select_parents(population):
    sorted_pop = sorted(population, key=lambda ind: fitness(ind), reverse=True)
    return sorted_pop[0], sorted_pop[1]

def genetic_algorithm(generations=100, pop_size=20):
    matches = generate_matches(teams)
    population = create_population(pop_size, matches)

    for _ in range(generations):
        new_population = []
        for _ in range(pop_size):
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
        population = new_population

    best = max(population, key=lambda ind: fitness(ind))
    return best, fitness(best)

# ========== GUI ==========
def run_ga_and_display():
    result, score = genetic_algorithm()
    output_text.delete("1.0", tk.END)
    output_text.insert(tk.END, f"🏆 Best Fitness Score: {score}\n\n")
    output_text.insert(tk.END, "📅 Final Tournament Schedule:\n\n")
    for match in result:
        output_text.insert(tk.END, f"{match[0]} vs {match[1]} on {match[2]} at {match[3]}\n")

root = tk.Tk()
root.title("Sports Tournament Scheduler - GA")
root.geometry("600x450")
root.resizable(False, False)

title = tk.Label(root, text="⚽ Genetic Algorithm – Tournament Scheduler", font=("Arial", 16, "bold"))
title.pack(pady=10)

btn_run = tk.Button(root, text="Generate Tournament Schedule", font=("Arial", 12), command=run_ga_and_display)
btn_run.pack(pady=10)

output_text = tk.Text(root, wrap=tk.WORD, font=("Consolas", 11))
output_text.pack(padx=10, pady=10, expand=True, fill=tk.BOTH)

root.mainloop()
