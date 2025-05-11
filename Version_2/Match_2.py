import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import random
from itertools import combinations
from collections import defaultdict
import csv
import matplotlib.pyplot as plt
import math
import statistics

# Example data
teams = [
    "Manchester City", "Arsenal", "Liverpool", "Chelsea","Manchester United","New Castle",
    "Tottenham","Brentford","Brighton","Everton"
]

venues = [
    "Etihad Stadium", "Emirates Stadium", "Anfield", "Stamford Bridge", "Old Trafford",
    "Wembley","Goodison Park","St. James' Park","Villa Park","Elland Road"
]

days = [f"Day {i}" for i in range(1,41)] 

time_slots = ["2:00-4:00", "5:00-7:00", "8:00-10:00"]

# Generate all unique matches
matches = list(combinations(teams, 2))

#============== Population ============
# Helper function to generate all possible slots
def all_possible_slots():
    slots = []
    for day in days:
        for time_slot in time_slots:
            for venue in venues:
                slots.append((day, time_slot, venue))
    return slots

# Generate random individual
def create_individual():
    schedule = []
    available_slots = all_possible_slots()
    random.shuffle(available_slots)

    if len(matches) > len(available_slots):
        raise Exception("Too many matches for available slots!")

    used_slots = set()
    team_played_day = defaultdict(set)

    matches_copy = matches.copy()
    random.shuffle(matches_copy)

    for match in matches_copy:
        assigned = False
        for slot in available_slots:
            day, time_slot, venue = slot
            if slot in used_slots:
                continue
            team1, team2 = match
            if (day in team_played_day[team1]) or (day in team_played_day[team2]):
                continue
            schedule.append((team1, team2, day, time_slot, venue))
            used_slots.add(slot)
            team_played_day[team1].add(day)
            team_played_day[team2].add(day)
            assigned = True
            break
        
        if not assigned:
            raise Exception("Couldn't assign all matches without conflicts!")
    
    return schedule

# Population creation with diversity
def create_population(size):
    population = []
    while len(population) < size:
        individual = create_individual()
        # Check if the individual is already in the population (to avoid identical schedules)
        if individual not in population:
            population.append(individual)
    return population


#============== Fitness Function ==================
# Base Fitness Function
def fitness_function(schedule):
    penalty = 0
    bonus = 0
    team_schedule = defaultdict(list)
    venue_schedule = defaultdict(list)

    for match in schedule:
        team1, team2, day, time_slot, venue = match
        day_index = days.index(day)

        team_schedule[team1].append((day_index, time_slot))
        team_schedule[team2].append((day_index, time_slot))

        venue_schedule[(day, time_slot, venue)].append((team1, team2))

    for team, plays in team_schedule.items():
        plays.sort()
        played_days = [d for d, _ in plays]

        if len(set(played_days)) < len(played_days):
            penalty += 50

        for i in range(1, len(played_days)):
            if played_days[i] - played_days[i - 1] == 1:
                penalty += 20

        if len(set(played_days)) > 1:
            try:
                var = statistics.variance(played_days)
                bonus += var * 0.5  
            except statistics.StatisticsError:
                pass  

    for slot, matches_at_slot in venue_schedule.items():
        if len(matches_at_slot) > 1:
            penalty += 100 * (len(matches_at_slot) - 1)

    fitness = max(0, 1000 - penalty + bonus)
    return fitness


#=============Genetic Algorithm Functions============

# Tournament Selection
def tournament_selection(population, pool_size, tournament_size=5):
    mating_pool = []

    for _ in range(pool_size):
        # Pick k individuals randomly (without replacement) for the tournament
        tournament = random.sample(population, tournament_size)
        
        # Calculate fitness for each individual in the tournament
        fitnesses = [fitness_function(ind) for ind in tournament]
        # Select the best (deterministic selection: p = 1)
        winner_index = fitnesses.index(max(fitnesses))
        winner = tournament[winner_index]
        
        # Add winner to mating pool
        mating_pool.append(winner)

    return mating_pool

# Order Crossover (OX)
def order_crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() >= crossover_rate:
        # No crossover, return the parents as they are
        return parent1, parent2

    # randomly select two points for the crossover
    point1 = random.randint(0, len(parent1) - 1)
    point2 = random.randint(point1 + 1, len(parent1))  # Make sure point2 > point1

    # Extract the segment from parent1
    segment = parent1[point1:point2]
    offspring1 = [match for match in parent2 if match not in segment]
    offspring1 = offspring1[:point1] + segment + offspring1[point1:]

    segment2 = parent2[point1:point2]
    offspring2 = [match for match in parent1 if match not in segment2]
    offspring2 = offspring2[:point1] + segment2 + offspring2[point1:]

    return offspring1, offspring2

# Swap Mutation
def swap_mutation(schedule, mutation_rate=0.1):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(schedule)), 2)
        schedule[idx1], schedule[idx2] = schedule[idx2], schedule[idx1]

        match_to_mutate = random.choice(schedule)
        match_to_mutate = list(match_to_mutate)
        match_to_mutate[2] = random.choice(days)
        match_to_mutate[4] = random.choice(venues)

        for i in range(len(schedule)):
            if schedule[i][0] == match_to_mutate[0] and schedule[i][1] == match_to_mutate[1]:
                schedule[i] = tuple(match_to_mutate)
                break

    return schedule

# Survivor Selection (elitism)
def survivor_selection(population, offspring):
    combined = population + offspring
    combined_sorted = sorted(combined, key=lambda ind: fitness_function(ind), reverse=True)
    return combined_sorted[:len(population)]



NUM_ISLANDS = 4
MIGRATION_INTERVAL = 10  # generations
MIGRATION_COUNT = 2      # individuals to migrate between islands

def create_islands(pop_size, num_islands):
    island_size = pop_size // num_islands
    return [create_population(island_size) for _ in range(num_islands)]

#====================Run Genetic Algorithm=============

def local_search(individual, probability=0.2):
    if random.random() < probability:
        new_individual = individual[:]
        i, j = random.sample(range(len(new_individual)), 2)
        new_individual[i], new_individual[j] = new_individual[j], new_individual[i]
        if fitness_function(new_individual) > fitness_function(individual):
            return new_individual
    return individual

def run_island_genetic_algorithm(
    population_size=40,
    max_generations=5000,
    mutation_rate=0.4,
    crossover_rate=0.9,
    local_search_prob=0.4
):
    islands = create_islands(population_size, NUM_ISLANDS)

    # 🔍 Apply local search to initial population (Step 1)
    for i in range(NUM_ISLANDS):
        for j in range(len(islands[i])):
            original = islands[i][j]
            improved = local_search(original, local_search_prob)
            islands[i][j] = improved

    best_overall = None
    best_fitness_overall = float('-inf')
    fitness_history = []

    for generation in range(max_generations):
        print(f"\n=== Generation {generation + 1} ===")

        for i in range(NUM_ISLANDS):
            population = islands[i]
            offspring = []

            for _ in range(len(population) // 2):
                parent1, parent2 = tournament_selection(population, pool_size=2, tournament_size=3)

                # 🔁 Crossover
                if random.random() < crossover_rate:
                    child1, child2 = order_crossover(parent1, parent2, crossover_rate)
                else:
                    child1, child2 = parent1[:], parent2[:]

                # 🔧 Mutation
                child1 = swap_mutation(child1, mutation_rate)
                child2 = swap_mutation(child2, mutation_rate)

                # 🔍 Local search on offspring (Step 3)
                child1 = local_search(child1, local_search_prob)
                child2 = local_search(child2, local_search_prob)

                offspring.extend([child1, child2])

            # 🌱 Replace population
            islands[i] = survivor_selection(population, offspring)

        # 🔁 Migration between islands
        if generation % MIGRATION_INTERVAL == 0 and generation > 0:
            print(">> Migration happening between islands...")
            for i in range(NUM_ISLANDS):
                source_island = islands[i]
                target_island = islands[(i + 1) % NUM_ISLANDS]

                migrants = sorted(source_island, key=lambda ind: fitness_function(ind), reverse=True)[:MIGRATION_COUNT]
                replacees = sorted(target_island, key=lambda ind: fitness_function(ind))[:MIGRATION_COUNT]

                for m, r in zip(migrants, replacees):
                    idx = target_island.index(r)
                    target_island[idx] = m

        # 🏆 Track best
        for pop in islands:
            best_in_island = max(pop, key=fitness_function)
            fitness = fitness_function(best_in_island)
            if fitness > best_fitness_overall:
                best_fitness_overall = fitness
                best_overall = best_in_island

        fitness_history.append(best_fitness_overall)

    print(f"\nTerminating: Reached maximum number of generations ({max_generations}).")
    return best_overall, best_fitness_overall, fitness_history

# === GUI Helper Functions ===
current_schedule = []

def run_ga_thread():
    def task():
        try:
            pop_size = int(entry_pop.get())
            generations = int(entry_gen.get())
            mutation = float(entry_mut.get())
            cross = float(entry_cross.get())
            seed_val = int(entry_seed.get())
            
            # Set the seed BEFORE anything random happens
            random.seed(seed_val)

            # schedule, score = run_genetic_algorithm(pop_size, generations, mutation_rate=mutation, crossover_rate=cross)
            schedule, score ,fitness_history = run_island_genetic_algorithm(pop_size, generations, mutation_rate=mutation, crossover_rate=cross)
            current_schedule.clear()
            current_schedule.extend(schedule)
            
            lbl_score["text"] = f"Best Fitness: {score:.2f}"
            tree.delete(*tree.get_children())
            for match in schedule:
                tree.insert("", "end", values=(f"{match[0]} vs {match[1]}", match[2], match[4], f"Slot {match[3]}"))
            
            # fitness_history.reverse()
            plt.figure(figsize=(10, 4))
            plt.plot(fitness_history,  linestyle='-', color='blue')
            plt.title("Best Fitness Over Generations")
            plt.xlabel("Generation")
            plt.ylabel("Fitness")
            plt.grid(True)
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", str(e))
    threading.Thread(target=task).start()

# Save schedule
def save_schedule(schedule, fmt):
    if not schedule:
        messagebox.showinfo("Info", "No schedule to save.")
        return

    file = filedialog.asksaveasfilename(defaultextension=".csv" if fmt == "CSV" else ".txt")
    if not file:
        return

    generations = entry_gen.get()
    population = entry_pop.get()
    mutation = entry_mut.get()
    crossover = entry_cross.get()
    fitness_val = lbl_score["text"]
    seed_val = entry_seed.get()

    with open(file, "w", newline="") as f:
        if fmt == "CSV":
            writer = csv.writer(f)
            writer.writerow(["Generations", generations])
            writer.writerow(["Population Size", population])
            writer.writerow(["Mutation Rate", mutation])
            writer.writerow(["Crossover Rate", crossover])
            writer.writerow(["Fitness", fitness_val])
            writer.writerow(["Seed", seed_val])
            writer.writerow([])  # Empty line
            writer.writerow(["Match", "Day", "Venue", "Time Slot"])
            for match in schedule:
                line = [f"{match[0]} vs {match[1]}", match[2], match[4], f"Slot {match[3]}"]
                writer.writerow(line)
        else:
            f.write(f"Generations: {generations}\n")
            f.write(f"Population Size: {population}\n")
            f.write(f"Mutation Rate: {mutation}\n")
            f.write(f"Crossover Rate: {crossover}\n")
            f.write(f"{fitness_val}\n\n")
            f.write(f"Seed: {seed_val}\n")
            f.write("Match | Day | Venue | Time Slot\n")
            for match in schedule:
                line = [f"{match[0]} vs {match[1]}", match[2], match[4], f"Slot {match[3]}"]
                f.write(" | ".join(line) + "\n")

# GUI Code 
def add_team():
    new_team = entry_team.get().strip()
    if new_team and new_team not in teams:
        teams.append(new_team)
        messagebox.showinfo("Success", f"Team '{new_team}' added.")
        update_match_list()  
    entry_team.delete(0, tk.END)

def update_match_list():
    global matches
    matches = list(combinations(teams, 2))

def add_venue():
    name = entry_venue.get()
    if name and name not in venues:
        venues.append(name)
        messagebox.showinfo("Success", f"Venue '{name}' added.")

def add_day():
    name = entry_day.get()
    if name and name not in days:
        days.append(name)
        messagebox.showinfo("Success", f"Day '{name}' added.")

def apply_theme():
    if is_dark_mode.get():
        # Dark Theme
        root.configure(bg="#1e1e1e")
        style.configure("TLabel", background="#1e1e1e", foreground="#d4d4d4")
        style.configure("TEntry", fieldbackground="#2d2d2d", foreground="#d4d4d4", insertcolor="#d4d4d4")
        style.configure("TButton", background="#007acc", foreground="white")
        style.configure("TCombobox", fieldbackground="#2d2d2d", foreground="#d4d4d4", background="#2d2d2d")
        style.configure("Treeview", background="#252526", foreground="#dcdcdc", fieldbackground="#252526")
        style.configure("Treeview.Heading", background="#007acc", foreground="white")
        style.configure("TLabelframe", background="#1e1e1e", foreground="#d4d4d4")
        style.configure("TLabelframe.Label", background="#1e1e1e", foreground="#007acc")
    else:
        # Light Theme
        root.configure(bg="#eaf6fb")
        style.configure("TLabel", background="#eaf6fb", foreground="#003366")
        style.configure("TEntry", fieldbackground="#ffffff", foreground="#003366", insertcolor="#003366")
        style.configure("TButton", background="#007acc", foreground="white")
        style.configure("TCombobox", fieldbackground="#ffffff", foreground="#003366", background="#ffffff")
        style.configure("Treeview", background="#ffffff", foreground="#003366", fieldbackground="#ffffff")
        style.configure("Treeview.Heading", background="#007acc", foreground="white")
        style.configure("TLabelframe", background="#eaf6fb", foreground="#003366")
        style.configure("TLabelframe.Label", background="#eaf6fb", foreground="#007acc")

# GUI setup
root = tk.Tk()
root.title("⚽ Tournament Scheduler via GA")
root.geometry("1000x700")

# Style object
style = ttk.Style()
style.theme_use("default")

# Theme toggle
is_dark_mode = tk.BooleanVar(value=False)
dark_toggle = ttk.Checkbutton(root, text="Dark Mode", variable=is_dark_mode, command=apply_theme)
dark_toggle.pack(anchor="ne", padx=10, pady=5)

# Title
lbl_title = ttk.Label(root, text="⚽ Sports Tournament Schedule (GA)", font=("Segoe UI", 18, "bold"))
lbl_title.pack(pady=10)

# --- Parameters Frame ---
frame_top = ttk.LabelFrame(root, text="Genetic Algorithm Parameters", padding=10)
frame_top.pack(padx=10, pady=10, fill="x")

labels = ["Generations", "Population", "Mutation", "Crossover"]
defaults = ["50", "10", "0.1", "0.7"]
entries = []

for i, (label_text, default) in enumerate(zip(labels, defaults)):
    ttk.Label(frame_top, text=label_text).grid(row=1, column=i, padx=5, pady=2)
    entry = ttk.Entry(frame_top, width=6)
    entry.insert(0, default)
    entry.grid(row=0, column=i, padx=5)
    entries.append(entry)

entry_gen, entry_pop, entry_mut, entry_cross = entries

# Seed
ttk.Label(frame_top, text="Seed").grid(row=1, column=4, padx=5)
entry_seed = ttk.Entry(frame_top, width=8)
entry_seed.insert(0, "42")
entry_seed.grid(row=0, column=4, padx=5)

# Buttons and Format
btn_generate = ttk.Button(frame_top, text="Generate", command=run_ga_thread)
btn_generate.grid(row=0, column=7, padx=10)

combo_format = ttk.Combobox(frame_top, values=["CSV", "Text"], state="readonly", width=7)
combo_format.set("CSV")
combo_format.grid(row=0, column=5, padx=5)

btn_save = ttk.Button(frame_top, text="Save", command=lambda: save_schedule(current_schedule, combo_format.get()))
btn_save.grid(row=0, column=6, padx=5)

lbl_score = ttk.Label(frame_top, text="", font=("Segoe UI", 10, "italic"))
lbl_score.grid(row=0, column=8, padx=10)

# --- TreeView for Schedule ---
cols = ("Match", "Day", "Venue", "Time")
tree = ttk.Treeview(root, columns=cols, show='headings', height=14)
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, width=200, anchor='center')
tree.pack(pady=10, padx=10, fill="x")

# --- Add Inputs Frame ---
frame_add = ttk.LabelFrame(root, text="Add Items", padding=10)
frame_add.pack(pady=10)

entry_team = ttk.Entry(frame_add, width=15)
entry_team.grid(row=0, column=0, padx=5)
btn_team = ttk.Button(frame_add, text="Add Team", command=add_team)
btn_team.grid(row=0, column=1, padx=5)

entry_venue = ttk.Entry(frame_add, width=15)
entry_venue.grid(row=0, column=2, padx=5)
btn_venue = ttk.Button(frame_add, text="Add Venue", command=add_venue)
btn_venue.grid(row=0, column=3, padx=5)

entry_day = ttk.Entry(frame_add, width=15)
entry_day.grid(row=0, column=4, padx=5)
btn_day = ttk.Button(frame_add, text="Add Day", command=add_day)
btn_day.grid(row=0, column=5, padx=5)

# Apply initial (light) theme
apply_theme()
root.mainloop()
