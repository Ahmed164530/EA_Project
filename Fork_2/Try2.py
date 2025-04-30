import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import random
from itertools import combinations
from collections import defaultdict
import csv

# Example data
teams = ["A", "B", "C", "D"]
venues = ["Stadium 1", "Stadium 2", "Stadium 3"]
days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5"]
time_slots = [0, 1, 2]  # 3 slots per day

# Generate all unique matches
matches = list(combinations(teams, 2))

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
    
    for match in matches:
        assigned = False
        random.shuffle(available_slots)
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

# Base Fitness Function
def fitness_function(schedule):
    penalty = 0
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
            if played_days[i] - played_days[i-1] == 1:
                penalty += 20

    for slot, matches_at_slot in venue_schedule.items():
        if len(matches_at_slot) > 1:
            penalty += 100 * (len(matches_at_slot) - 1)

    fitness = 1000 - penalty  
    return fitness

# Fitness Sharing Function
def shared_fitness(individual, population, sigma_share=0.3, alpha=1):
    def similarity(ind1, ind2):
        shared = sum(1 for m1, m2 in zip(ind1, ind2) if m1[:3] == m2[:3])  # same teams and day
        return shared / len(ind1)
    
    sh_sum = sum(
        (1 - (similarity(individual, other) / sigma_share) ** alpha) if similarity(individual, other) < sigma_share else 0
        for other in population
    )
    raw_fitness = fitness_function(individual)
    return raw_fitness / (1 + sh_sum)

# Population creation
def create_population(size):
    return [create_individual() for _ in range(size)]

#Tournament Selection (with shared fitness)
def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    fitnesses = [shared_fitness(ind, population) for ind in tournament]
    winner_index = fitnesses.index(max(fitnesses))
    return tournament[winner_index]

# Order Crossover
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    parent1_matches = [(m[0], m[1]) for m in parent1]
    parent2_matches = [(m[0], m[1]) for m in parent2]

    child1_matches = [None] * size
    child2_matches = [None] * size

    child1_matches[start:end] = parent1_matches[start:end]
    child2_matches[start:end] = parent2_matches[start:end]

    def fill_child(child, other_parent):
        idx = 0
        for match in other_parent:
            if match not in child:
                while child[idx] is not None:
                    idx += 1
                child[idx] = match
        return child

    child1_matches = fill_child(child1_matches, parent2_matches)
    child2_matches = fill_child(child2_matches, parent1_matches)

    available_slots = all_possible_slots()
    random.shuffle(available_slots)
    
    child1 = []
    child2 = []
    
    for match, slot in zip(child1_matches, available_slots):
        team1, team2 = match
        day, time_slot, venue = slot
        child1.append((team1, team2, day, time_slot, venue))
    
    for match, slot in zip(child2_matches, available_slots):
        team1, team2 = match
        day, time_slot, venue = slot
        child2.append((team1, team2, day, time_slot, venue))
    
    return child1, child2

# Swap Mutation
def swap_mutation(schedule, mutation_rate=0.2):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(schedule)), 2)
        schedule[idx1], schedule[idx2] = schedule[idx2], schedule[idx1]

        match_to_mutate = random.choice(schedule)
        match_to_mutate = list(match_to_mutate)
        match_to_mutate[3] = random.choice(time_slots)
        match_to_mutate[4] = random.choice(venues)

        for i in range(len(schedule)):
            if schedule[i][0] == match_to_mutate[0] and schedule[i][1] == match_to_mutate[1]:
                schedule[i] = tuple(match_to_mutate)
                break
    
    return schedule

#Survivor Selection (elitism)
def survivor_selection(population, offspring):
    combined = population + offspring
    combined_sorted = sorted(combined, key=lambda ind: shared_fitness(ind, combined), reverse=True)
    return combined_sorted[:len(population)]

# Run Genetic Algorithm
def run_genetic_algorithm(population_size=10, max_generations=100, convergence_threshold=1e-3, max_no_improvement=10):
    population = create_population(population_size)
    best_fitness = float('-inf')
    generations_without_improvement = 0

    # Initial parameters
    mutation_rate = 0.1

    for generation in range(max_generations):
        print(f"\nGeneration {generation + 1}:")
        # fitnesses = [fitness_function(ind) for ind in population]
        fitnesses = [shared_fitness(ind, population) for ind in population]
        best_current_fitness = max(fitnesses)
        best_index = fitnesses.index(best_current_fitness)
        best_individual = population[best_index]

        if best_current_fitness - best_fitness < convergence_threshold:
            generations_without_improvement += 1
        else:
            best_fitness = best_current_fitness
            generations_without_improvement = 0

        print(f"Best fitness: {best_current_fitness}")
        for match in best_individual:
            print(f"{match[0]} vs {match[1]} on {match[2]} - Slot {match[3]} at {match[4]}")

        if generations_without_improvement >= max_no_improvement:
            print("\nTermination condition reached: No improvement after several generations.")
            break

        # Adjust mutation rate adaptively
        if generations_without_improvement > 5:
            mutation_rate = min(1.0, mutation_rate * 1.1)  # exploration
        else:
            mutation_rate = max(0.01, mutation_rate * 0.9)  # exploitation

        offspring = []
        for _ in range(population_size // 2):
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child1, child2 = order_crossover(parent1, parent2)
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)
            offspring.extend([child1, child2])

        population = survivor_selection(population, offspring)
    return best_individual, best_fitness

# Run
run_genetic_algorithm()

# === GUI Integration ===
current_schedule = []

def run_ga_thread():
    def task():
        global current_schedule
        try:
            gens = int(entry_gen.get())
            pop = int(entry_pop.get())
            mut = float(entry_mut.get())
            best, score = run_genetic_algorithm(pop, gens, mut)
            current_schedule = best
            lbl_score.config(text=f"Best Fitness: {int(score)}")
            tree.delete(*tree.get_children())
            for match in best:
                team1, team2, day, time, venue = match
                tree.insert('', 'end', values=(f"{team1} vs {team2}", day, venue, f"Slot {time}"))
        except Exception as e:
            messagebox.showerror("Error", str(e))
    threading.Thread(target=task).start()

def save_schedule(schedule, format):
    if not schedule:
        messagebox.showwarning("No Schedule", "Generate schedule first.")
        return
    filetypes = [("CSV Files", "*.csv")] if format == "CSV" else [("Text Files", "*.txt")]
    ext = ".csv" if format == "CSV" else ".txt"
    path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=filetypes)
    if not path:
        return
    with open(path, 'w', newline='') as f:
        if format == "CSV":
            writer = csv.writer(f)
            writer.writerow(["Match", "Day", "Venue", "Time"])
            for match in schedule:
                writer.writerow([f"{match[0]} vs {match[1]}", match[2], match[4], f"Slot {match[3]}"])
        else:
            for match in schedule:
                f.write(f"{match[0]} vs {match[1]} on {match[2]} at {match[4]} (Slot {match[3]})\n")

# === GUI Code ===
def run_ga_thread():
    def task():
        try:
            pop_size = int(entry_pop.get())
            generations = int(entry_gen.get())
            mutation = float(entry_mut.get())
            schedule, score = run_genetic_algorithm(pop_size, generations, mutation)
            current_schedule.clear()
            current_schedule.extend(schedule)
            lbl_score["text"] = f"Best Fitness: {score:.2f}"
            tree.delete(*tree.get_children())
            for match in schedule:
                tree.insert("", "end", values=(f"{match[0]} vs {match[1]}", match[2], match[4], f"Slot {match[3]}"))
        except Exception as e:
            messagebox.showerror("Error", str(e))
    threading.Thread(target=task).start()

def save_schedule(schedule, fmt):
    if not schedule:
        messagebox.showinfo("Info", "No schedule to save.")
        return
    file = filedialog.asksaveasfilename(defaultextension=".csv" if fmt == "CSV" else ".txt")
    if not file:
        return
    with open(file, "w", newline="") as f:
        writer = csv.writer(f) if fmt == "CSV" else f
        for match in schedule:
            line = [f"{match[0]} vs {match[1]}", match[2], match[4], f"Slot {match[3]}"]
            if fmt == "CSV":
                writer.writerow(line)
            else:
                f.write(" | ".join(line) + "\n")
       
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

root = tk.Tk()
root.title("⚽ Tournament Scheduler via GA")
root.geometry("1000x700")

lbl_title = tk.Label(root, text="Sports Tournament Schedule (GA)", font=("Arial", 16, "bold"))
lbl_title.pack(pady=10)

frame_top = tk.Frame(root)
frame_top.pack()

entry_gen = tk.Entry(frame_top, width=5)
entry_gen.insert(0, "50")
entry_gen.grid(row=0, column=0)

entry_pop = tk.Entry(frame_top, width=5)
entry_pop.insert(0, "10")
entry_pop.grid(row=0, column=1)

entry_mut = tk.Entry(frame_top, width=5)
entry_mut.insert(0, "0.1")
entry_mut.grid(row=0, column=2)

btn_generate = tk.Button(frame_top, text="Generate", command=run_ga_thread)
btn_generate.grid(row=0, column=3, padx=5)

lbl_score = tk.Label(frame_top, text="")
lbl_score.grid(row=0, column=4, padx=10)

combo_format = ttk.Combobox(frame_top, values=["CSV", "Text"], state="readonly", width=6)
combo_format.set("CSV")
combo_format.grid(row=0, column=5)

btn_save = tk.Button(frame_top, text="Save", command=lambda: save_schedule(current_schedule, combo_format.get()))
btn_save.grid(row=0, column=6, padx=5)

cols = ("Match", "Day", "Venue", "Time")
tree = ttk.Treeview(root, columns=cols, show='headings', height=12)
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, width=200, anchor='center')
tree.pack(pady=10)

frame_add = tk.Frame(root)
frame_add.pack(pady=10)

entry_team = tk.Entry(frame_add, width=10)
entry_team.grid(row=0, column=0)
btn_team = tk.Button(frame_add, text="Add Team", command=add_team)
btn_team.grid(row=0, column=1)

entry_venue = tk.Entry(frame_add, width=10)
entry_venue.grid(row=0, column=2)
btn_venue = tk.Button(frame_add, text="Add Venue", command=add_venue)
btn_venue.grid(row=0, column=3)

entry_day = tk.Entry(frame_add, width=10)
entry_day.grid(row=0, column=4)
btn_day = tk.Button(frame_add, text="Add Day", command=add_day)
btn_day.grid(row=0, column=5)

current_schedule = []

root.mainloop()
