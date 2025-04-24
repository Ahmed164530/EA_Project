import tkinter as tk
from tkinter import ttk, filedialog
import random
import csv
from itertools import combinations

# =========================== Basic Config ===========================
teams = ["Team A", "Team B", "Team C", "Team D"]
venues = ["Stadium 1", "Stadium 2"]
days = ["Day 1", "Day 2", "Day 3", "Day 4"]

# =========================== GA Core ===========================
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
    team_days = {team: set() for team in teams}
    venue_days = {venue: set() for venue in venues}
    day_counts = {day: 0 for day in days}

    for match in individual:
        t1, t2, day, venue = match
        if day in team_days[t1] or day in team_days[t2]:
            score -= 3
        else:
            team_days[t1].add(day)
            team_days[t2].add(day)
        if day in venue_days[venue]:
            score -= 2
        else:
            venue_days[venue].add(day)
        day_counts[day] += 1

    avg_matches = sum(day_counts.values()) / len(days)
    for count in day_counts.values():
        if abs(count - avg_matches) > 1:
            score -= 1

    return score

def crossover(p1, p2):
    point = len(p1) // 2
    return p1[:point] + p2[point:]

def mutate(ind, rate=0.1):
    for i in range(len(ind)):
        if random.random() < rate:
            t1, t2, _, _ = ind[i]
            ind[i] = (t1, t2, random.choice(days), random.choice(venues))
    return ind

def create_population(size, matches):
    return [create_individual(matches) for _ in range(size)]

def select_parents(pop):
    sorted_pop = sorted(pop, key=lambda ind: fitness(ind), reverse=True)
    return sorted_pop[0], sorted_pop[1]

def genetic_algorithm(gens=100, pop_size=20, mutation_rate=0.1):
    matches = generate_matches(teams)
    population = create_population(pop_size, matches)
    for _ in range(gens):
        new_pop = []
        for _ in range(pop_size):
            p1, p2 = select_parents(population)
            child = crossover(p1, p2)
            child = mutate(child, mutation_rate)
            new_pop.append(child)
        population = new_pop
    best = max(population, key=lambda ind: fitness(ind))
    return best, fitness(best)

# =========================== Save Function ===========================
def save_schedule(schedule, filetype):
    filetypes = {
        "CSV": (("CSV Files", "*.csv"), ".csv"),
        "Text": (("Text Files", "*.txt"), ".txt")
    }

    file = filedialog.asksaveasfilename(
        defaultextension=filetypes[filetype][1],
        filetypes=[filetypes[filetype]],
        title="Save Schedule As"
    )

    if not file:
        return

    if filetype == "CSV":
        with open(file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Match", "Day", "Venue"])
            for match in schedule:
                writer.writerow([f"{match[0]} vs {match[1]}", match[2], match[3]])
    elif filetype == "Text":
        with open(file, mode='w') as f:
            for match in schedule:
                f.write(f"{match[0]} vs {match[1]} on {match[2]} at {match[3]}\n")

# =========================== Simulation Function ===========================
def run_simulation():
    configs = [
        {"gens": 50, "pop_size": 10, "mutation": 0.05},
        {"gens": 100, "pop_size": 20, "mutation": 0.1},
        {"gens": 200, "pop_size": 30, "mutation": 0.2}
    ]
    sim_results.delete(*sim_results.get_children())
    for cfg in configs:
        result, score = genetic_algorithm(cfg['gens'], cfg['pop_size'], cfg['mutation'])
        sim_results.insert("", "end", values=(cfg['gens'], cfg['pop_size'], cfg['mutation'], score))

# =========================== Tkinter GUI ===========================
def show_schedule():
    for i in tree.get_children():
        tree.delete(i)
    global current_schedule
    current_schedule, score = genetic_algorithm()
    for match in current_schedule:
        match_text = f"{match[0]} vs {match[1]}"
        tree.insert("", "end", values=(match_text, match[2], match[3]))
    lbl_score.config(text=f"Fitness Score: {score}")

# ========== GUI Layout ==========
root = tk.Tk()
root.title("⚽ Tournament Scheduler via GA")
root.geometry("800x600")

lbl_title = tk.Label(root, text="Sports Tournament Schedule (GA)", font=("Arial", 16, "bold"))
lbl_title.pack(pady=10)

frame_top = tk.Frame(root)
frame_top.pack(pady=5)

btn_generate = tk.Button(frame_top, text="Generate Schedule", font=("Arial", 12), command=show_schedule)
btn_generate.grid(row=0, column=0, padx=5)

lbl_score = tk.Label(frame_top, text="", font=("Arial", 12))
lbl_score.grid(row=0, column=1, padx=5)

combo_format = ttk.Combobox(frame_top, values=["CSV", "Text"], state="readonly")
combo_format.set("CSV")
combo_format.grid(row=0, column=2, padx=5)

btn_save = tk.Button(frame_top, text="Save Schedule", command=lambda: save_schedule(current_schedule, combo_format.get()))
btn_save.grid(row=0, column=3, padx=5)

# Schedule Table
cols = ("Match", "Day", "Venue")
tree = ttk.Treeview(root, columns=cols, show='headings', height=8)
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, width=200, anchor='center')
tree.pack(pady=10)

# Simulation
lbl_sim = tk.Label(root, text="Simulation: GA Configurations vs Fitness", font=("Arial", 12, "bold"))
lbl_sim.pack(pady=5)

sim_results = ttk.Treeview(root, columns=("Generations", "Pop Size", "Mutation Rate", "Score"), show="headings")
for col in ("Generations", "Pop Size", "Mutation Rate", "Score"):
    sim_results.heading(col, text=col)
    sim_results.column(col, width=150, anchor="center")
sim_results.pack(pady=5)

btn_sim = tk.Button(root, text="Run Simulation", command=run_simulation)
btn_sim.pack(pady=10)

current_schedule = []
root.mainloop()
