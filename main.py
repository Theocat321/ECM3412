"""
Small genetic algorithm for a toy bin-packing task.

Two variants...
- BPP1: 500 items, 10 bins, weights i
- BPP2: 500 items, 50 bins, weights i^2 / 2

Chromosome encodes bin assignments...
fitness = 100 / (1 + (max_bin_sum - min_bin_sum)).

Usage:
  python main.py
  python main.py --master-seed 123 --csv ga_out.csv
"""

from __future__ import annotations
import argparse
import csv
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class BPP:
    name: str
    k_items: int
    b_bins: int
    weights: List[float]


def make_bpp1() -> BPP:
    """BPP1: 500 items, 10 bins, weights i."""
    k = 500
    b = 10
    weights = [float(i) for i in range(1, k + 1)]
    return BPP("BPP1", k, b, weights)


def make_bpp2() -> BPP:
    """BPP2: 500 items, 50 bins, weights i^2 / 2."""
    k = 500
    b = 50
    weights = [ (i * i) / 2.0 for i in range(1, k + 1) ]
    return BPP("BPP2", k, b, weights)


@dataclass
class GAParams:
    population_size: int  # p
    pm: float             # mutation rate per gene
    tournament_size: int  # t
    pc: float = 0.8       # crossover rate (fixed)
    elitism: int = 1      # number of elites (fixed)
    max_evals: int = 10_000  # termination by fitness evaluations


@dataclass
class TrialResult:
    problem: str
    population_size: int
    pm: float
    tournament_size: int
    pc: float
    elitism: int
    max_evals: int
    trial_index: int
    seed: int
    best_fitness: float
    best_d: float
    generations: int
    evals_used: int


class BinPackingGA:
    """Minimal GA for balancing bin sums via assignments 1..b."""
    def __init__(self, problem: BPP, params: GAParams, rng: random.Random):
        self.problem = problem
        self.params = params
        self.rng = rng
        self.k = problem.k_items
        self.b = problem.b_bins
        self.weights = problem.weights

        # Evaluation counter
        self.evals_used = 0

        self.history: Dict[str, List[float]] = {"gen": [], "best": [], "mean": []}

    def random_chromosome(self) -> List[int]:
        """Return a random chromosome with genes in [1..b]."""
        return [self.rng.randint(1, self.b) for _ in range(self.k)]

    def _bins_sums(self, chrom: List[int]) -> Tuple[float, float]:
        """Return (min_sum, max_sum) of bin totals for a chromosome."""
        sums = [0.0] * self.b
        for i, bin_id in enumerate(chrom):
            sums[bin_id - 1] += self.weights[i]
        return (min(sums), max(sums))

    def fitness(self, chrom: List[int]) -> float:
        """Compute fitness = 100 / (1 + d), with d = max_bin_sum - min_bin_sum."""
        min_sum, max_sum = self._bins_sums(chrom)
        d = max_sum - min_sum
        fit = 100.0 / (1.0 + d)
        self.evals_used += 1
        return fit

    def tournament_select(self, population: List[List[int]], fitnesses: List[float]) -> List[int]:
        t = self.params.tournament_size
        best_idx = None
        for _ in range(t):
            i = self.rng.randrange(len(population))
            if best_idx is None or fitnesses[i] > fitnesses[best_idx]:
                best_idx = i
        return population[best_idx][:]

    def uniform_crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        c1 = p1[:]
        c2 = p2[:]
        if self.rng.random() < self.params.pc:
            for i in range(self.k):
                if self.rng.random() < 0.5:
                    c1[i], c2[i] = c2[i], c1[i]
        return c1, c2

    def mutate(self, chrom: List[int]) -> None:
        pm = self.params.pm
        for i in range(self.k):
            if self.rng.random() < pm:
                # Randomly reassign to any valid bin (1..b)
                chrom[i] = self.rng.randint(1, self.b)

    def run(self) -> Tuple[List[int], float, float, int, int]:
        """Run one trial. Returns (best, best_fitness, best_d, generations, evals_used)."""
        p = self.params.population_size
        max_evals = self.params.max_evals
        elite_n = self.params.elitism

        # Initialize population
        population = [self.random_chromosome() for _ in range(p)]
        fitnesses = [self.fitness(ind) for ind in population]

        # Track best
        best_idx = max(range(p), key=lambda i: fitnesses[i])
        best = population[best_idx][:]
        best_fit = fitnesses[best_idx]
        best_min, best_max = self._bins_sums(best)
        best_d = best_max - best_min

        self.history["gen"].append(0)
        self.history["best"].append(max(fitnesses))
        self.history["mean"].append(sum(fitnesses) / len(fitnesses))

        generations = 0

        # Stop if we already exhausted evals in initialization
        while self.evals_used < max_evals:
            generations += 1

            # Pick top elite_n
            elite_indices = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)[:elite_n]
            elites = [population[i][:] for i in elite_indices]
            elite_fits = [fitnesses[i] for i in elite_indices]

            # Create next population
            next_pop: List[List[int]] = []
            next_fit: List[float] = []

            # Ensure space left for elites at the end
            while len(next_pop) < p - elite_n and self.evals_used < max_evals:
                # Parents
                parent1 = self.tournament_select(population, fitnesses)
                parent2 = self.tournament_select(population, fitnesses)

                # Crossover
                child1, child2 = self.uniform_crossover(parent1, parent2)

                # Mutation
                self.mutate(child1)
                self.mutate(child2)

                # Evaluate children (respect eval budget)
                f1 = self.fitness(child1)
                if len(next_pop) < p - elite_n:
                    next_pop.append(child1)
                    next_fit.append(f1)

                if self.evals_used >= max_evals:
                    break

                f2 = self.fitness(child2)
                if len(next_pop) < p - elite_n:
                    next_pop.append(child2)
                    next_fit.append(f2)

            # If eval budget prevents filling, pad with elites (no new evals)
            while len(next_pop) < p - elite_n:
                # duplicate the best elite to keep size
                next_pop.append(elites[0][:])
                next_fit.append(elite_fits[0])

            # Add elites at the end (carry fitness values)
            population = next_pop + elites
            fitnesses = next_fit + elite_fits

            # Update best
            gen_best_idx = max(range(p), key=lambda i: fitnesses[i])
            if fitnesses[gen_best_idx] > best_fit:
                best = population[gen_best_idx][:]
                best_fit = fitnesses[gen_best_idx]
                mn, mx = self._bins_sums(best)  # not counted as a 'fitness' since we don't call fitness()
                best_d = mx - mn

            # Record per-generation stats
            self.history["gen"].append(generations)
            self.history["best"].append(max(fitnesses))
            self.history["mean"].append(sum(fitnesses) / len(fitnesses))

        return best, best_fit, best_d, generations, self.evals_used


def derived_seed(master_seed: int, problem_id: int, setting_id: int, trial_index: int) -> int:
    """Deterministic seed derivation across problems/settings/trials."""
    return master_seed + 10_000 * problem_id + 100 * setting_id + trial_index


def plot_history(history: Dict[str, List[float]], title: str, out_path: Path, show: bool = False) -> None:
    gens = history.get("gen", [])
    best = history.get("best", [])
    mean = history.get("mean", [])
    if not gens:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(gens, best, label="best fitness", color="tab:blue")
    plt.plot(gens, mean, label="mean fitness", color="tab:orange", alpha=0.8)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out_path))
    if show:
        plt.show()
    plt.close()


def run_experiments(master_seed: int, csv_path: str, *, plot: bool = False, plot_dir: str = "plots", show: bool = False) -> List[TrialResult]:
    problems = [make_bpp1(), make_bpp2()]  # problem_id is index (0, 1)

    # Settings (setting_id is index 0..3)
    settings: List[GAParams] = [
        GAParams(population_size=100, pm=0.01, tournament_size=3, pc=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, pm=0.05, tournament_size=3, pc=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, pm=0.01, tournament_size=7, pc=0.8, elitism=1, max_evals=10_000),
        GAParams(population_size=100, pm=0.05, tournament_size=7, pc=0.8, elitism=1, max_evals=10_000),
    ]

    n_trials = 5
    results: List[TrialResult] = []

    best_per_config: Dict[Tuple[int, int], Tuple[float, Dict[str, List[float]], str, Path]] = {}

    for prob_id, problem in enumerate(problems):
        for set_id, params in enumerate(settings):
            for trial in range(n_trials):
                seed = derived_seed(master_seed, prob_id, set_id, trial)
                rng = random.Random(seed)
                ga = BinPackingGA(problem, params, rng)
                best, best_fit, best_d, generations, evals_used = ga.run()

                results.append(TrialResult(
                    problem=problem.name,
                    population_size=params.population_size,
                    pm=params.pm,
                    tournament_size=params.tournament_size,
                    pc=params.pc,
                    elitism=params.elitism,
                    max_evals=params.max_evals,
                    trial_index=trial,
                    seed=seed,
                    best_fitness=best_fit,
                    best_d=best_d,
                    generations=generations,
                    evals_used=evals_used,
                ))

                # Per-trial log (comment out if too noisy)
                print(f"[{problem.name}] setting={set_id} trial={trial} seed={seed} "
                      f"best_fitness={best_fit:.8f} best_d={best_d:.4f} "
                      f"gens={generations} evals={evals_used}")

                # Track best trial for this configuration
                if plot:
                    key = (prob_id, set_id)
                    prev = best_per_config.get(key)
                    if (prev is None) or (best_fit > prev[0]):
                        fname = f"{problem.name.lower()}_set{set_id}_best.png"
                        out_path = Path(plot_dir) / fname
                        title = f"{problem.name}, setting {set_id} — best trial (seed {seed}, trial {trial})"
                        best_per_config[key] = (best_fit, ga.history.copy(), title, out_path)

    # Write CSV
    fieldnames = list(asdict(results[0]).keys()) if results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Plot only the best trial per configuration if requested
    if plot:
        for (_prob_id, _set_id), (_fit, hist, title, out_path) in best_per_config.items():
            plot_history(hist, title, out_path, show=show)

    return results


def summarize(results: List[TrialResult]) -> None:
    if not results:
        print("No results to summarize.")
        return

    # Group by (problem, pm, tournament_size)
    from collections import defaultdict
    groups: Dict[Tuple[str, float, int], List[TrialResult]] = defaultdict(list)
    for r in results:
        key = (r.problem, r.pm, r.tournament_size)
        groups[key].append(r)

    # Print table header
    print("\n=== Summary (mean ± std over 5 trials) ===")
    print("Problem, pm, tournament, mean_best_fitness, std_best_fitness, mean_best_d, std_best_d")
    for key, rs in sorted(groups.items()):
        bf = [x.best_fitness for x in rs]
        bd = [x.best_d for x in rs]
        mean_bf = sum(bf) / len(bf)
        mean_bd = sum(bd) / len(bd)
        std_bf = (sum((x - mean_bf) ** 2 for x in bf) / (len(bf) - 1)) ** 0.5 if len(bf) > 1 else 0.0
        std_bd = (sum((x - mean_bd) ** 2 for x in bd) / (len(bd) - 1)) ** 0.5 if len(bd) > 1 else 0.0
        print(f"{key[0]}, {key[1]:.3f}, {key[2]}, {mean_bf:.8f}, {std_bf:.8f}, {mean_bd:.4f}, {std_bd:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Bin Packing (BPP1 & BPP2)")
    parser.add_argument("--master-seed", type=int, default=42, help="Master seed (default: 42)")
    parser.add_argument("--csv", type=str, default="ga_bpp_results.csv", help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Plot fitness curve for the best trial per config")
    parser.add_argument("--plot-dir", type=str, default="plots", help="Directory to save plots (default: plots)")
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively (if supported)")
    args = parser.parse_args()

    results = run_experiments(
        master_seed=args.master_seed,
        csv_path=args.csv,
        plot=args.plot,
        plot_dir=args.plot_dir,
        show=args.show_plots,
    )
    summarize(results)

    print(f"\nAll trial results written to: {args.csv}")


if __name__ == "__main__":
    main()
