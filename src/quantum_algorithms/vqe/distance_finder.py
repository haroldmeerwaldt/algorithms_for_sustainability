import numpy as np

from src.quantum_algorithms.vqe.energy_calculation import GroundStateEnergyCalculation, AtomInfo
from src.quantum_algorithms.vqe.plotting import setup_plot, plot_point

box_size = 3


def minimize_ground_state_energy():
    ax = setup_plot()
    plot_point((0, 0, 0), ax, symbol="*", size=100)
    calc = GroundStateEnergyCalculation()
    for _ in range(2):
        coords = np.random.default_rng().uniform(-box_size, box_size, 3)
        # coords = (0, 0, 0.1)
        atom = AtomInfo("H", tuple(coords))
        calc.add_atom(atom)
        ground_state_energy = calc()
        print("\n", coords, ground_state_energy)
        ax.scatter(*coords, marker="o", c="b")
        learning_rate = 0.1
        epsilon = 0.01
        update_directions = [sign * learning_rate * np.array([int(n==0), int(n==1), int(n==2)]) for n in range(3) for sign in [1, -1]]
        ground_state_energy_j = np.inf
        last_direction_updated = -1
        while True:
            coords_j = coords.copy()
            for k, update_direction in enumerate(update_directions):
                ground_state_energy_k, coords = try_single_step(calc, coords, update_direction, ground_state_energy, ax)
                if ground_state_energy_k == ground_state_energy:
                    if last_direction_updated==k:
                        break
                else:
                    last_direction_updated = k
                    ground_state_energy = ground_state_energy_k

            # acceleration step
            ground_state_energy_l, coords = line_search(calc, coords, coords - coords_j, ground_state_energy, ax)
            if ground_state_energy_l < ground_state_energy:
                ground_state_energy = ground_state_energy_l

            if abs(ground_state_energy_j - ground_state_energy) < epsilon:
                break
            else:
                ground_state_energy_j = ground_state_energy

        print(f"done at position {coords} at distance {np.linalg.norm(coords)} with ground state energy {ground_state_energy}")

    plt.show()
    return ground_state_energy


def try_single_step(calc, coords, update_direction, ground_state_energy, ax):
    coords_i = coords + update_direction
    calc.update_last_atoms_coords(coords_i)
    ground_state_energy_i = calc()
    print(update_direction, ground_state_energy_i - ground_state_energy, end="")
    if ground_state_energy_i < ground_state_energy:
        ground_state_energy = ground_state_energy_i
        coords = coords_i
        print(coords, ground_state_energy)
        plot_point(coords, ax)
    else:
        print()
    return ground_state_energy, coords


def line_search(calc, coords, update_direction, ground_state_energy, ax):
    while True:
        ground_state_energy_i, coords = try_single_step(calc, coords, update_direction, ground_state_energy, ax)
        if ground_state_energy_i < ground_state_energy:
            ground_state_energy = ground_state_energy_i
        else:
            break
    return ground_state_energy, coords
