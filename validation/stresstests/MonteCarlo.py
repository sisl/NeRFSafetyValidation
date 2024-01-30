from tqdm import trange
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MonteCarlo(object):

    collisions = dict() # key: noise values, val: boolean (collided or not)

    def __init__(self, simulator, n_simulations, steps, noise_mean, noise_std, collision_grid):
        self.simulator = simulator
        self.n_simulations = n_simulations
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        # self.steps = steps
        self.steps = 1
        self.collision_grid = collision_grid

    # take in a state vector and output the grid coordinate
    # taken from nav/quad_plot.py
    def stateToGridCoord(self, state):
        grid_size = 100//5 # side//kernel_size
        state_float = grid_size*(state[:3] + 1) / 2
        state_coord = tuple(int(state_float[i]) for i in range(3))
        return state_coord

    def validate(self):
        for i in range(self.n_simulations):
            self.simulator.reset()
            print(f"Starting simulation {i}")
            for j in trange(self.steps):
                noise = torch.normal(self.noise_mean, self.noise_std)
                print(f"Step {j} with noise: {noise}")
                self.simulator.step(noise)
                
                current_state = self.simulator.current_state
                current_state_gridCoord = self.stateToGridCoord(current_state)
                collided = self.collision_grid[current_state_gridCoord]
                if collided:
                    print(f"Drone collided in state {current_state}")
                else:
                    print(f"Drone did NOT collide in state {current_state}")

                self.collisions[noise] = collided
            

