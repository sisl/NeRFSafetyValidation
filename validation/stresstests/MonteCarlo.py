from tqdm import trange
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MonteCarlo(object):


    def __init__(self, simulator, n_simulations, steps, noise_mean, noise_std):
        self.simulator = simulator
        self.n_simulations = n_simulations
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.steps = steps

    def validate(self):
        for i in range(self.n_simulations):
            self.simulator.reset()
            print(f"Starting simulation {i}")
            for j in trange(self.steps):
                noise = torch.normal(self.noise_mean, self.noise_std)
                print(f"Step {j} with noise: {noise}")
                self.simulator.step(noise)
            

