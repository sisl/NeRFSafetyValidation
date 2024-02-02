from tqdm import trange
import torch
import csv

from validation.utils.blenderUtils import runBlenderOnFailure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MonteCarlo(object):

    collisions = 0

    def __init__(self, simulator, n_simulations, steps, noise_mean, noise_std, collision_grid, blend_file, workspace):
        self.simulator = simulator
        self.n_simulations = n_simulations
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        # self.steps = steps
        self.steps = 2
        self.collision_grid = collision_grid
        self.blend_file = blend_file
        self.workspace = workspace

    def validate(self):
        for i in range(self.n_simulations):
            self.simulator.reset()
            print(f"Starting simulation {i}")
            for j in trange(self.steps):
                
                noise = torch.normal(self.noise_mean, self.noise_std)
                print(f"Step {j} with noise: {noise}")
                isCollision, collisionVal = self.simulator.step(noise, self.collision_grid)
                noiseList = [i, j]
                noiseList.extend(noise.cpu().numpy().tolist())
                noiseList.append(collisionVal)
                
                with open("./results/collisionValues.csv", "a") as csvFile:
                    print(f"Noise List: {noiseList}")
                    writer = csv.writer(csvFile)
                    writer.writerow(noiseList) 
     
                if isCollision:
                    self.collisions += 1
                    runBlenderOnFailure(self.blend_file, self.workspace, i, j)
                    break
        print(f"\n\t{self.collisions} collisions in {self.n_simulations} simulations, for a crash % of {100 * self.collisions/self.n_simulations}%\n")

