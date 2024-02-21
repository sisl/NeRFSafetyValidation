import csv
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.stats import norm
from validation.utils.blenderUtils import runBlenderOnFailure
import seaborn as sns
import matplotlib.pyplot as plt

import pdb

class CrossEntropyMethod:
    def __init__(self, simulator, q, p, m, m_elite, kmax, blend_file, workspace):
        """
        Initialize the CrossEntropyMethod class.

        Args:
            f: function to maximize. The simulator itself will return the value of the maximization function, so we don't need to pass it in.
            q: proposal distribution
            p: target distribution
            m: number of samples per iteration
            m_elite: number of elite samples per iteration
            kmax: number of iterations
        """
        self.simulator = simulator
        self.q = q # 12 x 12 (12 step #s and 12 noise parameters)
        self.p = p # same as above
        self.m = m # 13?
        self.m_elite = m_elite # 12?
        self.kmax = kmax # 2?
        self.means = [0] * 12
        self.covs = [0] * 12
        self.collisions = 0
        self.stepsToCollision = 0
        self.blend_file = blend_file
        self.workspace = workspace

        self.TOY_PROBLEM = False

    def optimize(self):
        """
        Perform the optimization process.

        Returns:
            mean: mean of the updated proposal distribution
            cov: covariance of the updated proposal distribution
            q: final proposal distribution
            best_solution: best solution found during the optimization process
            best_objective_value: value of the function_to_maximize at the best_solution
        """

        populationScores = []
        eliteScores = []
        zeroedWeight = False # still unsure how this happens, but takes care of when the weights are all 0

        for k in range(self.kmax):
            print(f"Starting population {k}")
            # sample and evaluate function on samples
            population = [] # 10 x 12 x 12 array (one noise for every simulation)
            risks = np.array([])
            outputSimulationList = []

            self.collisions = 0
            self.stepsToCollision = 0

            if self.TOY_PROBLEM:
                # plot the path of each simulation
                plt.figure()
            
            for simulationNumber in range(self.m):
                # ONE SIMULATION BELOW
                self.simulator.reset()
                noises = [self.q[stepNumber].sample() for stepNumber in range(12)] # precompute noise values
                trajectory = [noise.cpu().numpy() for noise in noises]

                if self.TOY_PROBLEM:
                    positions = np.array([[0, 0]], dtype=float)

                riskSteps = np.array([]) # store the score values for each step
                everCollided = False

                for stepNumber in range(12):  
                    outputStepList = [k, simulationNumber, stepNumber] # what will be written to the CSV
                    # noise = self.q[stepNumber].sample()
                    isCollision, collisionVal, currentPos = self.simulator.step(noises[stepNumber])


                    if self.TOY_PROBLEM:
                        # append the current position to positions
                        positions = np.append(positions, [currentPos], axis=0)

                    # append the noises
                    outputStepList.extend(trajectory[stepNumber])
                    # append the sdf value and positions
                    outputStepList.append(collisionVal)
                    outputStepList.extend(currentPos)
                    # output the collision value
                    outputStepList.append(isCollision)

                    # append the value of the step to the simulation data
                    outputSimulationList.append(outputStepList)

                    # store sdf value
                    riskSteps = np.append(riskSteps, collisionVal)

                    # check for collisions
                    if isCollision:
                        self.collisions += 1
                        self.stepsToCollision += stepNumber
                        everCollided = True
                        if not self.TOY_PROBLEM:
                            runBlenderOnFailure(self.blend_file, self.workspace, simulationNumber, stepNumber)
                        break
                
                if self.TOY_PROBLEM:
                    # plot the path of the simulation
                    plt.plot(positions[:, 0], positions[:, 1])                

                # store trajectory (for a simulation)
                population.append(trajectory)
                if self.TOY_PROBLEM:
                    risks = np.append(risks, riskSteps[-1]) # store the distance to the goal at the last step
                else:
                    risks = np.append(risks, min(riskSteps)) # store the smallest sdf value (the closest we get to a crash)
                
                # print the percentage of collisions and the average number of steps to collision, if a collision has occurred
                if everCollided:
                    print(f"Percentage of collisions: {self.collisions / (simulationNumber + 1) * 100}%")
                    print(f"Average number of steps to collision: {self.stepsToCollision / (self.collisions)}")

                if not self.TOY_PROBLEM:
                    # write results to CSV using the format in MonteCarlo.py
                    with open("./results/collisionValues.csv", "a") as csvFile:
                        print(f"Noise List: {trajectory}")
                        writer = csv.writer(csvFile)
                        for outputStepList in outputSimulationList:
                            outputStepList.append(everCollided)
                            writer.writerow(outputStepList) 

            if self.TOY_PROBLEM:
                # plot a star at the start and goal positions
                plt.plot(0, 0, 'ro')
                plt.plot(5, 5, 'go')
                plt.title(f"Simulation Paths for Population {k}")
                plt.savefig(f"./results/pltpaths/simulationPaths_pop{k}.png")
                plt.close()

            print(f"Average score of population {k}: {risks.mean()}")
            populationScores.append(risks.mean())
            # select elite samples and compute weights
            elite_indices = np.argsort(risks)[-self.m_elite:]
            elite_samples = torch.tensor(np.array(population)[elite_indices])
            # print average score of elite samples
            print(f"Average score of elite samples from population {k}: {risks[elite_indices].mean()}")
            eliteScores.append(risks[elite_indices].mean())

            weights = [0] * 12 # each step in each elite sample carries a weight

            # compute the weights
            for i in range(12):
                # compute the weights for the i-th step in each elite sample
                weights[i] = torch.exp(self.p[i].log_prob(elite_samples[:, i]) - self.q[i].log_prob(elite_samples[:, i]))
                print(f"Likelihood of step {i} of elite samples under p: {self.p[i].log_prob(elite_samples[:, i]).mean()}")
                # normalize the weights
                weights[i] = weights[i] / weights[i].sum()
                

                # update proposal distribution based on elite samples
                mean = weights[i] @ elite_samples[:, i] # (1 x 12) @ (12 x len(elite_samples)) = (1 x len(elite_samples))
                cov = torch.zeros(self.q[i].event_shape[0], self.q[i].event_shape[0])
                for j in range(len(elite_samples)):
                    diff = elite_samples[j, i] - mean
                    cov += weights[i][j] * torch.outer(diff, diff)
                cov = cov + 1e-5 * torch.eye(self.q[i].event_shape[0])  # add a small value to the diagonal for numerical stability

                self.means[i] = mean
                self.covs[i] = cov
                try:
                    self.q[i] = MultivariateNormal(mean, cov)
                except ValueError:
                    # may occur if f is improperly specified
                    # pdb.set_trace()
                    print(mean, cov)
                    print(f"Highly improbable weights at step {i} in population {k}! Exiting...")

                    zeroedWeight = True
                    break

                plt.figure()
                for sample in population:
                    sns.histplot(sample[i].numpy(), kde=True, bins=30)
                plt.title(f'Distribution of noise vectors at step {i}')
                plt.xlabel('Noise')
                plt.ylabel('Density')
                plt.savefig(f'./results/pltpaths/noise_distribution_step_{i}.png')
                plt.close()
            
            if zeroedWeight:
                break

            # print the updated proposal distribution
            print(f"Updated Proposal Distribution:")
            for i in range(12):
                print(f"Step {i}: Mean: {self.means[i]}, Covariance: {self.covs[i]}")


        # plot the population and elite scores

        plt.figure()
        plt.plot(populationScores)
        plt.plot(eliteScores)
        plt.legend(["Population", "Elite"])

        # x and y labels
        plt.xlabel("Population #")
        plt.ylabel("Average Score")
        plt.savefig(f"./results/pltpaths/populationScores.png")
        plt.close()



        print("===FINISHED OPTIMIZATION===")
        print("===NOMINAL VALUES===\n")

        # print the nominal values
        for i in range(12):
            print(f"Step {i}: Mean: {self.means[i]}, Covariance: {self.covs[i]}")

        # compute best solution and its objective value
        best_objective_value = 999999
        self.simulator.reset()
        for stepNumber in range(12):
            best_solutionMean = self.means[i]
            best_solutionCov = self.covs[i]

            dist = MultivariateNormal(best_solutionMean, best_solutionCov)

            noise = dist.sample()
            print(f"Step {stepNumber} with noise: {noise}")
            isCollision, collisionVal, currentPos = self.simulator.step(noise)
            best_objective_value = min(best_objective_value, collisionVal)
            print(f"Collision: {isCollision}, Collision Value: {collisionVal}, Current Position: {currentPos}")

            if isCollision: break
        
        return self.means, self.covs, self.q, best_solutionMean, best_solutionCov, best_objective_value
    
