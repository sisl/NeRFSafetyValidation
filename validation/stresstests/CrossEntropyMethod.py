import csv
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.stats import norm
from validation.utils.blenderUtils import runBlenderOnFailure

import pdb

class CrossEntropyMethod:
    def __init__(self, simulator, f, q, p, m, m_elite, kmax, noise_mean, noise_std, blend_file, workspace):
        """
        Initialize the CrossEntropyMethod class.

        Args:
            f: function to maximize
            q: proposal distribution
            p: target distribution
            m: number of samples per iteration
            m_elite: number of elite samples per iteration
            kmax: number of iterations
        """
        self.simulator = simulator
        self.f = f # 10 x 1 (one score for each simulation)
        self.q = q # 12 x 12 (12 step #s and 12 noise parameters)
        self.p = p # same as above
        self.m = m # 13?
        self.m_elite = m_elite # 12?
        self.kmax = kmax # 2?
        self.means = [0] * 12
        self.covs = [0] * 12
        self.collisions = 0
        self.stepsToCollision = 0
        self.noise_mean = noise_mean.cpu().numpy()
        self.noise_std = noise_std.cpu().numpy()
        self.blend_file = blend_file
        self.workspace = workspace

    def trajectoryLikelihood(self, noise):
        # get the likelihood of a noise measurement by finding each element's probability, logging each, and returning the sum
        likelihoods = norm.pdf(noise, loc = self.noise_mean, scale = self.noise_std)
        logLikelihoods = np.log(likelihoods)
        return logLikelihoods.sum()

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
        for k in range(self.kmax):
            print(f"Starting population {k}")
            # sample and evaluate function on samples
            # samples = self.q.sample((self.m,))
            population = [] # 10 x 12 x 12 array (one noise for every simulation)
            risks = np.array([])
            outputSimulationList = []

            for simulationNumber in range(10):
                # ONE SIMULATION BELOW
                self.simulator.reset()
                trajectory = []
                step = []
                riskSteps = []
                outputStepList = []
                everCollided = False

                for stepNumber in range(12):  
                    noise = self.q[stepNumber].sample()
                    isCollision, collisionVal, currentPos = self.simulator.step(noise)

                    # append the noises
                    noiseList = noise.cpu().numpy()
                    outputStepList.extend(noiseList)
                    step = noiseList

                    # append the sdf value and positions
                    outputStepList.append(collisionVal)
                    outputStepList.extend(currentPos)

                    # store sdf value
                    riskSteps.append(collisionVal)
                    trajectory.append(step) # store noise in trajectory

                    # check for collisions
                    if isCollision:
                        self.collisions += 1
                        self.stepsToCollision += stepNumber
                        everCollided = True
                        runBlenderOnFailure(self.blend_file, self.workspace, simulationNumber, stepNumber)
                        break

                # store trajectory (for a simulation)
                population.append(trajectory)
                risks = np.append(risks, min(riskSteps)) # store the smallest sdf value

                # calculate likelihood of the trajectory
                likelihood = self.trajectoryLikelihood(noiseList) # TODO: check this
                outputStepList.append(likelihood)
                
                # output the collision value
                outputStepList.append(isCollision)

                # append the value of the step to the simulation data
                outputSimulationList.append(outputStepList)

                # write results to CSV using the format in MonteCarlo.py
                with open("./results/collisionValues.csv", "a") as csvFile:
                    print(f"Noise List: {noiseList}")
                    writer = csv.writer(csvFile)
                    for outputStepList in outputSimulationList:
                        outputStepList.append(everCollided)
                        writer.writerow(outputStepList) 

            # select elite samples and compute weights
            elite_indices = np.argsort(risks)[-self.m_elite:]
            elite_samples = torch.tensor(np.array(population)[elite_indices])

            # print the elite samples and their indices
            # print(f"Elite Samples: {elite_samples}")
            # print(f"Elite Indices: {elite_indices}")
            # print(f"Risks: {risks}")

            weights = [0] * 12 # each step in each elite sample carries a weight

            # compute the weights
            for i in range(12):
                # compute the weights for the i-th step in each elite sample
                weights[i] = torch.exp(self.p[i].log_prob(elite_samples[:, i]) - self.q[i].log_prob(elite_samples[:, i]))
                
                # normalize the weights
                weights[i] = weights[i] / weights[i].sum()
                # pdb.set_trace()

                # update proposal distribution based on elite samples
                # mean = (elite_samples[:, i] * weights[i]).sum()
                mean = weights[i] @ elite_samples[:, i] # (1 x 12) @ (12 x len(elite_samples)) = (1 x len(elite_samples)
                cov = torch.zeros(self.q[i].event_shape[0], self.q[i].event_shape[0])
                for j in range(len(elite_samples)):
                    diff = elite_samples[j, i] - mean
                    cov += weights[i][j] * torch.outer(diff, diff)
                cov = cov + 1e-1 * torch.eye(self.q[i].event_shape[0])  # add a small value to the diagonal for numerical stability
                # pdb.set_trace()

                self.means[i] = mean
                self.covs[i] = cov
                self.q[i] = MultivariateNormal(mean, cov)

            # print the updated proposal distribution
            print(f"Updated Proposal Distribution:")
            for i in range(12):
                print(f"Step {i}: Mean: {self.means[i]}, Covariance: {self.covs[i]}")


        print("===FINISHED OPTIMIZATION===")
        print("===NOMINAL VALUES===\n")

        # print the nominal values
        for i in range(12):
            print(f"Step {i}: Mean: {self.means[i]}, Covariance: {self.covs[i]}")

        # compute best solution and its objective value
        # best_solution = self.means.mean(dim=0) # 12 x 12. One distribution for each step
        # best_objective_value = self.f(best_solution)
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
        
        # print(self.means, self.covs, self.q, best_solution, best_objective_value)
        return self.means, self.covs, self.q, best_solutionMean, best_solutionCov, best_objective_value
    
