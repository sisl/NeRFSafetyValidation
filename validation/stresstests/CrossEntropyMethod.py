import csv
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.stats import norm
from validation.utils.blenderUtils import runBlenderOnFailure

class CrossEntropyMethod:
    
    collisions = 0
    stepsToCollision = 0

    def __init__(self, simulator, f, q, p, m, m_elite, kmax, blend_file, workspace):
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
        self.m = m # 3?
        self.m_elite = m_elite # 2?
        self.kmax = kmax # 2?
        self.mean = torch.zeros(12)
        self.blend_file = blend_file
        self.workspace = workspace

    def trajectoryLikelihood(self, noise):
        # get the likelihood of a noise measurement by finding each element's probability, logging each, and returning the sum
        likelihoods = norm.pdf(noise, loc = self.noise_mean_cpu, scale = self.noise_std_cpu)
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
                    noise = self.q[i].sample()
                    print(f"Step {stepNumber} with noise: {noise}")
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
                likelihood = self.trajectoryLikelihood(noiseList) # TODO: check vals
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
            elite_samples = np.array(population)[elite_indices]

            weights = np.array((12, len(elite_samples))) # each step in each elite sample carries a weight

            means = torch.zeros((12, self.q[0].event_shape[0]))
            covs = torch.zeros((12, self.q[0].event_shape[0], self.q[0].event_shape[0]))

            # compute the weights
            for i in range(12):
                # compute the weights for the i-th step in each elite sample
                weights = torch.exp(self.p[i].log_prob(elite_samples[:, i]) - self.q[i].log_prob(elite_samples[:, i]))
                
                # normalize the weights
                weights = weights / weights.sum()
                
                # update proposal distribution based on elite samples
                mean = (elite_samples[:, i] * weights).sum()
                cov = torch.zeros(self.q[i].event_shape[0], self.q[i].event_shape[0])
                means[i] = mean
                covs[i] = cov
                for j in range(len(elite_samples)):
                    diff = elite_samples[j, i] - mean
                    cov += weights[j] * torch.ger(diff, diff)
                cov = cov + 1e-1 * torch.eye(self.q[i].event_shape[0])  # Add a small value to the diagonal for numerical stability
                
                self.q[i] = MultivariateNormal(mean, cov)

        # compute best solution and its objective value
        best_solution = means.mean(dim=0) # 12 x 12. One distribution for each step
        best_objective_value = self.f(best_solution)

        return means, covs, self.q, best_solution, best_objective_value
    
