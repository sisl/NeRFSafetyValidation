import csv
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.stats import norm
from validation.utils.blenderUtils import runBlenderOnFailure

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

        self.TOY_PROBLEM = True

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

        populationScores = []
        eliteScores = []
        zeroedWeight = False # still unsure how this happens, but takes care of when the weights are all 0

        for k in range(self.kmax):
            print(f"Starting population {k}")
            # sample and evaluate function on samples
            population = [] # 10 x 12 x 12 array (one noise for every simulation)
            risks = np.array([])
            outputSimulationList = []


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

                step = []
                riskSteps = np.array([])
                outputStepList = []
                everCollided = False

                for stepNumber in range(12):  
                    # noise = self.q[stepNumber].sample()
                    isCollision, collisionVal, currentPos = self.simulator.step(noises[stepNumber])


                    if self.TOY_PROBLEM:
                        # append the current position to positions
                        positions = np.append(positions, [currentPos], axis=0)

                    # append the noises
                    # noiseList = noise.cpu().numpy()
                    outputStepList.extend(trajectory[stepNumber])
                    # step = noiseList

                    # append the sdf value and positions
                    outputStepList.append(collisionVal)
                    outputStepList.extend(currentPos)

                    # store sdf value
                    riskSteps = np.append(riskSteps, collisionVal)
                    # trajectory.append(step) # store noise in trajectory

                    # check for collisions
                    if isCollision:
                        self.collisions += 1
                        self.stepsToCollision += stepNumber
                        everCollided = True
                        # runBlenderOnFailure(self.blend_file, self.workspace, simulationNumber, stepNumber)
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
                
                

                # output the collision value
                outputStepList.append(isCollision)

                # append the value of the step to the simulation data
                outputSimulationList.append(outputStepList)

                # write results to CSV using the format in MonteCarlo.py
                # with open("./results/collisionValues.csv", "a") as csvFile:
                #     print(f"Noise List: {noiseList}")
                #     writer = csv.writer(csvFile)
                #     for outputStepList in outputSimulationList:
                #         outputStepList.append(everCollided)
                #         writer.writerow(outputStepList) 

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
                

                # update proposal distribution based on elite samples
                # mean = (elite_samples[:, i] * weights[i]).sum()
                mean = weights[i] @ elite_samples[:, i] # (1 x 12) @ (12 x len(elite_samples)) = (1 x len(elite_samples)
                cov = torch.zeros(self.q[i].event_shape[0], self.q[i].event_shape[0])
                for j in range(len(elite_samples)):
                    diff = elite_samples[j, i] - mean
                    cov += weights[i][j] * torch.outer(diff, diff)
                cov = cov + 1e-1 * torch.eye(self.q[i].event_shape[0])  # add a small value to the diagonal for numerical stability

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
            
            if zeroedWeight:
                break

            # print the updated proposal distribution
            # print(f"Updated Proposal Distribution:")
            # for i in range(12):
            #     print(f"Step {i}: Mean: {self.means[i]}, Covariance: {self.covs[i]}")


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
    
