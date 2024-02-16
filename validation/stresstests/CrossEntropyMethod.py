import csv
import torch
from torch.distributions import MultivariateNormal
import numpy as np
from scipy.stats import norm
from validation.utils.blenderUtils import runBlenderOnFailure

class CrossEntropyMethod:
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
        self.trajs = np.array((12, 10, 12))
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
        all_elite_samples = torch.zeros((self.kmax, self.m_elite, self.q.event_shape[0]))
        all_elite_scores = torch.zeros((self.kmax, self.m_elite))

        for k in range(self.kmax):
            # sample and evaluate function on samples
            # samples = self.q.sample((self.m,))
            noises = [] # 10 x 12 x 12 array (one noise for every simulation)
            for simulationNumber in range(10):
                # ONE SIMULATION BELOW
                self.simulator.reset()
                outputSimulationList = []
                outputStepList = []
                risks = []
                for stepNumber in range(12):  
                    noise = torch.normal(self.mean, self.q[i])
                    print(f"Step {stepNumber} with noise: {noise}")
                    isCollision, collisionVal, currentPos = self.simulator.step(noise)

                    # append the noises
                    noises.append(noise)
                    noiseList = noise.cpu().numpy()
                    outputStepList.extend(noiseList)
                    
                    # append the sdf value and positions
                    outputStepList.append(collisionVal)
                    outputStepList.extend(currentPos)

                # TODO: store trajectory (for a simulation)
                self.trajs[simulationNumber] = outputStepList

                # calculate likelihood of the trajectory
                likelihood = self.trajectoryLikelihood(noiseList)
                outputStepList.append(likelihood)
                
                # output the collision value
                outputStepList.append(isCollision)

                # append the value of the step to the simulation data
                outputSimulationList.append(outputStepList)

                # check for collisions
                if isCollision:
                    self.collisions += 1
                    self.stepsToCollision += stepNumber
                    everCollided = True
                    runBlenderOnFailure(self.blend_file, self.workspace, simulationNumber, stepNumber)
                    break

                # TODO: store the "risks" (take the min of the sdf values and store it in f)
                # risks = torch.tensor([self.f(min(outputStepList)) for n in noises])#f(samples)
                risks.append(self.f(min(outputStepList)))

                # write results to CSV using the format in MonteCarlo.py
                with open("./results/collisionValues.csv", "a") as csvFile:
                    print(f"Noise List: {noiseList}")
                    writer = csv.writer(csvFile)
                    for outputStepList in outputSimulationList:
                        outputStepList.append(everCollided)
                        writer.writerow(outputStepList) 

            # select elite samples and compute weights
            elite_samples = noises[risks.topk(min(self.m, self.m_elite)).indices]
            
            # for loop start
            for i in range(12):
                weights = torch.exp(self.p[i].log_prob(elite_samples.squeeze()) -  self.q[i].log_prob(elite_samples.squeeze()))
                weights = weights / weights.sum()

                # update proposal distribution based on elite samples
                mean = (elite_samples * weights.unsqueeze(1)).sum(dim=0)
                cov = torch.cov(elite_samples.T, aweights=weights)
                cov = cov + 1e-1*torch.eye(self.q[i].event_shape[0])
                self.q[i] = MultivariateNormal(mean, cov)

                # save elite samples and their scores
                all_elite_samples[k] = elite_samples
                all_elite_scores[k] = risks.topk(min(self.m, self.m_elite)).values

        # TODO: compute best solution and its objective value
        best_solution = mean # 12 x 12. One distribution for each step
        best_objective_value = self.f(best_solution)

        return mean, cov, self.q, best_solution, best_objective_value
    
