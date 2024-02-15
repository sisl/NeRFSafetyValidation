import torch
from torch.distributions import MultivariateNormal

class CrossEntropyMethod:
    def __init__(self, f, q, p, m, m_elite, kmax):
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
        self.f = f # 10 x 1 (one score for each simulation)
        self.q = q # 12 x 12 (12 step #s and 12 noise parameters)
        self.p = p # same as above
        self.m = m # 3?
        self.m_elite = m_elite # 2?
        self.kmax = kmax # 2?
        self.mean = torch.zeros(12)
        self.trajs = np.array((12, 10, 12))

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
                for i in range(12):  
                    noise = torch.normal(self.mean, self.q[i])
                    
                    # do a simulation
                    # step()


                # store trajectory (for a simulation)

                # calculate likelihood of the trajectory

                # store the "risks" (take the min of the sdf values and store it in f)

                # write results to CSV using the format in MonteCarlo.py


            # scores = torch.tensor([self.f(s) for s in samples])

            # select elite samples and compute weights
            elite_samples = samples[scores.topk(min(self.m, self.m_elite)).indices]
            
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
                all_elite_scores[k] = scores.topk(min(self.m, self.m_elite)).values


            # for loop end

        # compute best solution and its objective value
        best_solution = mean # 12 x 12. One distribution for each step
        best_objective_value = self.f(best_solution)

        return mean, cov, self.q, best_solution, best_objective_value
    
