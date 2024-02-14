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
        self.f = f
        self.q = q
        self.p = p
        self.m = m
        self.m_elite = m_elite
        self.kmax = kmax

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
            samples = self.q.sample((self.m,))
            scores = torch.tensor([self.f(s) for s in samples])

            # select elite samples and compute weights
            elite_samples = samples[scores.topk(min(self.m, self.m_elite)).indices]
            weights = torch.exp(self.p.log_prob(elite_samples.squeeze()) -  self.q.log_prob(elite_samples.squeeze()))
            weights = weights / weights.sum()

            # update proposal distribution based on elite samples
            mean = (elite_samples * weights.unsqueeze(1)).sum(dim=0)
            cov = torch.cov(elite_samples.T, aweights=weights)
            cov = cov + 1e-1*torch.eye(self.q.event_shape[0])
            self.q = MultivariateNormal(mean, cov)

            # save elite samples and their scores
            all_elite_samples[k] = elite_samples
            all_elite_scores[k] = scores.topk(min(self.m, self.m_elite)).values

        # compute best solution and its objective value
        best_solution = mean
        best_objective_value = self.f(best_solution)

        return mean, cov, self.q, best_solution, best_objective_value
    