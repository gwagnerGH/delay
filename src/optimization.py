"""Optimization codes for CAP6.

Theo Moers
tlm2160@columbia.edu
Columbia University

based originally on Adam Bauer

This code contains three optimization classes: GeneticAlgorithm,
GradientSearch, and CoordinateDescent. The current version of CAP6 uses
the first two in tandem.
"""

import multiprocessing
import os
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
try:
    import copyreg
except:
    import copy_reg as copyreg

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from src.tools import _pickle_method, _unpickle_method

copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class ObjectiveWithGradient(object):
    """Model-space maximization objective with an exact/structured gradient."""

    def utility(self, m):
        raise NotImplementedError

    def gradient(self, m):
        raise NotImplementedError

    def value_and_gradient(self, m):
        m = np.asarray(m, dtype=float)
        return self.utility(m), self.gradient(m)


class UtilityGradientAdapter(ObjectiveWithGradient):
    """Adapter for utility-like objects that already expose exact gradients."""

    def __init__(self, utility):
        self.model = utility
        if not (
            hasattr(utility, "gradient")
            or hasattr(utility, "value_and_gradient")
        ):
            raise NotImplementedError(
                "adjoint_lbfgsb requires a utility/objective with gradient(m) "
                "or value_and_gradient(m). The current EZUtility stack does "
                "not expose an exact adjoint yet, so this mode fails fast "
                "instead of silently using finite differences."
            )

    def utility(self, m):
        if hasattr(self.model, "utility"):
            return float(np.asarray(self.model.utility(np.asarray(m, dtype=float))).reshape(-1)[0])
        if hasattr(self.model, "value"):
            return float(self.model.value(np.asarray(m, dtype=float)))
        raise NotImplementedError("Objective does not expose utility(m) or value(m).")

    def gradient(self, m):
        if hasattr(self.model, "gradient"):
            return np.asarray(self.model.gradient(np.asarray(m, dtype=float)), dtype=float)
        _, grad = self.value_and_gradient(m)
        return grad

    def value_and_gradient(self, m):
        m = np.asarray(m, dtype=float)
        if hasattr(self.model, "value_and_gradient"):
            value, grad = self.model.value_and_gradient(m)
            return float(np.asarray(value).reshape(-1)[0]), np.asarray(grad, dtype=float)
        return self.utility(m), self.gradient(m)


class ScipyObjective(object):
    """SciPy minimization wrapper for active scaled optimizer variables."""

    def __init__(self, objective, lower_bounds, upper_bounds):
        self.objective = objective
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        self.active_mask = (self.upper_bounds - self.lower_bounds) > 1e-14
        self.active_lower = self.lower_bounds[self.active_mask]
        self.active_upper = self.upper_bounds[self.active_mask]
        self.active_width = self.active_upper - self.active_lower
        self.nfev = 0
        self.ngev = 0

    def full_from_scaled(self, x_active):
        full = self.lower_bounds.copy()
        if len(self.active_width):
            x_active = np.clip(np.asarray(x_active, dtype=float), 0.0, 1.0)
            full[self.active_mask] = self.active_lower + x_active * self.active_width
        return full

    def fun(self, x_active):
        self.nfev += 1
        utility = self.objective.utility(self.full_from_scaled(x_active))
        if not np.isfinite(utility):
            return 1e100
        return -float(utility)

    def jac(self, x_active):
        self.ngev += 1
        full = self.full_from_scaled(x_active)
        grad_m = np.asarray(self.objective.gradient(full), dtype=float)
        if grad_m.shape[0] != self.lower_bounds.shape[0]:
            raise ValueError(
                "Gradient length {} does not match mitigation length {}".format(
                    grad_m.shape[0], self.lower_bounds.shape[0]
                )
            )
        grad_x = -grad_m[self.active_mask] * self.active_width
        if not np.all(np.isfinite(grad_x)):
            raise FloatingPointError("Non-finite adjoint gradient")
        return grad_x

    def fun_and_jac(self, x_active):
        self.nfev += 1
        self.ngev += 1
        full = self.full_from_scaled(x_active)
        utility, grad_m = self.objective.value_and_gradient(full)
        grad_m = np.asarray(grad_m, dtype=float)
        if grad_m.shape[0] != self.lower_bounds.shape[0]:
            raise ValueError(
                "Gradient length {} does not match mitigation length {}".format(
                    grad_m.shape[0], self.lower_bounds.shape[0]
                )
            )
        fun = -float(utility) if np.isfinite(utility) else 1e100
        grad_x = -grad_m[self.active_mask] * self.active_width
        if not np.all(np.isfinite(grad_x)):
            raise FloatingPointError("Non-finite adjoint gradient")
        return fun, grad_x


def objective_with_gradient_from_utility(utility):
    if isinstance(utility, ObjectiveWithGradient):
        return utility
    return UtilityGradientAdapter(utility)


def prolong_policy_nearest_ancestor(source_policy, source_tree, target_tree):
    """Map a lower-resolution tree policy onto a target tree deterministically."""

    source_policy = np.asarray(source_policy, dtype=float)
    if len(source_policy) != source_tree.num_decision_nodes:
        raise ValueError("source_policy length does not match source_tree")
    mapped = np.zeros(target_tree.num_decision_nodes, dtype=float)
    source_years = np.asarray(source_tree.decision_times, dtype=float)
    for target_node in range(target_tree.num_decision_nodes):
        target_period = target_tree.get_period(target_node)
        target_year = float(target_tree.decision_times[target_period])
        source_period = int(np.searchsorted(source_years, target_year, side="right") - 1)
        source_period = min(max(source_period, 0), source_tree.num_periods - 1)
        target_state = target_tree.get_state(target_node, target_period)
        source_nodes = source_tree.get_num_nodes_period(source_period)
        target_nodes = target_tree.get_num_nodes_period(target_period)
        source_state = int(np.floor((target_state + 0.5) * source_nodes / target_nodes))
        source_state = min(max(source_state, 0), source_nodes - 1)
        mapped[target_node] = source_policy[source_tree.get_node(source_period, source_state)]
    return mapped


class GeneticAlgorithm(object):
    """Optimization algorithm for the CAP6 model.

    The genetic algorithm is a stochastic optimization algorithm which
    searches for optima using ideas inspired from Darwin's evoluationary
    theory. The fitness of a given indivudal is assessed, and we assume that
    the best fitness implies the solution to the problem.

    Parameters
    ----------
    pop_amount : int
        number of individuals in the population
    num_feature : int
        number of elements in each individual, i.e. number of nodes in
        tree-model
    num_generations : int
        number of generations of the populations to be evaluated
    bound : float
        upper bound of mitigation in each node
    cx_prob : float
         probability of mating
    mut_prob : float
        probability of mutation.
    utility : `Utility` object
        object of utility class
    fixed_values : ndarray, optional
        nodes to keep fixed
    fixed_indices : ndarray, optional
        indices of nodes to keep fixed
    print_progress : bool, optional
        if the progress of the evolution should be printed

    Attributes
    ----------
    pop_amount : int
        number of individuals in the population
    num_feature : int
        number of elements in each individual, i.e. number of nodes in
        tree-model
    num_generations : int
        number of generations of the populations to be evaluated
    bound : float
        upper bound of mitigation in each node
    cx_prob : float
         probability of mating
    mut_prob : float
        probability of mutation.
    u : `Utility` object
        object of utility class
    fixed_values : ndarray, optional
        nodes to keep fixed
    fixed_indices : ndarray, optional
        indices of nodes to keep fixed
    print_progress : bool, optional
        if the progress of the evolution should be printed
    u_hist: (num_generations,) array
        history of best value of objective function
    """

    def __init__(self, pop_amount, num_generations, cx_prob, mut_prob, bound,
                 num_feature, utility, fixed_values=None,
                 fixed_indices=None, print_progress=False,
                 upper_bounds=None):
        self.num_feature = num_feature
        self.pop_amount = pop_amount
        self.num_gen = num_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.u = utility
        self.bound = bound
        self.fixed_values = fixed_values
        self.fixed_indices = fixed_indices
        self.print_progress = print_progress
        self.upper_bounds = upper_bounds
        self.u_hist = None
        self.map_timeout_seconds = max(1, int(os.environ.get("GA_MAP_TIMEOUT_SECONDS", "3600")))
        self.max_tasks_per_child = max(1, int(os.environ.get("GA_MAX_TASKS_PER_CHILD", "2000")))

    def _worker_count(self):
        for name in ("GA_WORKERS", "N_GA_WORKERS", "NSLOTS"):
            raw = os.environ.get(name, "").strip()
            if not raw:
                continue
            try:
                return max(1, int(raw))
            except ValueError:
                pass
        return max(1, multiprocessing.cpu_count())

    def _apply_bounds(self, values, lower=0.0):
        """Project mitigation vector(s) into the feasible box constraints."""
        bounded = np.clip(values, lower, self.bound)
        if self.upper_bounds is not None:
            bounded = np.minimum(bounded, self.upper_bounds)
        if self.fixed_values is not None:
            fixed_values = np.asarray(self.fixed_values).flatten()
            if bounded.ndim == 1:
                bounded[self.fixed_indices] = fixed_values
            else:
                bounded[:, self.fixed_indices] = fixed_values
        return bounded

    def run(self):
        """Start the evolution process.

        The evolution steps are:
            1. Select the individuals to perform cross-over and mutation.
            2. Cross over among the selected candidate.
            3. Mutate result as offspring.
            4. Combine the result of offspring and parent together. And
            selected the top 80 percent of original population amount.
            5. Random Generate 20 percent of original population amount new
            individuals and combine the above new population.

        Returns
        -------
        tuple
            final population and the fitness for the final population

        Note
        ----
        Uses the :mod:`~multiprocessing` package.

        """
        print("----------------Genetic Evolution Starting----------------")
        # generate initial population
        pop = self._generate_population(self.pop_amount)
        worker_count = self._worker_count()
        print("GA workers: {}".format(worker_count))
        pool = multiprocessing.Pool(
            processes=worker_count, maxtasksperchild=self.max_tasks_per_child
        )

        def evaluate_population(values, stage):
            result = pool.map_async(self._evaluate, values)
            try:
                return result.get(timeout=self.map_timeout_seconds)
            except multiprocessing.TimeoutError:
                pool.terminate()
                pool.join()
                raise RuntimeError(
                    "GA population evaluation timed out after {} seconds during {}. "
                    "Set GA_MAP_TIMEOUT_SECONDS to adjust this watchdog.".format(
                        self.map_timeout_seconds, stage
                    )
                )

        # make first set of fitnesses
        fitness = evaluate_population(pop, "initial population")
        fitness = np.array([val[0] for val in fitness])
        print("The fitness is ", fitness)
        self.u_hist = np.zeros(self.num_gen)

        # go through generations
        for g in range(0, self.num_gen):
            print ("-- Generation {} --".format(g+1))

            # select which members of the population get passed on
            pop_select = self._select(np.copy(pop), rate=1)

            # cross over genes
            self._uniform_cross_over(pop_select, 0.50)

            # randomly mutate some number of members
            self._uniform_mutation(pop_select, 0.25,
                                   np.exp(-float(g)/self.num_gen)**2)
            #self._mutate(pop_select, 0.05)

            # evaluate fitness of the selected population
            fitness_select = evaluate_population(pop_select, "generation {} selected population".format(g + 1))
            fitness_select = np.array([val[0] for val in fitness_select])

            # append selected population and fitness for passing on to existing
            # population and fitness
            pop_tmp = np.append(pop, pop_select, axis=0)
            fitness_tmp = np.append(fitness, fitness_select, axis=0)

            # select surviving population and fitness
            pop_survive, fitness_survive = self._survive(pop_tmp, fitness_tmp)

            # make new population to enter into surviving population
            pop_new = self._generate_population(self.pop_amount\
                                                - len(pop_survive))

            # evaluate fitness of new population
            fitness_new = evaluate_population(pop_new, "generation {} new population".format(g + 1))
            fitness_new = np.array([val[0] for val in fitness_new])

            # merge surviving population with new population
            pop = np.append(pop_survive, pop_new, axis=0)
            fitness = np.append(fitness_survive, fitness_new, axis=0)
            if self.print_progress:
                self._show_evolution(fitness, pop)
            self.u_hist[g] = fitness[0]

        # return final fitness and popoulation values
        fitness = evaluate_population(pop, "final population")
        fitness = np.array([val[0] for val in fitness])
        pool.close()
        pool.join()
        return pop, fitness

    def _generate_population(self, size):
        """Generate an initial population.

        Return 1D-array of random values in the given bound as the initial
        population.

        Parameters
        ----------
        size: int
            size of population

        Returns
        -------
        pop: nd array
            array of population values
        """

        pop = np.random.random([size, self.num_feature])*self.bound
        return self._apply_bounds(pop)

    def _evaluate(self, individual):
        """Returns the utility of given individual.

        Parameters
        ----------
        individual: nd array
            individual whose utility we want to calculate

        Returns
        -------
        utility: nd array
            array of utility values
        """

        value = self.u.utility(self._apply_bounds(np.asarray(individual).copy()))
        value = np.asarray(value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(value)):
            return np.asarray([-np.inf])
        return value

    def _select(self, pop, rate):
        """Returns a 1D-array of selected individuals.

        Parameters
        ----------
        pop : ndarray
            population given by 2D-array with shape ('pop_amount',
            'num_feature')
        rate : float
            the probability of an individual being selected

        Returns
        -------
        ndarray
            selected individuals
        """

        index = np.random.choice(self.pop_amount, int(rate*self.pop_amount),
                                 replace=False)
        return pop[index,:]

    def _uniform_cross_over(self, pop, ind_prob):
        """Performs a uniform cross-over of the population.

        Parameters
        ----------
        pop : ndarray
            population given by 2D-array with shape ('pop_amount',
            'num_feature')
        ind_prob : float
            probability of feature cross-over
        """

        child_group1 = pop[::2]
        child_group2 = pop[1::2]
        for child1, child2 in zip(child_group1, child_group2):
            size = min(len(child1), len(child2))
            for i in range(size):
                if np.random.random() < ind_prob:
                    child1[i], child2[i] = child2[i], child1[i]

    def _uniform_mutation(self, pop, ind_prob, scale=2.0):
        """Mutates individual's elements. The individual has a probability of
        `mut_prob` of being selected and every element in this individual has a
        probability `ind_prob` of being mutated. The mutated value is the
        current value plus a scaled uniform [-0.5,0.5] random value.

        Parameters
        ----------
        pop : ndarray
            population given by 2D-array with shape ('pop_amount',
            'num_feature')
        ind_prob : float
            probability of feature mutation
        scale : float
            scaling constant of the random generated number for mutation
        """

        pop_len = len(pop)
        mutate_index = np.random.choice(pop_len, int(self.mut_prob * pop_len),
                                        replace=False)
        for i in mutate_index:
            prob = np.random.random(self.num_feature)
            inc = (np.random.random(self.num_feature) - 0.5)*scale
            pop[i] += (prob > (1.0-ind_prob)).astype(int)*inc
            pop[i] = self._apply_bounds(pop[i], lower=1e-5)

    def _mutate(self, pop, ind_prob, scale=2.0):
        """Mutates individual's elements. The individual has a probability of
        `mut_prob` of being selected and every element in this individual has a
        probability `ind_prob` of being mutated. The mutated value is a random
        number.

        Parameters
        ----------
        pop : ndarray
            population given by 2D-array with shape ('pop_amount',
            'num_feature')
        ind_prob : float
            probability of feature mutation
        scale : float
            scaling constant of the random generated number for mutation

        """
        pop_tmp = np.copy(pop)
        mutate_index = np.random.choice(self.pop_amount,
                                        int(self.mut_prob * self.pop_amount),
                                        replace=False)
        for i in mutate_index:
            feature_index = np.random.choice(self.num_feature,
                                             int(ind_prob * self.num_feature),
                                             replace=False)
            for j in feature_index:
                if self.fixed_indices is not None and j in self.fixed_indices:
                    continue
                else:
                    pop[i][j] = max(0.0,
                                    pop[i][j]+(np.random.random()-0.5)*scale)
                    pop[i] = self._apply_bounds(pop[i])

    def _survive(self, pop_tmp, fitness_tmp):
        """The 80 percent of the individuals with best fitness survives to
        the next generation.

        Parameters
        ----------
        pop_tmp : ndarray
            population
        fitness_tmp : ndarray
            fitness values of `pop_temp`

        Returns
        -------
        ndarray
            individuals that survived
        """

        index_fits  = np.argsort(fitness_tmp)[::-1]
        fitness = fitness_tmp[index_fits]
        pop = pop_tmp[index_fits]
        num_survive = int(0.8*self.pop_amount)
        survive_pop = np.copy(pop[:num_survive])
        survive_fitness = np.copy(fitness[:num_survive])
        return np.copy(survive_pop), np.copy(survive_fitness)

    def _random_index(self, individuals, size):
        """Generate a random index of individuals of size 'size'.

        Parameters
        ----------
        individuals : ndarray or list
            2D-array of individuals
        size : int
            number of indices to generate

        Returns
        -------
        ndarray
            1D-array of indices

        """

        inds_size = len(individuals)
        return np.random.choice(inds_size, size)

    def _selection_tournament(self, pop, k, tournsize, fitness):
        """Select `k` individuals from the input `individuals` using `k`
        tournaments of `tournsize` individuals.

        Parameters
        ----------
        individuals : ndarray or list
            2D-array of individuals to select from
        k : int
             number of individuals to select
        tournsize : int
            number of individuals participating in each tournament

        Returns
        -------
        ndarray s
            selected individuals
        """

        chosen = []
        for i in range(k):
            index = self._random_index(pop, tournsize)
            aspirants = pop[index]
            aspirants_fitness = fitness[index]
            chosen_index = np.where(aspirants_fitness\
                                    == np.max(aspirants_fitness))[0]
            if len(chosen_index) != 0:
                chosen_index = chosen_index[0]
            chosen.append(aspirants[chosen_index])
        return np.array(chosen)

    def _two_point_cross_over(self, pop):
        """Performs a two-point cross-over of the population.

        Parameters
        ----------
        pop : ndarray
            population given by 2D-array with shape ('pop_amount', 'num_feature')
        """

        child_group1 = pop[::2]
        child_group2 = pop[1::2]
        for child1, child2 in zip(child_group1, child_group2):
            if np.random.random() <= self.cx_prob:
                cxpoint1 = np.random.randint(1, self.num_feature)
                cxpoint2 = np.random.randint(1, self.num_feature - 1)
                if cxpoint2 >= cxpoint1:
                    cxpoint2 += 1
                else: # Swap the two cx points
                    cxpoint1, cxpoint2 = cxpoint2, cxpoint1
                child1[cxpoint1:cxpoint2], child2[cxpoint1:cxpoint2] \
                = child2[cxpoint1:cxpoint2].copy(), child1[cxpoint1:cxpoint2].copy()
                if self.fixed_values is not None:
                    child1[self.fixed_indices] = self.fixed_values
                    child2[self.fixed_indices] = self.fixed_values

    def _show_evolution(self, fits, pop):
        """Print statistics of the evolution of the population.

        Parameters
        ----------
        fits: nd array
            fitness values

        pop: nd array
            population values
        """

        length = len(pop)
        mean = fits.mean()
        std = fits.std()
        min_val = fits.min()
        max_val = fits.max()
        print(" Min {} \n Max {} \n Avg {}".format(min_val, max_val, mean))
        print(" Std {} \n Population Size {}".format(std, length))
        print(" Best Individual: ", pop[np.argmax(fits)])

    def _survive(self, pop_tmp, fitness_tmp):
        """The 80 percent of the individuals with best fitness survives to
        the next generation.

        Parameters
        ----------
        pop_tmp : ndarray
            population
        fitness_tmp : ndarray
            fitness values of `pop_temp`

        Returns
        -------
        ndarray
            individuals that survived
        """

        index_fits  = np.argsort(fitness_tmp)[::-1]
        fitness = fitness_tmp[index_fits]
        pop = pop_tmp[index_fits]
        num_survive = int(0.8*self.pop_amount)
        survive_pop = np.copy(pop[:num_survive])
        survive_fitness = np.copy(fitness[:num_survive])
        return np.copy(survive_pop), np.copy(survive_fitness)

class GradientSearch(object):
    """Gradient search optimization algorithm for the CAP6 model.

    Parameters
    ----------
    utility : `Utility` object
        object of utility class
    learning_rate : float
        starting learning rate of gradient descent
    var_nums : int
        number of elements in array to optimize
    accuracy : float
        stop value for the gradient descent
    iterations : int
        maximum number of iterations
    fixed_values : ndarray, optional
        nodes to keep fixed
    fixed_indices : ndarray, optional
        indices of nodes to keep fixed
    print_progress : bool, optional
        if the progress of the evolution should be printed
    scale_alpha : ndarray, optional
        array to scale the learning rate

    Attributes
    ----------
    utility : `Utility` object
        object of utility class
    learning_rate : float
        starting learning rate of gradient descent
    var_nums : int
        number of elements in array to optimize
    accuracy : float
        stop value for the gradient descent
    iterations : int
        maximum number of iterations
    fixed_values : ndarray, optional
        nodes to keep fixed
    fixed_indices : ndarray, optional
        indices of nodes to keep fixed
    print_progress : bool, optional
        if the progress of the evolution should be printed
    scale_alpha : ndarray, optional
        array to scale the learning rate
    """

    def __init__(self, utility, var_nums, accuracy=1e-06, iterations=100,
                 fixed_values=None, fixed_indices=None, print_progress=False,
                 scale_alpha=None, upper_bound=1.5, upper_bounds=None):
        self.u = utility
        self.var_nums = var_nums
        self.accuracy = accuracy
        self.iterations = iterations
        self.fixed_values  = fixed_values
        self.fixed_indices = fixed_indices
        self.print_progress = print_progress
        self.scale_alpha = scale_alpha
        self.upper_bound = upper_bound
        self.upper_bounds = upper_bounds
        if scale_alpha is None:
            self.scale_alpha = np.exp(np.linspace(0.0, 3.0, var_nums))

    def _apply_bounds(self, values):
        """Project mitigation vector into global and per-node box constraints."""
        bounded = np.asarray(values).copy()
        if self.upper_bound is not None:
            bounded = np.clip(bounded, 0.0, self.upper_bound)
        else:
            bounded = np.maximum(bounded, 0.0)
        if self.upper_bounds is not None:
            bounded = np.minimum(bounded, self.upper_bounds)
        if self.fixed_values is not None:
            fixed_values = np.asarray(self.fixed_values).flatten()
            bounded[self.fixed_indices] = fixed_values
        return bounded

    @staticmethod
    def _scalar_utility(utility):
        """Return the first utility value as a Python scalar."""
        return float(np.asarray(utility).reshape(-1)[0])

    def _partial_grad(self, i):
        """Calculate the ith element of the gradient vector.

        Parameters
        ----------
        i: int
            index of array to calculate gradient for
        """
        m_copy = self.m.copy()
        m_copy[i] = m_copy[i] - self.delta if (m_copy[i] - self.delta)>=0 else 0.0
        m_copy = self._apply_bounds(m_copy)
        minus_utility = self.u.utility(m_copy)
        m_copy[i] += 2*self.delta
        m_copy = self._apply_bounds(m_copy)
        plus_utility = self.u.utility(m_copy)
        grad = self._scalar_utility(plus_utility-minus_utility) / (2*self.delta)
        return grad, i

    def numerical_gradient(self, m, delta=1e-08, fixed_indices=None):
        """Calculate utility gradient numerically.

        Parameters
        ----------
        m : ndarray or list
            array of mitigation
        delta : float, optional
            change in mitigation
        fixed_indices : ndarray or list, optional
            indices of gradient that should not be calculated

        Returns
        -------
        ndarray
            gradient

        """
        self.delta = delta
        self.m = m
        if fixed_indices is None:
            fixed_indices = []
        grad = np.zeros(len(m))
        if not isinstance(m, np.ndarray):
            self.m = np.array(m)
        pool = multiprocessing.Pool()
        indices = np.delete(list(range(len(m))), fixed_indices)
        res = pool.map(self._partial_grad, indices)
        for g, i in res:
            grad[i] = g
        pool.close()
        pool.join()
        del self.m
        del self.delta
        return grad

    def _accelerate_scale(self, accelerator, prev_grad, grad):
        """Accelerate scale.

        Parameters
        ----------
        accelerator: nd array
            array of accelerations?

        prev_grad: nd array
            gradient proir to current point?

        grad: nd array
            current point gradient?

        Returns
        -------
        accelerator: nd array
            accelerator?
        """
        sign_vector = np.sign(prev_grad * grad)
        scale_vector = np.ones(self.var_nums) * ( 1 + 0.10)
        accelerator[sign_vector <= 0] = 1
        accelerator *= scale_vector
        return accelerator

    def gradient_descent(self, initial_point, return_last=False):
        """Gradient descent algorithm. The `initial_point` is updated using the
        Adam algorithm. Adam uses the history of the gradient to compute individual
        step sizes for each element in the mitigation vector. The vector of step
        sizes are calculated using estimates of the first and second moments of
        the gradient.

        Parameters
        ----------
        initial_point : ndarray
            initial guess of the mitigation
        return_last : bool, optional
            if True the function returns the last point, else the point
                with highest utility

        Returns
        -------
        tuple
            (best point, best utility)
        """

        initial_point = self._apply_bounds(initial_point)

        num_decision_nodes = initial_point.shape[0]
        x_hist = np.zeros((self.iterations+1, num_decision_nodes))
        u_hist = np.zeros(self.iterations+1)
        u_hist[0] = self._scalar_utility(self.u.utility(initial_point))
        x_hist[0] = initial_point

        beta1, beta2 = 0.90, 0.999
        eta = 0.0015
        eps = 1e-5
        m_t, v_t = 0, 0

        prev_grad = 0.0
        accelerator = np.ones(self.var_nums)

        for i in range(1, self.iterations):
            grad = self.numerical_gradient(x_hist[i-1],
                                           fixed_indices=self.fixed_indices)
            m_t = beta1*m_t + (1-beta1)*grad
            v_t = beta2*v_t + (1-beta2)*np.power(grad, 2)
            m_hat = m_t / (1-beta1**(i))
            v_hat = v_t / (1-beta2**(i))
            if i != 0:
                accelerator = self._accelerate_scale(accelerator, prev_grad,
                                                     grad)

            new_x = x_hist[i-1] + ((eta*m_hat)/(v_hat**(0.5)+eps))\
                    * accelerator
            new_x = self._apply_bounds(new_x)

            x_hist[i] = new_x
            u_hist[i] = self._scalar_utility(self.u.utility(new_x))
            prev_grad = grad.copy()

            if self.print_progress:
                print("-- Iteration {} -- \n Current Utility: {}".format(i+1,\
                                                                         u_hist[i]))
                print(new_x)

        if return_last:
            return x_hist[i+1], u_hist[i+1]

        best_index = np.argmax(u_hist)
        return x_hist[best_index], u_hist[best_index]

    def run(self, initial_point_list, topk=4):
        """Initiate the gradient search algorithm.

        Parameters
        ----------
        initial_point_list : list
            list of initial points to select from
        topk : int, optional
            select and run gradient descent on the `topk` first points of
            `initial_point_list`

            Adam: The first four points in CAP6 are the four "most fit"
            members from the Genetic Algorithm

        Returns
        -------
        tuple
            best mitigation point and the utility of the best mitigation point

        Raises
        ------
        ValueError
            If `topk` is larger than the length of `initial_point_list`.

        Note
        ----
        Uses the :mod:`~multiprocessing` package.

        """
        print("----------------Gradient Search Starting----------------")

        if topk > len(initial_point_list):
            raise ValueError("topk {} > number of initial points\
                             {}".format(topk, len(initial_point_list)))

        candidate_points = initial_point_list[:topk]
        mitigations = []
        utilities = np.zeros(topk)
        for cp, count in zip(candidate_points, list(range(topk))):
            if not isinstance(cp, np.ndarray):
                cp = np.array(cp)
            print("Starting process {} of Gradient Descent".format(count+1))
            m, u  = self.gradient_descent(cp)
            mitigations.append(m)
            utilities[count] = u
        best_index = np.argmax(utilities)
        return mitigations[best_index], utilities[best_index]


class CandidateScreenedLBFGSB(object):
    """Deterministic candidate-screened multi-start L-BFGS-B optimizer."""

    def __init__(self, utility, lower_bounds, upper_bounds, warm_starts=None,
                 objective_with_gradient=None, gradient_mode="finite_difference",
                 optimizer_name=None,
                 n_candidates=256, n_local_starts=8, max_candidates=1024,
                 max_local_starts=16, maxiter=150, ftol=1e-7, gtol=1e-5,
                 utility_spread_tol=1e-7, utility_spread_rel_tol=1e-3,
                 escalate_on_dispersion=True,
                 start_design="sobol", seed=20250706, print_progress=False,
                 scenario_name="lbfgsb", candidate_progress_every=25,
                 callback_progress_every=10, n_workers=1,
                 screening_workers=None, gradient_workers=None,
                 local_start_workers=1, finite_diff_step=1e-8,
                 warm_start_perturbations=16,
                 warm_start_perturbation_scale=0.05,
                 structured_start_count=64,
                 near_full_mitigation=0.98,
                 start_boundary_epsilon=1e-6,
                 preserve_diverse_starts=True,
                 min_diverse_start_groups=4,
                 perturbation_check=True,
                 perturbation_step=0.01,
                 perturbation_tol=1e-7,
                 perturbation_rel_tol=1e-6,
                 perturbation_block_count=8,
                 local_start_max_utility_gap=np.inf,
                 local_start_max_relative_utility_gap=0.25,
                 min_local_starts_after_filter=2,
                 gradient_progress_every=1,
                 kkt_check=True, projected_gradient_tol=None,
                 validate_gradient=False,
                 gradient_validation_epsilons=None,
                 gradient_validation_directions=4,
                 gradient_validation_abs_tol=1e-5,
                 gradient_validation_rel_tol=1e-3,
                 gradient_validation_seed=None,
                 nonsmooth_kkt_check=True,
                 nonsmooth_kkt_step=1e-4,
                 nonsmooth_kkt_utility_gain_tol=1e-10,
                 nonsmooth_kkt_max_coordinates=32,
                 mandatory_starts=None):
        self.utility = utility
        self.objective_with_gradient = objective_with_gradient
        self.gradient_mode = str(gradient_mode).lower()
        self.optimizer_name = optimizer_name or (
            "adjoint_lbfgsb" if self.gradient_mode == "adjoint" else "lbfgsb_multistart"
        )
        if self.gradient_mode not in ("finite_difference", "adjoint"):
            raise ValueError("gradient_mode must be 'finite_difference' or 'adjoint'")
        self.lower_bounds = np.asarray(lower_bounds, dtype=float)
        self.upper_bounds = np.asarray(upper_bounds, dtype=float)
        if self.lower_bounds.shape != self.upper_bounds.shape:
            raise ValueError("lower_bounds and upper_bounds must have the same shape")
        if np.any(self.upper_bounds < self.lower_bounds):
            raise ValueError("upper_bounds must be >= lower_bounds")
        self.warm_starts = list([] if warm_starts is None else warm_starts)
        self.mandatory_starts = []
        mandatory_start_values = (
            [] if mandatory_starts is None else mandatory_starts
        )
        for index, start in enumerate(mandatory_start_values):
            start = np.asarray(start, dtype=float)
            if start.shape != self.lower_bounds.shape:
                raise ValueError(
                    "mandatory start {} has shape {}; expected {}".format(
                        index, start.shape, self.lower_bounds.shape
                    )
                )
            if not np.all(np.isfinite(start)):
                raise ValueError("mandatory start {} must be finite".format(index))
            if np.any(start < self.lower_bounds) or np.any(start > self.upper_bounds):
                raise ValueError(
                    "mandatory start {} must lie within the optimizer bounds".format(
                        index
                    )
                )
            self.mandatory_starts.append(start.copy())
        self.n_candidates = int(n_candidates)
        self.n_local_starts = int(n_local_starts)
        self.max_candidates = int(max_candidates)
        self.max_local_starts = int(max_local_starts)
        self.maxiter = int(maxiter)
        self.ftol = float(ftol)
        self.gtol = float(gtol)
        self.utility_spread_tol = float(utility_spread_tol)
        self.utility_spread_rel_tol = float(utility_spread_rel_tol)
        self.escalate_on_dispersion = bool(escalate_on_dispersion)
        self.start_design = str(start_design).lower()
        self.seed = int(seed) % (2**32 - 1)
        self.print_progress = print_progress
        self.scenario_name = scenario_name
        self.candidate_progress_every = max(0, int(candidate_progress_every))
        self.callback_progress_every = max(0, int(callback_progress_every))
        self.n_workers = max(1, int(n_workers))
        self.screening_workers = max(
            1, int(screening_workers if screening_workers is not None else self.n_workers)
        )
        self.gradient_workers = max(
            1, int(gradient_workers if gradient_workers is not None else self.n_workers)
        )
        self.local_start_workers = max(1, int(local_start_workers))
        self.finite_diff_step = float(finite_diff_step)
        self.warm_start_perturbations = max(0, int(warm_start_perturbations))
        self.warm_start_perturbation_scale = float(warm_start_perturbation_scale)
        self.structured_start_count = max(0, int(structured_start_count))
        self.near_full_mitigation = float(near_full_mitigation)
        self.start_boundary_epsilon = max(0.0, float(start_boundary_epsilon))
        self.preserve_diverse_starts = bool(preserve_diverse_starts)
        self.min_diverse_start_groups = max(0, int(min_diverse_start_groups))
        self.perturbation_check = bool(perturbation_check)
        self.perturbation_step = max(0.0, float(perturbation_step))
        self.perturbation_tol = max(0.0, float(perturbation_tol))
        self.perturbation_rel_tol = max(0.0, float(perturbation_rel_tol))
        self.perturbation_block_count = max(0, int(perturbation_block_count))
        self.local_start_max_utility_gap = float(local_start_max_utility_gap)
        self.local_start_max_relative_utility_gap = float(
            local_start_max_relative_utility_gap
        )
        self.min_local_starts_after_filter = max(
            1, int(min_local_starts_after_filter)
        )
        self.gradient_progress_every = max(0, int(gradient_progress_every))
        self.kkt_check = bool(kkt_check)
        self.projected_gradient_tol = (
            float(projected_gradient_tol)
            if projected_gradient_tol is not None else self.gtol
        )
        self.nonsmooth_kkt_check = bool(nonsmooth_kkt_check)
        self.nonsmooth_kkt_step = max(0.0, float(nonsmooth_kkt_step))
        self.nonsmooth_kkt_utility_gain_tol = max(
            0.0, float(nonsmooth_kkt_utility_gain_tol)
        )
        self.nonsmooth_kkt_max_coordinates = max(
            1, int(nonsmooth_kkt_max_coordinates)
        )
        self.validate_gradient = bool(validate_gradient)
        self.gradient_validation_epsilons = tuple(
            float(eps) for eps in (
                gradient_validation_epsilons
                if gradient_validation_epsilons is not None
                else (1e-3, 1e-4, 1e-5, 1e-6)
            )
        )
        self.gradient_validation_directions = max(
            1, int(gradient_validation_directions)
        )
        self.gradient_validation_abs_tol = float(gradient_validation_abs_tol)
        self.gradient_validation_rel_tol = float(gradient_validation_rel_tol)
        self.gradient_validation_seed = (
            int(gradient_validation_seed)
            if gradient_validation_seed is not None else self.seed + 2654435761
        ) % (2**32 - 1)
        self.active_mask = (self.upper_bounds - self.lower_bounds) > 1e-14
        self.active_indices = np.where(self.active_mask)[0]
        self.fixed_indices = np.where(~self.active_mask)[0]
        self.active_lower = self.lower_bounds[self.active_mask]
        self.active_upper = self.upper_bounds[self.active_mask]
        self.active_width = self.active_upper - self.active_lower
        self.scipy_objective = None
        if self.gradient_mode == "adjoint":
            if self.objective_with_gradient is None:
                self.objective_with_gradient = objective_with_gradient_from_utility(
                    utility
                )
            self.scipy_objective = ScipyObjective(
                self.objective_with_gradient,
                self.lower_bounds,
                self.upper_bounds,
            )
        self._gradient_validation_diag = {
            "gradient_validation_status": (
                "skipped" if not self.validate_gradient else "not_run"
            )
        }
        self._eval_count = 0

    @staticmethod
    def _parallel_map(func, values, workers):
        values = list(values)
        if int(workers) <= 1 or len(values) <= 1:
            return [func(value) for value in values]
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            return list(executor.map(func, values))

    @staticmethod
    def _scalar_utility(value):
        return float(np.asarray(value).reshape(-1)[0])

    def _full_from_scaled(self, x):
        full = self.lower_bounds.copy()
        if len(self.active_indices):
            x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
            full[self.active_mask] = self.active_lower + x * self.active_width
        return full

    def _scaled_from_full(self, m):
        m = np.asarray(m, dtype=float)
        if m.shape[0] != self.lower_bounds.shape[0]:
            return None
        projected = np.minimum(np.maximum(m, self.lower_bounds), self.upper_bounds)
        if not len(self.active_indices):
            return np.zeros(0)
        return (projected[self.active_mask] - self.active_lower) / self.active_width

    def _scaled_from_active_mitigation(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape[0] != len(self.active_indices):
            return None
        full = self.lower_bounds.copy()
        full[self.active_mask] = values
        return self._scaled_from_full(full)

    def _clip_start_scaled(self, x):
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        if not len(x):
            return x
        eps = min(self.start_boundary_epsilon, 0.5)
        if eps <= 0.0:
            return x
        return np.clip(x, eps, 1.0 - eps)

    def _utility_at_full(self, m):
        if self.objective_with_gradient is not None:
            try:
                value = self.objective_with_gradient.utility(np.asarray(m, dtype=float))
            except Exception:
                return -np.inf
            return value if np.isfinite(value) else -np.inf
        try:
            value = self._scalar_utility(self.utility.utility(np.asarray(m, dtype=float)))
        except Exception:
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    def _objective_scaled(self, x):
        self._eval_count += 1
        utility = self._utility_at_full(self._full_from_scaled(x))
        if not np.isfinite(utility):
            return 1e100
        return -utility

    def _objective_value_scaled(self, x):
        if self.scipy_objective is not None:
            return self.scipy_objective.fun(x)
        utility = self._utility_at_full(self._full_from_scaled(x))
        if not np.isfinite(utility):
            return 1e100
        return -utility

    def _finite_difference_gradient(self, x, objective, f0=None,
                                    workers=None):
        x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
        if f0 is None:
            f0 = objective(x)
        dim = len(x)
        if dim == 0:
            return np.zeros(0)
        step_default = max(self.finite_diff_step, np.sqrt(np.finfo(float).eps))
        jobs = []
        for index in range(dim):
            forward_step = min(step_default, 1.0 - x[index])
            if forward_step > 0.0:
                x_step = x.copy()
                x_step[index] += forward_step
                jobs.append((index, forward_step, 1.0, x_step))
            else:
                backward_step = min(step_default, x[index])
                if backward_step <= 0.0:
                    jobs.append((index, 1.0, 0.0, x.copy()))
                else:
                    x_step = x.copy()
                    x_step[index] -= backward_step
                    jobs.append((index, backward_step, -1.0, x_step))

        def evaluate(job):
            index, step, direction, x_step = job
            if direction == 0.0:
                return index, 0.0
            f_step = objective(x_step)
            if direction > 0.0:
                grad = (f_step - f0) / step
            else:
                grad = (f0 - f_step) / step
            return index, grad

        results = self._parallel_map(
            evaluate,
            jobs,
            max(1, int(workers if workers is not None else self.gradient_workers)),
        )
        grad = np.zeros(dim)
        for index, value in results:
            grad[index] = value
        return grad

    @staticmethod
    def _projected_gradient(x, grad, bound_tol=1e-8):
        if len(grad) == 0:
            return np.zeros(0)
        x = np.asarray(x, dtype=float)
        projected = np.asarray(grad, dtype=float).copy()
        at_lower = x <= bound_tol
        at_upper = x >= 1.0 - bound_tol
        projected[at_lower] = np.minimum(projected[at_lower], 0.0)
        projected[at_upper] = np.maximum(projected[at_upper], 0.0)
        return projected

    @classmethod
    def _projected_gradient_inf_norm(cls, x, grad, bound_tol=1e-8):
        projected = cls._projected_gradient(x, grad, bound_tol=bound_tol)
        if len(projected) == 0:
            return 0.0
        return float(np.max(np.abs(projected)))

    def _kkt_location(self, x, grad):
        projected = self._projected_gradient(x, grad)
        if len(projected) == 0:
            return {
                "projected_grad_inf_norm": 0.0,
                "projected_grad_l2_norm": 0.0,
                "worst_kkt_node": -1,
                "worst_kkt_time": -1,
                "worst_kkt_state": -1,
            }
        worst_active = int(np.argmax(np.abs(projected)))
        worst_node = int(self.active_indices[worst_active])
        worst_time = -1
        worst_state = -1
        tree = getattr(self.utility, "tree", None)
        if tree is not None:
            try:
                worst_period = int(tree.get_period(worst_node))
                worst_time = int(tree.decision_times[worst_period])
                worst_state = int(tree.get_state(worst_node, worst_period))
            except Exception:
                worst_time = -1
                worst_state = -1
        return {
            "projected_grad_inf_norm": float(np.max(np.abs(projected))),
            "projected_grad_l2_norm": float(np.linalg.norm(projected)),
            "worst_kkt_node": worst_node,
            "worst_kkt_time": worst_time,
            "worst_kkt_state": worst_state,
        }

    def _effective_utility_spread_tol(self, best_utility):
        scale = max(1.0, abs(float(best_utility))) if np.isfinite(best_utility) else 1.0
        return max(self.utility_spread_tol, self.utility_spread_rel_tol * scale)

    def _nonsmooth_kkt_diagnostics(self, m, x, grad, smooth_pass,
                                   damage_interp_knots, cost_kink_nodes):
        """Check finite feasible directions when an adjoint lands on a cusp.

        At a damage-interpolation knot or the piecewise-cost kink at mitigation
        one, the selected adjoint slope can fail a smooth projected-gradient
        check even if utility decreases on both sides. This narrowly gated
        check tests both feasible coordinate directions for every
        projected-gradient violator, plus their joint projected-descent
        direction. It cannot override without full coverage.
        """

        projected = self._projected_gradient(x, grad)
        violating = np.where(
            np.abs(projected) > self.projected_gradient_tol
        )[0]
        diagnostics = {
            "smooth_projected_gradient_pass": bool(smooth_pass),
            "nonsmooth_kkt_check": bool(self.nonsmooth_kkt_check),
            "nonsmooth_kkt_step": float(self.nonsmooth_kkt_step),
            "nonsmooth_kkt_utility_gain_tol": float(
                self.nonsmooth_kkt_utility_gain_tol
            ),
            "nonsmooth_kkt_max_coordinates": int(
                self.nonsmooth_kkt_max_coordinates
            ),
            # Keep the original field for compatibility; it has historically
            # counted damage-interpolation knots only.
            "nonsmooth_kkt_detected_knots": int(damage_interp_knots),
            "nonsmooth_kkt_detected_damage_knots": int(damage_interp_knots),
            "nonsmooth_kkt_detected_cost_kink_nodes": int(cost_kink_nodes),
            "nonsmooth_kkt_violating_coordinates": int(len(violating)),
            "nonsmooth_kkt_tested_coordinates": 0,
            "nonsmooth_kkt_tested_nodes": "",
            "nonsmooth_kkt_evals": 0,
            "nonsmooth_kkt_max_utility_gain": np.nan,
            "nonsmooth_kkt_best_direction": "",
            "nonsmooth_kkt_all_evals_finite": True,
            "nonsmooth_kkt_coverage_complete": False,
            "nonsmooth_kkt_pass": False,
            "nonsmooth_kkt_override_applied": False,
            "nonsmooth_kkt_status": "not_run",
        }
        if smooth_pass:
            diagnostics["nonsmooth_kkt_status"] = "skipped_smooth_pass"
            return diagnostics
        if not self.nonsmooth_kkt_check:
            diagnostics["nonsmooth_kkt_status"] = "skipped_disabled"
            return diagnostics
        if self.gradient_mode != "adjoint":
            diagnostics["nonsmooth_kkt_status"] = "skipped_non_adjoint"
            return diagnostics
        if int(damage_interp_knots) <= 0 and int(cost_kink_nodes) <= 0:
            diagnostics["nonsmooth_kkt_status"] = (
                "skipped_no_damage_or_cost_kinks"
            )
            return diagnostics
        if self.nonsmooth_kkt_step <= 0.0:
            diagnostics["nonsmooth_kkt_status"] = "inconclusive_zero_step"
            return diagnostics
        if not len(violating):
            diagnostics["nonsmooth_kkt_status"] = "inconclusive_no_violators"
            return diagnostics

        order = violating[np.argsort(np.abs(projected[violating]))[::-1]]
        selected = order[:self.nonsmooth_kkt_max_coordinates]
        coverage_complete = len(selected) == len(violating)
        diagnostics["nonsmooth_kkt_coverage_complete"] = bool(coverage_complete)
        diagnostics["nonsmooth_kkt_tested_coordinates"] = int(len(selected))
        diagnostics["nonsmooth_kkt_tested_nodes"] = ",".join(
            str(int(self.active_indices[index])) for index in selected
        )

        base_utility = self._utility_at_full(np.asarray(m, dtype=float))
        diagnostics["nonsmooth_kkt_evals"] = 1
        if not np.isfinite(base_utility):
            diagnostics["nonsmooth_kkt_all_evals_finite"] = False
            diagnostics["nonsmooth_kkt_status"] = "inconclusive_nonfinite_base"
            return diagnostics

        gains = []
        tested_points = set()

        def evaluate(trial, label):
            trial = np.clip(np.asarray(trial, dtype=float), 0.0, 1.0)
            if np.array_equal(trial, x):
                return
            key = np.ascontiguousarray(trial).tobytes()
            if key in tested_points:
                return
            tested_points.add(key)
            trial_utility = self._utility_at_full(self._full_from_scaled(trial))
            diagnostics["nonsmooth_kkt_evals"] += 1
            if not np.isfinite(trial_utility):
                diagnostics["nonsmooth_kkt_all_evals_finite"] = False
                gains.append((np.nan, label))
                return
            gains.append((float(trial_utility - base_utility), label))

        for index in selected:
            node = int(self.active_indices[index])
            for sign, direction_name in ((-1.0, "down"), (1.0, "up")):
                trial = np.asarray(x, dtype=float).copy()
                trial[index] = np.clip(
                    trial[index] + sign * self.nonsmooth_kkt_step,
                    0.0,
                    1.0,
                )
                evaluate(trial, "node:{}:{}".format(node, direction_name))

        joint_direction = np.zeros_like(projected)
        joint_direction[selected] = -projected[selected]
        joint_norm = float(np.max(np.abs(joint_direction)))
        if joint_norm > 0.0:
            joint_direction /= joint_norm
            evaluate(
                np.asarray(x, dtype=float)
                + self.nonsmooth_kkt_step * joint_direction,
                "joint_projected_descent",
            )

        finite_gains = [item for item in gains if np.isfinite(item[0])]
        if finite_gains:
            best_gain, best_direction = max(finite_gains, key=lambda item: item[0])
            diagnostics["nonsmooth_kkt_max_utility_gain"] = float(best_gain)
            diagnostics["nonsmooth_kkt_best_direction"] = str(best_direction)
        if not gains:
            diagnostics["nonsmooth_kkt_status"] = "inconclusive_no_feasible_directions"
            return diagnostics
        if not diagnostics["nonsmooth_kkt_all_evals_finite"]:
            diagnostics["nonsmooth_kkt_status"] = "inconclusive_nonfinite_evaluation"
            return diagnostics
        if diagnostics["nonsmooth_kkt_max_utility_gain"] > self.nonsmooth_kkt_utility_gain_tol:
            diagnostics["nonsmooth_kkt_status"] = "failed_improving_direction"
            return diagnostics
        if not coverage_complete:
            diagnostics["nonsmooth_kkt_status"] = "inconclusive_coordinate_limit"
            return diagnostics

        diagnostics["nonsmooth_kkt_pass"] = True
        diagnostics["nonsmooth_kkt_override_applied"] = True
        diagnostics["nonsmooth_kkt_status"] = "passed_no_improving_direction"
        return diagnostics

    def _kkt_diagnostics(self, m):
        if not self.kkt_check or len(self.active_indices) == 0:
            return {
                "kkt_check": bool(self.kkt_check),
                "projected_gradient_max_abs": np.nan,
                "projected_grad_inf_norm": np.nan,
                "projected_grad_l2_norm": np.nan,
                "worst_kkt_node": -1,
                "worst_kkt_time": -1,
                "worst_kkt_state": -1,
                "projected_gradient_tol": float(self.projected_gradient_tol),
                "projected_gradient_pass": False,
                "smooth_projected_gradient_pass": False,
                "nonsmooth_kkt_status": "skipped_kkt_disabled_or_no_active_variables",
                "nonsmooth_kkt_pass": False,
                "nonsmooth_kkt_override_applied": False,
                "diagnostic_gradient_nfev": 0,
            }
        x = self._scaled_from_full(m)
        if self.gradient_mode == "adjoint":
            grad = self.scipy_objective.jac(x)
            location = self._kkt_location(x, grad)
            projected_norm = location["projected_grad_inf_norm"]
            smooth_pass = bool(projected_norm <= self.projected_gradient_tol)
            adjoint_diagnostics = {}
            if hasattr(self.objective_with_gradient, "diagnostics"):
                adjoint_diagnostics = self.objective_with_gradient.diagnostics()
            try:
                damage_interp_knots = max(
                    0, int(adjoint_diagnostics.get("num_damage_interp_knots", 0))
                )
            except (TypeError, ValueError, OverflowError):
                damage_interp_knots = 0
            try:
                cost_kink_nodes = max(
                    0, int(adjoint_diagnostics.get("num_cost_kink_nodes", 0))
                )
            except (TypeError, ValueError, OverflowError):
                cost_kink_nodes = 0
            nonsmooth = self._nonsmooth_kkt_diagnostics(
                m,
                x,
                grad,
                smooth_pass=smooth_pass,
                damage_interp_knots=damage_interp_knots,
                cost_kink_nodes=cost_kink_nodes,
            )
            location.update({
                "kkt_check": True,
                "projected_gradient_max_abs": float(projected_norm),
                "projected_gradient_tol": float(self.projected_gradient_tol),
                "projected_gradient_pass": bool(
                    smooth_pass or nonsmooth["nonsmooth_kkt_pass"]
                ),
                "diagnostic_gradient_nfev": 0,
                "diagnostic_gradient_ngev": 1,
            })
            location.update(nonsmooth)
            return location
        eval_state = {"nfev": 0}
        eval_lock = threading.Lock()

        def objective(z):
            with eval_lock:
                eval_state["nfev"] += 1
            return self._objective_value_scaled(z)

        f0 = objective(x)
        grad = self._finite_difference_gradient(
            x,
            objective,
            f0=f0,
            workers=self.gradient_workers,
        )
        location = self._kkt_location(x, grad)
        projected_norm = location["projected_grad_inf_norm"]
        location.update({
            "kkt_check": True,
            "projected_gradient_max_abs": float(projected_norm),
            "projected_gradient_tol": float(self.projected_gradient_tol),
            "projected_gradient_pass": bool(projected_norm <= self.projected_gradient_tol),
            "smooth_projected_gradient_pass": bool(
                projected_norm <= self.projected_gradient_tol
            ),
            "nonsmooth_kkt_status": "skipped_non_adjoint",
            "nonsmooth_kkt_pass": False,
            "nonsmooth_kkt_override_applied": False,
            "diagnostic_gradient_nfev": int(eval_state["nfev"]),
            "diagnostic_gradient_ngev": 0,
        })
        return location

    def _anchor_scaled_starts(self):
        anchors = []
        if not len(self.active_indices):
            return [("anchor_fixed", np.zeros(0))]
        eps = min(self.start_boundary_epsilon, 0.5)
        lower = np.full(len(self.active_indices), eps)
        upper = np.full(len(self.active_indices), 1.0 - eps)
        anchors.append(("anchor_lower", lower))
        anchors.append(("anchor_midpoint", np.full(len(self.active_indices), 0.5)))
        near_full_full = np.minimum(
            self.upper_bounds,
            min(self.near_full_mitigation, 1.0 - 1e-8),
        )
        near_full = self._scaled_from_full(near_full_full)
        if near_full is not None:
            anchors.append(("anchor_near_full", near_full))
        anchors.append(("anchor_upper", upper))
        return anchors

    def _sobol_scaled_starts(self, count, seed_offset=0):
        if count <= 0:
            return []
        dim = len(self.active_indices)
        if dim == 0:
            return [("sobol_fixed", np.zeros(0))]
        if self.start_design != "sobol":
            rng = np.random.RandomState((self.seed + seed_offset) % (2**32 - 1))
            return [
                ("random", start)
                for start in rng.random_sample((count, dim))
            ]
        sampler = qmc.Sobol(
            d=dim,
            scramble=True,
            seed=(self.seed + seed_offset) % (2**32 - 1),
        )
        power = int(np.ceil(np.log2(max(1, count))))
        return [
            ("sobol", start)
            for start in sampler.random_base2(power)[:count]
        ]

    @staticmethod
    def _is_mandatory_source(source):
        return str(source).startswith("mandatory_start_")

    def _unique_scaled(self, starts):
        unique = []
        seen = set()
        for entry in starts:
            if isinstance(entry, tuple) and len(entry) == 2:
                source, start = entry
            else:
                source, start = "unknown", entry
            if self._is_mandatory_source(source):
                # Mandatory points are contractual, including exact boundaries.
                start = np.clip(np.asarray(start, dtype=float), 0.0, 1.0)
            else:
                start = self._clip_start_scaled(start)
            key = tuple(np.round(start, 12))
            if key not in seen:
                seen.add(key)
                unique.append((source, start))
        return unique

    def _mandatory_scaled_starts(self):
        starts = []
        for index, mandatory in enumerate(self.mandatory_starts):
            scaled = self._scaled_from_full(mandatory)
            if scaled is None:
                raise RuntimeError(
                    "validated mandatory start {} could not be scaled".format(index)
                )
            starts.append((
                "mandatory_start_{}".format(index),
                np.clip(np.asarray(scaled, dtype=float), 0.0, 1.0),
            ))
        return starts

    def _warm_start_scaled_starts(self, seed_offset=0):
        starts = []
        if not len(self.active_indices):
            return starts
        rng = np.random.RandomState(
            (self.seed + seed_offset + 104729) % (2**32 - 1)
        )
        for warm in self.warm_starts:
            scaled = self._scaled_from_full(warm)
            if scaled is None:
                continue
            starts.append(("warm_start", scaled))
            for _ in range(self.warm_start_perturbations):
                perturbation = rng.normal(
                    loc=0.0,
                    scale=self.warm_start_perturbation_scale,
                    size=len(self.active_indices),
                )
                starts.append((
                    "warm_start_perturbed",
                    self._clip_start_scaled(scaled + perturbation),
                ))
        return starts

    @staticmethod
    def _source_group(source):
        source = str(source)
        if CandidateScreenedLBFGSB._is_mandatory_source(source):
            return "mandatory_start"
        if source.startswith("warm_start"):
            return "warm_start"
        if source.startswith("structured_constant"):
            return "structured_constant"
        if source.startswith("structured_ramp"):
            return "structured_ramp"
        if source.startswith("structured_sigmoid"):
            return "structured_sigmoid"
        if source.startswith("structured_plateau"):
            return "structured_plateau"
        if source.startswith("anchor"):
            return "anchor"
        if source.startswith("sobol") or source.startswith("random"):
            return "space_filling"
        return source

    def _try_select_start(self, idx, utilities, sources, start_values,
                          selected, selected_sources, selected_utilities,
                          selected_keys, best_utility, utility_gap_limit,
                          min_after_filter, enforce_gap=True):
        if len(selected) >= min_after_filter and enforce_gap:
            utility = float(utilities[idx])
            if (
                np.isfinite(utility_gap_limit)
                and np.isfinite(best_utility)
                and best_utility - utility > utility_gap_limit
            ):
                return False
        key = self._scaled_key(start_values[idx])
        if key in selected_keys:
            return False
        selected_keys.add(key)
        selected.append(start_values[idx])
        selected_sources.append(sources[idx])
        selected_utilities.append(float(utilities[idx]))
        return True

    def _structured_scaled_starts(self, count):
        if count <= 0 or not len(self.active_indices):
            return []
        dim = len(self.active_indices)
        t = np.linspace(0.0, 1.0, dim)
        starts = []
        max_m = min(self.near_full_mitigation, 1.0 - 1e-8)

        def add_profile(source, profile):
            scaled = self._scaled_from_active_mitigation(
                np.clip(np.asarray(profile, dtype=float), 0.0, max_m)
            )
            if scaled is not None:
                starts.append((source, scaled))

        for level in (0.2, 0.35, 0.5, 0.65, 0.8, 0.9, max_m):
            add_profile("structured_constant", np.full(dim, min(level, max_m)))
        for start_level in (0.0, 0.15, 0.3):
            for end_level in (0.65, 0.8, 0.9, max_m):
                for power in (0.75, 1.0, 1.5, 2.0):
                    profile = start_level + (end_level - start_level) * (t ** power)
                    add_profile("structured_ramp", profile)
        for center in (0.2, 0.35, 0.5, 0.65):
            for steepness in (6.0, 10.0, 14.0):
                profile = max_m / (1.0 + np.exp(-steepness * (t - center)))
                add_profile("structured_sigmoid", profile)
        for plateau in (0.6, 0.8, 0.9, max_m):
            for knee in (0.2, 0.4, 0.6):
                profile = plateau * np.minimum(1.0, t / max(knee, 1e-12))
                add_profile("structured_plateau", profile)
        return starts[:count]

    def _candidate_starts(self, n_candidates, seed_offset=0):
        required = self._mandatory_scaled_starts()
        required.extend(self._warm_start_scaled_starts(seed_offset))
        required.extend(self._structured_scaled_starts(self.structured_start_count))
        required.extend(self._anchor_scaled_starts())
        required = self._unique_scaled(required)
        sobol_count = max(0, int(n_candidates) - len(required))
        starts = required + self._sobol_scaled_starts(sobol_count, seed_offset)
        starts = self._unique_scaled(starts)
        return required, starts

    @staticmethod
    def _scaled_key(start):
        return tuple(np.round(np.asarray(start, dtype=float), 12))

    def _screen_candidates(self, starts, n_local_starts, required_count,
                           exclude_keys=None):
        exclude_keys = set(exclude_keys or [])
        utilities = []
        total = len(starts)
        sources = [source for source, _ in starts]
        start_values = [start for _, start in starts]
        if self.print_progress:
            print(
                "L-BFGS-B {} screening {} candidates".format(self.scenario_name, total),
                flush=True,
            )
        utilities = self._parallel_map(
            lambda start: self._utility_at_full(self._full_from_scaled(start)),
            start_values,
            self.screening_workers,
        )
        for idx, utility in enumerate(utilities, 1):
            if (
                self.print_progress
                and self.candidate_progress_every
                and (idx == total or idx % self.candidate_progress_every == 0)
            ):
                best = np.max(utilities[:idx]) if utilities else np.nan
                print(
                    "L-BFGS-B {} screened {}/{} candidates; best utility {:.12g}".format(
                        self.scenario_name, idx, total, best
                    ),
                    flush=True,
                )
        utilities = np.asarray(utilities, dtype=float)
        order = np.argsort(utilities)[::-1]
        selected = []
        selected_sources = []
        selected_utilities = []
        selected_keys = set(exclude_keys)
        best_utility = float(utilities[order[0]]) if len(order) else -np.inf
        min_after_filter = min(
            int(n_local_starts),
            max(1, int(self.min_local_starts_after_filter)),
        )
        utility_gap_limit = self.local_start_max_utility_gap
        if (
            np.isfinite(self.local_start_max_relative_utility_gap)
            and np.isfinite(best_utility)
        ):
            relative_gap_limit = (
                self.local_start_max_relative_utility_gap
                * max(1.0, abs(best_utility))
            )
            utility_gap_limit = min(utility_gap_limit, relative_gap_limit)
        self._last_local_start_utility_gap_limit = float(utility_gap_limit)

        mandatory_indices = [
            idx for idx, source in enumerate(sources)
            if self._is_mandatory_source(source)
        ]
        pending_mandatory_indices = [
            idx for idx in mandatory_indices
            if self._scaled_key(start_values[idx]) not in exclude_keys
        ]
        if len(pending_mandatory_indices) > int(n_local_starts):
            raise ValueError(
                "n_local_starts={} cannot accommodate {} mandatory starts".format(
                    int(n_local_starts), len(pending_mandatory_indices)
                )
            )
        for idx in pending_mandatory_indices:
            if not np.isfinite(utilities[idx]):
                raise RuntimeError(
                    "mandatory start {} has non-finite screened utility".format(
                        sources[idx]
                    )
                )
            selected_mandatory = self._try_select_start(
                idx, utilities, sources, start_values,
                selected, selected_sources, selected_utilities,
                selected_keys, best_utility, utility_gap_limit,
                min_after_filter, enforce_gap=False,
            )
            if not selected_mandatory:
                raise RuntimeError(
                    "failed to select mandatory start {}".format(sources[idx])
                )

        if self.preserve_diverse_starts and int(n_local_starts) > 1:
            best_by_group = {}
            for idx in order:
                group = self._source_group(sources[idx])
                if group not in best_by_group:
                    best_by_group[group] = idx
            group_priority = [
                "warm_start",
                "structured_ramp",
                "structured_sigmoid",
                "structured_plateau",
                "structured_constant",
                "space_filling",
                "anchor",
            ]
            diverse_limit = min(
                int(n_local_starts),
                max(0, int(self.min_diverse_start_groups)),
                len(best_by_group),
            )
            diverse_order = [
                best_by_group[group] for group in group_priority
                if group in best_by_group
            ]
            diverse_order.extend(
                idx for group, idx in best_by_group.items()
                if group not in group_priority
            )
            for idx in diverse_order:
                if len(selected) >= diverse_limit:
                    break
                self._try_select_start(
                    idx, utilities, sources, start_values,
                    selected, selected_sources, selected_utilities,
                    selected_keys, best_utility, utility_gap_limit,
                    min_after_filter, enforce_gap=False,
                )

        for idx in order:
            if len(selected) >= int(n_local_starts):
                break
            self._try_select_start(
                idx, utilities, sources, start_values,
                selected, selected_sources, selected_utilities,
                selected_keys, best_utility, utility_gap_limit,
                min_after_filter, enforce_gap=True,
            )
        selected_utilities = np.asarray(selected_utilities, dtype=float)
        return selected, utilities, selected_utilities, selected_sources

    def _local_solve(self, x0, gradient_workers=None):
        start_time = time.time()
        x0 = np.clip(np.asarray(x0, dtype=float), 0.0, 1.0)
        start_m = self._full_from_scaled(x0)
        start_u = self._utility_at_full(start_m)
        eval_state = {"nfev": 0}
        eval_lock = threading.Lock()
        best_eval = {
            "fun": -start_u if np.isfinite(start_u) else np.inf,
            "x": x0.copy(),
            "source": "start",
        }
        best_eval_lock = threading.Lock()

        def record_evaluation(x, fun, source):
            if not np.isfinite(fun):
                return
            with best_eval_lock:
                if fun < best_eval["fun"]:
                    best_eval["fun"] = float(fun)
                    best_eval["x"] = np.clip(np.asarray(x, dtype=float), 0.0, 1.0).copy()
                    best_eval["source"] = source

        def objective(x):
            with eval_lock:
                eval_state["nfev"] += 1
            value = self._objective_value_scaled(x)
            record_evaluation(x, value, "objective")
            return value

        gradient_state = {"count": 0}
        local_scipy_objective = (
            ScipyObjective(
                self.objective_with_gradient,
                self.lower_bounds,
                self.upper_bounds,
            )
            if self.gradient_mode == "adjoint"
            else None
        )

        def report_gradient_start(gradient_count, f0):
            if (
                self.print_progress
                and self.gradient_progress_every
                and gradient_count % self.gradient_progress_every == 0
            ):
                print(
                    "L-BFGS-B {} gradient {} start; utility {:.12g}; objective evals {}; workers {}".format(
                        self.scenario_name,
                        gradient_count,
                        -f0 if np.isfinite(f0) else np.nan,
                        eval_state["nfev"],
                        int(gradient_workers if gradient_workers is not None else self.gradient_workers),
                    ),
                    flush=True,
                )

        def report_gradient_done(gradient_count):
            if (
                self.print_progress
                and self.gradient_progress_every
                and gradient_count % self.gradient_progress_every == 0
            ):
                print(
                    "L-BFGS-B {} gradient {} done; objective evals {}".format(
                        self.scenario_name,
                        gradient_count,
                        eval_state["nfev"],
                    ),
                    flush=True,
                )

        def adjoint_objective_and_gradient(x):
            gradient_state["count"] += 1
            with eval_lock:
                eval_state["nfev"] += 1
            fun, grad = local_scipy_objective.fun_and_jac(x)
            record_evaluation(x, fun, "value_and_gradient")
            report_gradient_start(gradient_state["count"], fun)
            report_gradient_done(gradient_state["count"])
            return fun, grad

        def jacobian(x):
            gradient_state["count"] += 1
            gradient_count = gradient_state["count"]
            f0 = self._objective_value_scaled(x)
            record_evaluation(x, f0, "jacobian")
            report_gradient_start(gradient_count, f0)
            grad = self._finite_difference_gradient(
                x,
                objective,
                f0=f0,
                workers=gradient_workers,
            )
            report_gradient_done(gradient_count)
            return grad

        if len(self.active_indices) == 0:
            return {
                "m": start_m,
                "utility": start_u,
                "start_utility": start_u,
                "success": True,
                "scipy_success": True,
                "guarded_start_kept": False,
            "message": "no active variables",
            "nfev": 1,
            "ngev": 0,
            "nit": 0,
            "runtime_seconds": time.time() - start_time,
        }
        callback_state = {"nit": 0}

        def callback(xk):
            callback_state["nit"] += 1
            every = self.callback_progress_every
            if self.print_progress and every and callback_state["nit"] % every == 0:
                utility = self._utility_at_full(self._full_from_scaled(xk))
                print(
                    "L-BFGS-B {} iter {}; utility {:.12g}; objective evals {}".format(
                        self.scenario_name,
                        callback_state["nit"],
                        utility,
                        eval_state["nfev"],
                    ),
                    flush=True,
                )

        if self.gradient_mode == "adjoint":
            try:
                result = minimize(
                    adjoint_objective_and_gradient,
                    x0,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(0.0, 1.0)] * len(self.active_indices),
                    callback=callback,
                    options={"maxiter": self.maxiter, "ftol": self.ftol, "gtol": self.gtol},
                )
            except FloatingPointError as error:
                # A local adjoint evaluation can be undefined even when the
                # screened GA candidate has finite utility. Reject this local
                # refinement and retain that finite candidate as a fallback.
                return {
                    "m": start_m,
                    "utility": start_u,
                    "start_utility": start_u,
                    "success": False,
                    "scipy_success": False,
                    "guarded_start_kept": True,
                    "best_eval_retained": False,
                    "best_eval_source": "start",
                    "best_eval_utility": float(start_u) if np.isfinite(start_u) else np.nan,
                    "nonfinite_adjoint_gradient_rejected": True,
                    "message": "rejected local start after non-finite adjoint gradient: {}".format(error),
                    "nfev": int(eval_state["nfev"]),
                    "ngev": int(gradient_state["count"]),
                    "nit": int(callback_state["nit"]),
                    "runtime_seconds": time.time() - start_time,
                }
        else:
            result = minimize(
                objective,
                x0,
                jac=jacobian,
                method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * len(self.active_indices),
                callback=callback,
                options={"maxiter": self.maxiter, "ftol": self.ftol, "gtol": self.gtol},
            )
        m = self._full_from_scaled(result.x)
        u = self._utility_at_full(m)
        guard_tol = 1e-10 * max(1.0, abs(start_u))
        message = str(result.message)
        result_fun = -u if np.isfinite(u) else np.inf
        record_evaluation(result.x, result_fun, "result")
        best_eval_m = self._full_from_scaled(best_eval["x"])
        best_eval_u = self._utility_at_full(best_eval_m)
        best_eval_retained = np.isfinite(best_eval_u) and best_eval_u > u + guard_tol
        if best_eval_retained:
            m = best_eval_m
            u = best_eval_u
            message = "{}; kept best evaluated point from {}".format(
                message, best_eval["source"]
            )

        guarded_start_kept = (not np.isfinite(u)) or (u + guard_tol < start_u)
        if guarded_start_kept:
            m = start_m
            u = start_u
            best_eval_retained = False
            message = "{}; kept screened start because local solve worsened utility".format(
                message
            )
        accepted = np.isfinite(u) and (not guarded_start_kept or np.isfinite(start_u))
        return {
            "m": m,
            "utility": u,
            "start_utility": start_u,
            "success": bool(accepted),
            "scipy_success": bool(result.success),
            "guarded_start_kept": bool(guarded_start_kept),
            "best_eval_retained": bool(best_eval_retained),
            "best_eval_source": str(best_eval["source"]),
            "best_eval_utility": float(best_eval_u) if np.isfinite(best_eval_u) else np.nan,
            "nonfinite_adjoint_gradient_rejected": False,
            "message": message,
            "nfev": int(eval_state["nfev"]),
            "ngev": int(gradient_state["count"]),
            "nit": int(getattr(result, "nit", 0)),
            "runtime_seconds": time.time() - start_time,
        }

    def _bound_diagnostics(self, m):
        tol = 1e-8
        active = max(1, len(m))
        active_indices = np.asarray(self.active_indices, dtype=int)
        if len(active_indices):
            active_m = np.asarray(m)[active_indices]
            active_lower = self.lower_bounds[active_indices]
            active_upper = self.upper_bounds[active_indices]
            n_active_at_kink = int(np.sum(np.abs(active_m - 1.0) <= tol))
            active_crosses_kink = (
                (active_lower < 1.0 - tol) & (active_upper > 1.0 + tol)
            )
            n_active_crosses_kink = int(np.sum(active_crosses_kink))
            n_active_free_at_kink = int(np.sum(
                active_crosses_kink & (np.abs(active_m - 1.0) <= tol)
            ))
        else:
            n_active_at_kink = 0
            n_active_crosses_kink = 0
            n_active_free_at_kink = 0
        n_above_one = int(np.sum(m > 1.0 + tol))
        n_at_kink = int(np.sum(np.abs(m - 1.0) <= tol))
        return {
            "n_at_lower": int(np.sum(np.abs(m - self.lower_bounds) <= tol)),
            "n_at_upper": int(np.sum(np.abs(m - self.upper_bounds) <= tol)),
            "n_at_mitigation_kink": n_at_kink,
            "share_at_mitigation_kink": float(n_at_kink / float(active)),
            "all_at_mitigation_kink": bool(n_at_kink == len(m)),
            "n_active_at_mitigation_kink": n_active_at_kink,
            "share_active_at_mitigation_kink": float(
                n_active_at_kink / float(max(1, len(active_indices)))
            ),
            "all_active_at_mitigation_kink": bool(
                len(active_indices) > 0 and n_active_at_kink == len(active_indices)
            ),
            "n_active_crosses_mitigation_kink": n_active_crosses_kink,
            "n_active_free_at_mitigation_kink": n_active_free_at_kink,
            "share_active_free_at_mitigation_kink": float(
                n_active_free_at_kink / float(max(1, n_active_crosses_kink))
            ),
            "all_active_free_at_mitigation_kink": bool(
                n_active_crosses_kink > 0
                and n_active_free_at_kink == n_active_crosses_kink
            ),
            "n_above_one": n_above_one,
            "share_above_one": float(n_above_one / float(active)),
            "max_m": float(np.max(m)) if len(m) else np.nan,
            "min_m": float(np.min(m)) if len(m) else np.nan,
        }

    @staticmethod
    def _tree_summary(prefix, tree):
        values = []
        try:
            items = tree.tree.values()
        except Exception:
            return {}
        for array in items:
            arr = np.asarray(array, dtype=float).reshape(-1)
            if len(arr):
                values.extend(arr[np.isfinite(arr)].tolist())
        if not values:
            return {}
        values = np.asarray(values, dtype=float)
        return {
            "{}_min".format(prefix): float(np.min(values)),
            "{}_max".format(prefix): float(np.max(values)),
            "{}_mean".format(prefix): float(np.mean(values)),
        }

    def _welfare_decomposition_diagnostics(self, m):
        if not hasattr(self.utility, "utility"):
            return {"welfare_decomposition_available": False}
        try:
            trees = self.utility.utility(np.asarray(m, dtype=float), return_trees=True)
        except Exception:
            return {"welfare_decomposition_available": False}
        diag = {"welfare_decomposition_available": True}
        tree_map = {
            "utility_tree": "Utility",
            "consumption_tree": "Consumption",
            "cost_tree": "Cost",
            "certain_equivalence_tree": "CertainEquivalence",
        }
        for prefix, key in tree_map.items():
            if key in trees:
                diag.update(self._tree_summary(prefix, trees[key]))
        return diag

    def _perturbation_diagnostics(self, m, base_utility, effective_spread_tol):
        diag = {
            "perturbation_check": bool(self.perturbation_check),
            "perturbation_pass": True,
            "perturbation_failed": False,
            "n_perturbation_evals": 0,
            "max_perturbation_utility_gain": 0.0,
            "best_perturbation_kind": "",
            "best_perturbation_direction": "",
            "best_perturbation_size": 0,
            "perturbation_tol": float(self.perturbation_tol),
            "perturbation_rel_tol": float(self.perturbation_rel_tol),
            "effective_perturbation_tol": float(self.perturbation_tol),
        }
        if (
            not self.perturbation_check
            or self.perturbation_step <= 0.0
            or not np.isfinite(base_utility)
            or not len(self.active_indices)
        ):
            return diag
        m = np.asarray(m, dtype=float)
        tol = 1e-8
        active_indices = np.asarray(self.active_indices, dtype=int)
        active_m = m[active_indices]
        active_lower = self.lower_bounds[active_indices]
        active_upper = self.upper_bounds[active_indices]
        active_at_boundary = (
            (np.abs(active_m - active_lower) <= tol)
            | (np.abs(active_m - active_upper) <= tol)
        )
        active_crosses_kink = (
            (active_lower < 1.0 - tol) & (active_upper > 1.0 + tol)
        )
        active_at_kink = np.abs(active_m - 1.0) <= tol
        suspicious_indices = active_indices[active_at_boundary | (active_crosses_kink & active_at_kink)]
        if len(suspicious_indices) == 0:
            return diag
        effective_tol = max(
            self.perturbation_tol,
            self.perturbation_rel_tol * max(1.0, abs(float(base_utility))),
            0.1 * float(effective_spread_tol),
        )
        diag["effective_perturbation_tol"] = float(effective_tol)
        jobs = []
        for index in suspicious_indices:
            jobs.append(("node", np.array([index], dtype=int), -1.0))
            jobs.append(("node", np.array([index], dtype=int), 1.0))
        if self.perturbation_block_count > 0 and len(suspicious_indices) > 1:
            block_count = min(self.perturbation_block_count, len(suspicious_indices))
            for block in np.array_split(suspicious_indices, block_count):
                if len(block):
                    jobs.append(("block", np.asarray(block, dtype=int), -1.0))
                    jobs.append(("block", np.asarray(block, dtype=int), 1.0))

        def evaluate(job):
            kind, indices, direction = job
            trial = m.copy()
            trial[indices] = np.minimum(
                np.maximum(
                    trial[indices] + direction * self.perturbation_step,
                    self.lower_bounds[indices],
                ),
                self.upper_bounds[indices],
            )
            if np.allclose(trial[indices], m[indices], atol=0.0, rtol=0.0):
                return kind, direction, len(indices), -np.inf
            utility = self._utility_at_full(trial)
            return kind, direction, len(indices), utility - base_utility

        results = self._parallel_map(
            evaluate,
            jobs,
            self.screening_workers,
        )
        finite_results = [
            result for result in results
            if np.isfinite(result[3])
        ]
        diag["n_perturbation_evals"] = int(len(finite_results))
        if finite_results:
            best = max(finite_results, key=lambda item: item[3])
            diag["max_perturbation_utility_gain"] = float(best[3])
            diag["best_perturbation_kind"] = str(best[0])
            diag["best_perturbation_direction"] = (
                "up" if best[1] > 0.0 else "down"
            )
            diag["best_perturbation_size"] = int(best[2])
        failed = bool(diag["max_perturbation_utility_gain"] > effective_tol)
        diag["perturbation_pass"] = bool(not failed)
        diag["perturbation_failed"] = failed
        return diag

    def _gradient_validation_point(self):
        dim = len(self.active_indices)
        if dim == 0:
            return np.zeros(0)
        rng = np.random.RandomState(self.gradient_validation_seed)
        candidates = [
            0.25 + 0.5 * rng.random_sample(dim)
            for _ in range(8)
        ]
        candidates.extend(
            np.full(dim, level, dtype=float)
            for level in (0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.99)
        )
        for candidate in candidates:
            x = self._clip_start_scaled(candidate)
            m = self._full_from_scaled(x)
            near_kink = np.abs(m[self.active_indices] - 1.0) <= 0.05
            if np.any(near_kink):
                x[near_kink] = np.clip(
                    x[near_kink] - 0.15, 0.05, 0.95
                )
            x = self._clip_start_scaled(x)
            try:
                value = self._utility_at_full(self._full_from_scaled(x))
            except Exception:
                continue
            if np.isfinite(value):
                return x
        return None

    def _validate_adjoint_gradient(self):
        diag = {
            "gradient_validation_status": "skipped",
            "gradient_validation_max_abs_error": np.nan,
            "gradient_validation_max_rel_error": np.nan,
            "gradient_validation_directions": int(self.gradient_validation_directions),
            "gradient_validation_epsilons": ",".join(
                "{:.0e}".format(eps) for eps in self.gradient_validation_epsilons
            ),
            "gradient_validation_message": "",
        }
        if self.gradient_mode != "adjoint":
            self._gradient_validation_diag = diag
            return diag
        if not self.validate_gradient:
            diag["gradient_validation_status"] = "skipped"
            self._gradient_validation_diag = diag
            return diag
        x0 = self._gradient_validation_point()
        if x0 is None:
            diag["gradient_validation_status"] = "failed"
            diag["gradient_validation_message"] = (
                "no finite interior objective point found for gradient validation"
            )
            self._gradient_validation_diag = diag
            return diag
        try:
            base_grad = self.scipy_objective.jac(x0)
        except Exception as exc:
            diag["gradient_validation_status"] = "failed"
            diag["gradient_validation_message"] = str(exc)
            self._gradient_validation_diag = diag
            return diag
        rng = np.random.RandomState((self.gradient_validation_seed + 17) % (2**32 - 1))
        abs_errors = []
        rel_errors = []
        rows = []
        for direction_index in range(self.gradient_validation_directions):
            direction = rng.normal(size=len(x0))
            norm = np.linalg.norm(direction)
            if norm == 0.0:
                continue
            direction = direction / norm
            adjoint_dot = float(np.dot(base_grad, direction))
            for eps in self.gradient_validation_epsilons:
                x_plus = np.clip(x0 + eps * direction, 0.0, 1.0)
                x_minus = np.clip(x0 - eps * direction, 0.0, 1.0)
                if (
                    np.any(np.abs(self._full_from_scaled(x_plus)[self.active_indices] - 1.0) <= 1e-8)
                    or np.any(np.abs(self._full_from_scaled(x_minus)[self.active_indices] - 1.0) <= 1e-8)
                ):
                    continue
                f_plus = self.scipy_objective.fun(x_plus)
                f_minus = self.scipy_objective.fun(x_minus)
                fd = float((f_plus - f_minus) / (2.0 * eps))
                abs_error = abs(fd - adjoint_dot)
                rel_error = abs_error / max(1.0, abs(fd), abs(adjoint_dot))
                abs_errors.append(abs_error)
                rel_errors.append(rel_error)
                rows.append((eps, fd, adjoint_dot, abs_error, rel_error))
        if not rows:
            diag["gradient_validation_status"] = "failed"
            diag["gradient_validation_message"] = "no valid validation directions away from kink/bounds"
            self._gradient_validation_diag = diag
            return diag
        best_row = min(rows, key=lambda row: row[4])
        diag.update({
            "gradient_validation_status": "passed",
            "gradient_validation_max_abs_error": float(max(abs_errors)),
            "gradient_validation_max_rel_error": float(max(rel_errors)),
            "gradient_validation_best_epsilon": float(best_row[0]),
            "gradient_validation_best_fd_directional": float(best_row[1]),
            "gradient_validation_best_adjoint_dot": float(best_row[2]),
            "gradient_validation_best_abs_error": float(best_row[3]),
            "gradient_validation_best_rel_error": float(best_row[4]),
            "gradient_validation_message": "directional derivative validation passed",
        })
        if (
            diag["gradient_validation_max_abs_error"] > self.gradient_validation_abs_tol
            and diag["gradient_validation_max_rel_error"] > self.gradient_validation_rel_tol
        ):
            diag["gradient_validation_status"] = "failed"
            diag["gradient_validation_message"] = (
                "directional derivative validation failed"
            )
        self._gradient_validation_diag = diag
        return diag

    @staticmethod
    def _finalize_lbfgsb_diagnostics(diag):
        dispersion_failed = bool(
            diag["final_utility_spread"] > diag["effective_utility_spread_tol"]
        )
        accepted = bool(diag.get("lbfgsb_best_result_accepted", False))
        active_variables = int(diag.get("n_active_variables", 0))
        kkt_check = bool(diag.get("kkt_check", False))
        projected_gradient_pass = bool(diag.get("projected_gradient_pass", False))
        scipy_success = bool(diag.get("lbfgsb_scipy_success", False))
        if active_variables == 0:
            stationarity_ok = True
        elif kkt_check:
            stationarity_ok = projected_gradient_pass
        else:
            stationarity_ok = True
        mitigation_kink_failed = bool(
            diag.get("all_active_free_at_mitigation_kink", False)
            and not diag.get("nonsmooth_kkt_pass", False)
        )
        perturbation_failed = bool(diag.get("perturbation_failed", False))
        diag["dispersion_failed"] = dispersion_failed
        diag["stationarity_failed"] = bool(not stationarity_ok)
        diag["mitigation_kink_failed"] = mitigation_kink_failed
        diag["perturbation_failed"] = perturbation_failed
        diag["lbfgsb_converged"] = bool(
            (not dispersion_failed)
            and stationarity_ok
            and (not mitigation_kink_failed)
            and (not perturbation_failed)
        )
        diag["lbfgsb_success"] = bool(accepted and diag["lbfgsb_converged"])
        diag["success_diagnostics"] = bool(diag["lbfgsb_success"])
        return diag

    def _stage(self, n_candidates, n_local_starts, seed_offset=0,
               previous_results=None, exclude_keys=None, stage_start=None):
        if stage_start is None:
            stage_start = time.time()
        previous_results = list(previous_results or [])
        exclude_keys = set(exclude_keys or [])
        required, candidates = self._candidate_starts(n_candidates, seed_offset)
        selected, candidate_utilities, selected_start_utilities, selected_sources = self._screen_candidates(
            candidates, n_local_starts, len(required), exclude_keys=exclude_keys
        )
        new_results = []
        new_keys = []
        requested_local_starts = int(n_local_starts)
        local_workers = min(max(1, self.local_start_workers), max(1, len(selected)))
        gradient_workers_per_start = max(
            1, int(self.gradient_workers / max(1, local_workers))
        )

        def solve_selected(item):
            index, start = item
            result = self._local_solve(
                start,
                gradient_workers=gradient_workers_per_start,
            )
            if index < len(selected_sources):
                result["start_source"] = selected_sources[index]
            else:
                result["start_source"] = "unknown"
            return index, result

        def print_start(index, start):
            start_utility = (
                selected_start_utilities[index]
                if index < len(selected_start_utilities) else
                self._utility_at_full(self._full_from_scaled(start))
            )
            if self.print_progress:
                print(
                    "L-BFGS-B {} local start {} of {}; source {}; screened utility {:.12g}".format(
                        self.scenario_name,
                        index + 1,
                        len(selected),
                        selected_sources[index] if index < len(selected_sources) else "unknown",
                        start_utility,
                    ),
                    flush=True,
                )

        def record_solved(index, result):
            start = selected[index]
            new_results.append(result)
            new_keys.append(self._scaled_key(start))
            if self.print_progress:
                print(
                    "L-BFGS-B {} local start {} done; source {}; utility {:.12g}; start {:.12g}; success {}; scipy_success {}; guarded {}; nit {}; nfev {}; seconds {:.1f}".format(
                        self.scenario_name,
                        index + 1,
                        result.get("start_source", "unknown"),
                        result["utility"],
                        result["start_utility"],
                        result["success"],
                        result["scipy_success"],
                        result["guarded_start_kept"],
                        result["nit"],
                        result["nfev"],
                        result["runtime_seconds"],
                    ),
                    flush=True,
                )

        if local_workers <= 1:
            for index, start in enumerate(selected):
                print_start(index, start)
                solved_index, result = solve_selected((index, start))
                record_solved(solved_index, result)
        else:
            for index, start in enumerate(selected):
                print_start(index, start)
            solved = self._parallel_map(
                solve_selected,
                list(enumerate(selected)),
                local_workers,
            )
            solved = sorted(solved, key=lambda item: item[0])
            for index, result in solved:
                record_solved(index, result)
        local_results = previous_results + new_results
        mandatory_candidate_count = int(sum(
            1 for source, _ in candidates
            if self._is_mandatory_source(source)
        ))
        mandatory_local_count = int(sum(
            1 for result in local_results
            if self._is_mandatory_source(result.get("start_source", ""))
        ))
        if mandatory_local_count < mandatory_candidate_count:
            raise RuntimeError(
                "only {} of {} mandatory starts reached the local solver".format(
                    mandatory_local_count, mandatory_candidate_count
                )
            )
        final_utilities = np.array([r["utility"] for r in local_results])
        best_index = int(np.argmax(final_utilities)) if len(final_utilities) else 0
        best_result = local_results[best_index]
        final_order = np.argsort(final_utilities)[::-1]
        sorted_final = final_utilities[final_order]
        best_final = float(sorted_final[0]) if len(sorted_final) else -np.inf
        second_best = float(sorted_final[1]) if len(sorted_final) > 1 else np.nan
        median_final = float(np.median(final_utilities)) if len(final_utilities) else np.nan
        spread = best_final - second_best if np.isfinite(second_best) else 0.0
        median_spread = best_final - median_final if np.isfinite(median_final) else 0.0
        spread_rel = float(spread / max(1.0, abs(best_final))) if np.isfinite(best_final) else np.nan
        effective_spread_tol = self._effective_utility_spread_tol(best_final)
        best_m = best_result["m"]
        if len(sorted_final) > 1:
            second_result = local_results[int(final_order[1])]
            best_second_diff = np.asarray(best_m) - np.asarray(second_result["m"])
            best_second_linf = float(np.max(np.abs(best_second_diff)))
            best_second_l2 = float(np.linalg.norm(best_second_diff))
        else:
            best_second_linf = np.nan
            best_second_l2 = np.nan
        near_best_mask = final_utilities >= best_final - effective_spread_tol
        near_best_results = [
            local_results[index] for index in np.where(near_best_mask)[0]
        ]
        if near_best_results:
            near_distances = [
                float(np.max(np.abs(np.asarray(result["m"]) - np.asarray(best_m))))
                for result in near_best_results
            ]
            near_best_solution_linf = float(np.max(near_distances))
        else:
            near_best_solution_linf = np.nan
        unique_solutions = {
            tuple(np.round(r["m"], 8)) for r in local_results if np.isfinite(r["utility"])
        }
        if best_result.get("nonfinite_adjoint_gradient_rejected", False):
            kkt_diag = {
                "kkt_check": bool(self.kkt_check),
                "projected_gradient_max_abs": np.nan,
                "projected_grad_inf_norm": np.nan,
                "projected_grad_l2_norm": np.nan,
                "worst_kkt_node": -1,
                "worst_kkt_time": -1,
                "worst_kkt_state": -1,
                "projected_gradient_tol": float(self.projected_gradient_tol),
                "projected_gradient_pass": False,
                "diagnostic_gradient_nfev": 0,
                "diagnostic_gradient_ngev": 0,
                "kkt_status": "skipped_nonfinite_adjoint_gradient",
            }
        else:
            kkt_diag = self._kkt_diagnostics(best_m)
        diag = {
            "optimizer": self.optimizer_name,
            "optimizer_name": self.optimizer_name,
            "gradient_mode": self.gradient_mode,
            "n_active_variables": int(len(self.active_indices)),
            "n_fixed_variables": int(len(self.fixed_indices)),
            "n_workers": int(self.n_workers),
            "screening_workers": int(self.screening_workers),
            "gradient_workers": int(self.gradient_workers),
            "local_start_workers": int(self.local_start_workers),
            "gradient_workers_per_start": int(gradient_workers_per_start),
            "n_candidates_evaluated": int(len(candidates)),
            "n_required_candidates": int(len(required)),
            "n_mandatory_starts_requested": int(len(self.mandatory_starts)),
            "n_mandatory_start_candidates": mandatory_candidate_count,
            "n_mandatory_local_starts": mandatory_local_count,
            "mandatory_starts_selected": bool(
                mandatory_candidate_count > 0
                and mandatory_local_count >= mandatory_candidate_count
            ),
            "n_structured_start_candidates": int(sum(
                1 for source, _ in candidates if str(source).startswith("structured_")
            )),
            "n_warm_start_candidates": int(sum(
                1 for source, _ in candidates if str(source).startswith("warm_start")
            )),
            "n_anchor_start_candidates": int(sum(
                1 for source, _ in candidates if str(source).startswith("anchor_")
            )),
            "n_local_starts": int(len(local_results)),
            "n_new_local_starts": int(len(new_results)),
            "n_prior_local_starts": int(len(previous_results)),
            "n_local_starts_requested": int(requested_local_starts),
            "n_local_starts_filtered": int(max(0, requested_local_starts - len(selected))),
            "local_start_max_utility_gap": float(self.local_start_max_utility_gap),
            "local_start_max_relative_utility_gap": float(
                self.local_start_max_relative_utility_gap
            ),
            "local_start_effective_utility_gap": float(
                getattr(self, "_last_local_start_utility_gap_limit", np.nan)
            ),
            "min_local_starts_after_filter": int(self.min_local_starts_after_filter),
            "preserve_diverse_starts": bool(self.preserve_diverse_starts),
            "min_diverse_start_groups": int(self.min_diverse_start_groups),
            "selected_start_sources": ",".join(selected_sources),
            "selected_start_source_groups": ",".join(
                self._source_group(source) for source in selected_sources
            ),
            "n_selected_start_source_groups": int(len({
                self._source_group(source) for source in selected_sources
            })),
            "best_start_utility": float(np.max(candidate_utilities)) if len(candidate_utilities) else np.nan,
            "median_start_utility": float(np.median(candidate_utilities)) if len(candidate_utilities) else np.nan,
            "worst_start_utility": float(np.min(candidate_utilities)) if len(candidate_utilities) else np.nan,
            "best_selected_start_utility": float(np.max(selected_start_utilities)) if len(selected_start_utilities) else np.nan,
            "median_selected_start_utility": float(np.median(selected_start_utilities)) if len(selected_start_utilities) else np.nan,
            "worst_selected_start_utility": float(np.min(selected_start_utilities)) if len(selected_start_utilities) else np.nan,
            "best_final_utility": best_final,
            "second_best_final_utility": second_best,
            "median_final_utility": median_final,
            "final_utility_spread": float(spread),
            "final_utility_spread_rel": float(spread_rel),
            "utility_spread_tol": float(self.utility_spread_tol),
            "utility_spread_rel_tol": float(self.utility_spread_rel_tol),
            "effective_utility_spread_tol": float(effective_spread_tol),
            "best_minus_median_final_utility": float(median_spread),
            "n_near_best_final_utility": int(np.sum(near_best_mask)),
            "best_second_solution_linf": best_second_linf,
            "best_second_solution_l2": best_second_l2,
            "near_best_solution_linf": near_best_solution_linf,
            "n_unique_local_solutions": int(len(unique_solutions)),
            "escalation_used": False,
            "fallback_used": bool(
                best_result.get("nonfinite_adjoint_gradient_rejected", False)
            ),
            "lbfgsb_best_result_accepted": bool(best_result["success"]),
            "lbfgsb_success": False,
            "lbfgsb_message": best_result["message"],
            "lbfgsb_scipy_success": bool(best_result["scipy_success"]),
            "success_scipy": bool(best_result["scipy_success"]),
            "lbfgsb_guarded_start_kept": bool(best_result["guarded_start_kept"]),
            "lbfgsb_best_eval_retained": bool(best_result.get("best_eval_retained", False)),
            "lbfgsb_best_eval_source": str(best_result.get("best_eval_source", "")),
            "lbfgsb_best_eval_utility": float(best_result.get("best_eval_utility", np.nan)),
            "n_guarded_start_kept": int(sum(r["guarded_start_kept"] for r in local_results)),
            "n_nonfinite_adjoint_gradient_rejections": int(sum(
                r.get("nonfinite_adjoint_gradient_rejected", False)
                for r in local_results
            )),
            "n_best_eval_retained": int(sum(r.get("best_eval_retained", False) for r in local_results)),
            "lbfgsb_nfev": int(sum(r["nfev"] for r in local_results)),
            "lbfgsb_ngev": int(sum(r.get("ngev", 0) for r in local_results)),
            "lbfgsb_nit": int(sum(r["nit"] for r in local_results)),
            "utility_value": best_final,
            "objective_value": float(-best_final) if np.isfinite(best_final) else np.inf,
            "runtime_seconds": float(time.time() - stage_start),
        }
        diag.update(self._gradient_validation_diag)
        diag.update(kkt_diag)
        diag.update(self._bound_diagnostics(best_result["m"]))
        diag.update({
            "num_active_bounds": int(diag.get("n_at_lower", 0) + diag.get("n_at_upper", 0)),
            "num_nodes_near_m_eq_1": int(diag.get("n_at_mitigation_kink", 0)),
            "num_nodes_near_upper_bound": int(diag.get("n_at_upper", 0)),
            "num_nodes_near_lower_bound": int(diag.get("n_at_lower", 0)),
        })
        diag.update(self._perturbation_diagnostics(
            best_result["m"], best_final, effective_spread_tol
        ))
        diag["perturbation_check_status"] = (
            "failed" if diag.get("perturbation_failed", False) else
            ("passed" if diag.get("perturbation_check", False) else "skipped")
        )
        diag.update(self._welfare_decomposition_diagnostics(best_result["m"]))
        diag["_local_results"] = local_results
        if self.objective_with_gradient is not None and hasattr(
            self.objective_with_gradient, "diagnostics"
        ):
            if self.gradient_mode == "adjoint":
                if best_result.get("nonfinite_adjoint_gradient_rejected", False):
                    diag["adjoint_diagnostics_status"] = "skipped_nonfinite_adjoint_gradient"
                else:
                    self.objective_with_gradient.value_and_gradient(best_result["m"])
            diag.update(self.objective_with_gradient.diagnostics())
        return (
            best_result["m"],
            float(best_result["utility"]),
            diag,
            local_results,
            exclude_keys.union(new_keys),
        )

    def run(self):
        stage_start = time.time()
        validation = self._validate_adjoint_gradient()
        if validation.get("gradient_validation_status") == "failed":
            raise RuntimeError(
                "adjoint_lbfgsb gradient validation failed: {}".format(
                    validation.get("gradient_validation_message", "")
                )
            )
        m, utility, diag, local_results, selected_keys = self._stage(
            self.n_candidates, self.n_local_starts, stage_start=stage_start
        )
        should_escalate = (
            self.escalate_on_dispersion
            and diag["final_utility_spread"] > diag["effective_utility_spread_tol"]
            and (self.n_candidates < self.max_candidates
                 or self.n_local_starts < self.max_local_starts)
        )
        if should_escalate:
            if self.print_progress:
                print(
                    "L-BFGS-B {} escalating because best-second utility gap {:.12g} exceeds {:.12g}".format(
                        self.scenario_name,
                        diag["final_utility_spread"],
                        diag["effective_utility_spread_tol"],
                    ),
                    flush=True,
                )
            next_candidates = min(self.max_candidates, max(self.n_candidates * 2, self.n_candidates + 1))
            remaining_local = max(0, self.max_local_starts - len(local_results))
            next_local = min(
                remaining_local,
                max(self.n_local_starts * 2, self.n_local_starts + 1),
            )
            m2, utility2, diag2, _, _ = self._stage(
                next_candidates,
                next_local,
                seed_offset=7919,
                previous_results=local_results,
                exclude_keys=selected_keys,
                stage_start=stage_start,
            )
            diag2["escalation_used"] = True
            self._finalize_lbfgsb_diagnostics(diag2)
            if utility2 >= utility:
                return m2, utility2, diag2
            diag["escalation_used"] = True
        self._finalize_lbfgsb_diagnostics(diag)
        diag["success_diagnostics"] = bool(diag.get("lbfgsb_success", False))
        return m, utility, diag
