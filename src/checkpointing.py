import numpy as np


class Checkpointing:

    def __init__(self, delta_t, t0=0.0):
        """
        Store the numerical solution at regular time intervals.

        Parameters
        ----------
        delta_t : float
            Time interval between checkpoints, in seconds.
        """

        self.delta_t = delta_t

        # Storage
        self.ichecked = []
        self.tchecked = []
        self.uchecked = []

        # First checkpoint
        self.time2check = t0


    def add(self, iteration, t, u):
        """
        Store the current numerical solution.
        """

        print(f"Checkpointing data at t = {t:.2f} s, iteration = {iteration}.")

        self.ichecked.append(iteration)
        self.tchecked.append(t)

        # Important: copy the array
        self.uchecked.append(np.copy(u))

        # Schedule next checkpoint
        self.time2check += self.delta_t