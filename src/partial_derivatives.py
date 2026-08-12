import numpy as np

import src.total_derivatives


class Partial:
    def __init__(
            self,
            function: np.ndarray,
            x: np.ndarray = None,
            y: np.ndarray = None
    ) -> \
            None:
        self.function = function
        self.x = x if x is not None else np.linspace(0, 1, function.shape[1])
        self.y = y if y is not None else np.linspace(0, 1, function.shape[0])


    def x_is_periodic(self) -> bool:
        """
        Check if function is periodic in x-direction.
        Compares first and last columns.
        """
        return np.allclose(self.function[:, 0], self.function[:, -1])


    def y_is_periodic(self) -> bool:
        """
        Check if function is periodic in y-direction.
        Compares first and last rows.
        """
        return np.allclose(self.function[0, :], self.function[-1, :])


    def biased_dx(self) -> np.ndarray:
        """
        Uses central difference approximation for interior points, forward difference approximation
        at the first point, and backwards difference approximation at the last point to compute the
        partial derivative with respect to x-direction in uniform grids.
        """
        return np.apply_along_axis(
            src.total_derivatives.biased_central_difference_approximation,
            axis=1,
            arr=self.function,
            domain=self.x
        )


    def periodic_dx(self) -> np.ndarray:
        """Uses periodic wrapping at boundaries along with central difference approximation
         to compute the partial derivative with respect to x-direction."""
        return np.apply_along_axis(
            src.total_derivatives.periodic_central_difference_approximation,
            axis=1,
            arr=self.function,
            domain=self.x
        )


    def biased_dy(self) -> np.ndarray:
        """
        Uses central difference approximation for interior points, forward difference approximation
        at the first point, and backwards difference approximation at the last point to compute the
        partial derivative with respect to y-direction in uniform grids.
        """
        return np.apply_along_axis(
            src.total_derivatives.biased_central_difference_approximation,
            axis=0,
            arr=self.function,
            domain=self.y
        )


    def periodic_dy(self) -> np.ndarray:
        """Uses periodic wrapping at boundaries along with central difference approximation
         to compute the partial derivative with respect to y-direction in uniform grids."""
        return np.apply_along_axis(
            src.total_derivatives.periodic_central_difference_approximation,
            axis=0,
            arr=self.function,
            domain=self.y
        )


    def dx(self) -> np.ndarray:
        """
        Compute the partial derivative with respect to x-direction in uniform grids choosing
        the most appropriate scheme to handle the boundary conditions (biased or periodic).
        """
        if self.x_is_periodic():
            return self.periodic_dx()
        else:
            return self.biased_dx()


    def dy(self) -> np.ndarray:
        """
        Compute the partial derivative with respect to y-direction in uniform grids choosing
        the most appropriate scheme to handle the boundary conditions (biased or periodic).
        """
        if self.y_is_periodic():
            return self.periodic_dy()
        else:
            return self.biased_dy()


    def biased_dxx(self) -> np.ndarray:
        """
        Compute the second order partial derivative approximation with respect to x in
        uniform grids using biased forward difference approximation for the
        first step and biased backward difference approximation for the last step.
        """
        return np.apply_along_axis(
            src.total_derivatives.biased_second_order_derivative_approximation,
            axis=1,
            arr=self.function,
            domain=self.x
        )


    def neumann_dxx(self):
        """
        Compute the second order partial derivative approximation with respect to x in
        uniform grids using Von Neumann boundary conditions.
        """
        return np.apply_along_axis(
            src.total_derivatives.neumann_second_order_derivative_approximation,
            axis=1,
            arr=self.function,
            domain=self.x
        )


    def periodic_dxx(self) -> np.ndarray:
        """
        Compute the second order partial derivative approximation in uniform grids
        with respect to the x-direction of the function in uniform grids using the
        periodic boundary condition.
        """
        return np.apply_along_axis(
            src.total_derivatives.periodic_second_order_derivative_approximation,
            axis=1,
            arr=self.function,
            domain=self.x
        )


    def biased_dyy(self) -> np.ndarray:
        """
        Compute the second order partial derivative approximation with respect to y in
        uniform grids using biased forward difference approximation for the
        first step and biased backward difference approximation for the last step.
        """
        return np.apply_along_axis(
            src.total_derivatives.biased_second_order_derivative_approximation,
            axis=0,
            arr=self.function,
            domain=self.y
        )


    def neumann_dyy(self):
        """
        Compute the second order partial derivative approximation with respect to y in
        uniform grids using Von Neumann boundary conditions.
        """
        return np.apply_along_axis(
            src.total_derivatives.neumann_second_order_derivative_approximation,
            axis=0,
            arr=self.function,
            domain=self.y
        )


    def periodic_dyy(self) -> np.ndarray:
        """
        Compute the second order partial derivative approximation with respect to the y-direction
        of the function in uniform grids using the periodic boundary condition.
        """
        return np.apply_along_axis(
            src.total_derivatives.periodic_second_order_derivative_approximation,
            axis=0,
            arr=self.function,
            domain=self.y
        )


    def dxx(self):
        """Compute the second order partial derivative with respect to x-direction
        with the most appropriate scheme."""
        if self.x_is_periodic():
            return self.periodic_dxx()
        else:
            return self.biased_dxx()


    def dyy(self):
        """Compute the second order partial derivative with respect to y-direction
        with the most appropriate scheme."""
        if self.y_is_periodic():
            return self.periodic_dyy()
        else:
            return self.biased_dyy()


    def at_x0(self, x0: float) -> np.ndarray:
        x0 = np.argmin(self.x - x0)
        return None


    def __str__(self) -> str:
        return (
            f"Partial(field shape = {self.function.shape},"
            f"x = {self.x},"
            f"y = {self.y}),"
            f"x_periodic={self.x_is_periodic()}, "
            f"y_periodic={self.y_is_periodic()})"
        )


    def __repr__(self) -> str:
        return (
            f"Partial(function=np.array(shape={self.function.shape}), "
            f"x=np.array(len={len(self.x)}), "
            f"y=np.array(len={len(self.y)}))"
        )


    def __add__(self, other):
        if not isinstance(other, Partial):
            return NotImplemented

        if self.function.shape != other.function.shape:
            raise ValueError("Function shapes must match.")

        if not np.allclose(self.x, other.x) or not np.allclose(self.y, other.y):
            raise ValueError("Domains must match.")

        return Partial(
            self.function + other.function,
            self.x,
            other.y
        )

    def __sub__(self, other):
        if not isinstance(other, Partial):
            return NotImplemented

        if self.function.shape != other.function.shape:
            raise ValueError("Function shapes must match.")

        if not np.allclose(self.x, other.x) or not np.allclose(self.y, other.y):
            raise ValueError("Domains must match.")

        return Partial(
            self.function - other.function,
            self.x,
            other.y
        )

    def __mul__(self, other: float):
        return Partial(self.function * other, self.x, self.y)