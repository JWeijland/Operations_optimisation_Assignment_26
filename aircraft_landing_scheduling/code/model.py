#!/usr/bin/env python3
"""
MIP Model for Aircraft Landing Scheduling
Mathematical formulation based on Beasley et al. (2000)
Uses PuLP for optimization modeling
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from pulp import *
from .data_loader import ProblemInstance, Aircraft


@dataclass
class Solution:
    """
    Represents a solution to the aircraft landing problem.

    Attributes:
        landing_times: Dictionary mapping aircraft ID to landing time
        runway_assignments: Dictionary mapping aircraft ID to runway number
        objective_value: Total cost of the solution
        solve_time: Time taken to solve (seconds)
        status: Solution status (Optimal, Feasible, Infeasible, etc.)
        gap: Optimality gap (for MIP solutions)
    """
    landing_times: Dict[int, float]
    runway_assignments: Dict[int, int]
    objective_value: float
    solve_time: float
    status: str
    gap: float = 0.0

    def get_landing_time(self, aircraft_id: int) -> float:
        """Get landing time for specific aircraft."""
        return self.landing_times.get(aircraft_id, -1)

    def get_runway(self, aircraft_id: int) -> int:
        """Get runway assignment for specific aircraft."""
        return self.runway_assignments.get(aircraft_id, -1)

    def get_schedule_by_runway(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get landing schedule organized by runway.

        Returns:
            Dictionary mapping runway number to list of (aircraft_id, landing_time) tuples
        """
        schedule = {}
        for aircraft_id, runway in self.runway_assignments.items():
            if runway not in schedule:
                schedule[runway] = []
            schedule[runway].append((aircraft_id, self.landing_times[aircraft_id]))

        # Sort by landing time within each runway
        for runway in schedule:
            schedule[runway].sort(key=lambda x: x[1])

        return schedule

    def is_feasible(self) -> bool:
        """Check if solution status indicates feasibility."""
        return self.status in ['Optimal', 'Feasible']

    def __repr__(self) -> str:
        return (f"Solution(aircraft={len(self.landing_times)}, "
                f"runways={len(set(self.runway_assignments.values()))}, "
                f"cost={self.objective_value:.2f}, "
                f"status={self.status})")


class AircraftLandingModel:
    """
    Mixed Integer Programming model for aircraft landing scheduling.

    Based on the formulation in Beasley et al. (2000):

    Decision Variables:
        x_i: Landing time of aircraft i
        α_i: Time before target (early)
        β_i: Time after target (late)
        δ_ij: Binary, 1 if aircraft i lands before j
        y_ir: Binary, 1 if aircraft i assigned to runway r (multiple runways)

    Objective:
        Minimize Σ(g_i * α_i + h_i * β_i)

    Constraints:
        1. Time windows: E_i ≤ x_i ≤ L_i
        2. Target deviation: x_i = T_i - α_i + β_i
        3. Separation: If i before j on same runway: x_j ≥ x_i + S_ij
        4. Ordering: δ_ij + δ_ji = 1 (one must be before the other on same runway)
        5. Runway assignment: Σ y_ir = 1 (each aircraft to one runway)
    """

    def __init__(self, instance: ProblemInstance, num_runways: int = 1):
        """
        Initialize the MIP model.

        Args:
            instance: Problem instance data
            num_runways: Number of available runways
        """
        self.instance = instance
        self.num_runways = num_runways
        self.model = None
        self.variables = {}

        # Model parameters
        self.big_m = self._calculate_big_m()

    def _calculate_big_m(self) -> float:
        """
        Calculate Big-M constant for disjunctive constraints.
        M should be larger than any possible time difference.
        """
        max_time = max(a.latest_time for a in self.instance.aircraft)
        max_sep = np.max(self.instance.separation_matrix)
        return max_time + max_sep + 100

    def build_model(self):
        """
        Build the complete MIP model.
        """
        print(f"Building MIP model: {self.instance.num_aircraft} aircraft, "
              f"{self.num_runways} runway(s)")

        # Create model
        self.model = LpProblem("Aircraft_Landing_Scheduling", LpMinimize)

        # Create variables
        self._create_variables()

        # Set objective function
        self._set_objective()

        # Add constraints
        self._add_time_window_constraints()
        self._add_target_deviation_constraints()
        self._add_separation_constraints()

        if self.num_runways > 1:
            self._add_runway_assignment_constraints()

        print(f"Model built: {self.model.numVariables()} variables, "
              f"{self.model.numConstraints()} constraints")

    def _create_variables(self):
        """Create all decision variables."""
        aircraft = self.instance.aircraft
        n = len(aircraft)

        # Landing time variables: x_i
        self.variables['x'] = {
            a.id: LpVariable(f"x_{a.id}",
                           lowBound=a.appearance_time,
                           upBound=a.latest_time,
                           cat='Continuous')
            for a in aircraft
        }

        # Early/late deviation variables: α_i, β_i
        self.variables['alpha'] = {
            a.id: LpVariable(f"alpha_{a.id}",
                           lowBound=0,
                           upBound=a.target_time - a.appearance_time,
                           cat='Continuous')
            for a in aircraft
        }

        self.variables['beta'] = {
            a.id: LpVariable(f"beta_{a.id}",
                           lowBound=0,
                           upBound=a.latest_time - a.target_time,
                           cat='Continuous')
            for a in aircraft
        }

        # Ordering variables: δ_ij
        self.variables['delta'] = {}
        for i in range(n):
            for j in range(i + 1, n):
                id_i = aircraft[i].id
                id_j = aircraft[j].id
                self.variables['delta'][(id_i, id_j)] = LpVariable(
                    f"delta_{id_i}_{id_j}",
                    cat='Binary'
                )

        # Runway assignment variables (if multiple runways): y_ir
        if self.num_runways > 1:
            self.variables['y'] = {}
            for a in aircraft:
                for r in range(1, self.num_runways + 1):
                    self.variables['y'][(a.id, r)] = LpVariable(
                        f"y_{a.id}_{r}",
                        cat='Binary'
                    )

            # Same runway indicator: z_ij (1 if i and j on same runway)
            self.variables['z'] = {}
            for i in range(n):
                for j in range(i + 1, n):
                    id_i = aircraft[i].id
                    id_j = aircraft[j].id
                    self.variables['z'][(id_i, id_j)] = LpVariable(
                        f"z_{id_i}_{id_j}",
                        cat='Binary'
                    )

    def _set_objective(self):
        """Set the objective function: minimize total cost."""
        aircraft = self.instance.aircraft

        objective = lpSum([
            a.early_penalty * self.variables['alpha'][a.id] +
            a.late_penalty * self.variables['beta'][a.id]
            for a in aircraft
        ])

        self.model += objective, "Total_Cost"

    def _add_time_window_constraints(self):
        """
        Add time window constraints: E_i ≤ x_i ≤ L_i
        (Already enforced by variable bounds, but we can add explicitly)
        """
        for a in self.instance.aircraft:
            self.model += (
                self.variables['x'][a.id] >= a.appearance_time,
                f"EarlyBound_{a.id}"
            )
            self.model += (
                self.variables['x'][a.id] <= a.latest_time,
                f"LateBound_{a.id}"
            )

    def _add_target_deviation_constraints(self):
        """
        Add target time deviation constraints: x_i = T_i - α_i + β_i
        """
        for a in self.instance.aircraft:
            self.model += (
                self.variables['x'][a.id] ==
                a.target_time - self.variables['alpha'][a.id] +
                self.variables['beta'][a.id],
                f"TargetDeviation_{a.id}"
            )

    def _add_separation_constraints(self):
        """
        Add separation constraints between aircraft.

        For single runway:
            If δ_ij = 1: x_j ≥ x_i + S_ij
            If δ_ij = 0: x_i ≥ x_j + S_ji

        For multiple runways:
            Only enforce if on same runway (z_ij = 1)
        """
        aircraft = self.instance.aircraft
        n = len(aircraft)

        for i in range(n):
            for j in range(i + 1, n):
                id_i = aircraft[i].id
                id_j = aircraft[j].id
                sep_ij = self.instance.get_separation(i, j)
                sep_ji = self.instance.get_separation(j, i)

                delta_ij = self.variables['delta'][(id_i, id_j)]
                x_i = self.variables['x'][id_i]
                x_j = self.variables['x'][id_j]

                if self.num_runways == 1:
                    # Single runway: one ordering must be respected
                    # If delta_ij = 1 (i before j): x_j >= x_i + S_ij
                    self.model += (
                        x_j >= x_i + sep_ij - self.big_m * (1 - delta_ij),
                        f"Sep_{id_i}_before_{id_j}"
                    )

                    # If delta_ij = 0 (j before i): x_i >= x_j + S_ji
                    self.model += (
                        x_i >= x_j + sep_ji - self.big_m * delta_ij,
                        f"Sep_{id_j}_before_{id_i}"
                    )

                else:
                    # Multiple runways: only enforce if on same runway
                    z_ij = self.variables['z'][(id_i, id_j)]

                    # If z_ij = 1 and delta_ij = 1: x_j >= x_i + S_ij
                    self.model += (
                        x_j >= x_i + sep_ij - self.big_m * (2 - z_ij - delta_ij),
                        f"Sep_{id_i}_before_{id_j}"
                    )

                    # If z_ij = 1 and delta_ij = 0: x_i >= x_j + S_ji
                    self.model += (
                        x_i >= x_j + sep_ji - self.big_m * (1 + delta_ij - z_ij),
                        f"Sep_{id_j}_before_{id_i}"
                    )

    def _add_runway_assignment_constraints(self):
        """
        Add constraints for multiple runway case.
        """
        aircraft = self.instance.aircraft
        n = len(aircraft)

        # Each aircraft assigned to exactly one runway
        for a in aircraft:
            self.model += (
                lpSum([self.variables['y'][(a.id, r)]
                      for r in range(1, self.num_runways + 1)]) == 1,
                f"OneRunway_{a.id}"
            )

        # Link z_ij to runway assignments
        # z_ij = 1 if aircraft i and j on same runway
        for i in range(n):
            for j in range(i + 1, n):
                id_i = aircraft[i].id
                id_j = aircraft[j].id
                z_ij = self.variables['z'][(id_i, id_j)]

                for r in range(1, self.num_runways + 1):
                    y_ir = self.variables['y'][(id_i, r)]
                    y_jr = self.variables['y'][(id_j, r)]

                    # If both on runway r, then z_ij = 1
                    self.model += (
                        z_ij >= y_ir + y_jr - 1,
                        f"SameRunway_{id_i}_{id_j}_r{r}_lower"
                    )

                # z_ij = 0 if on different runways
                self.model += (
                    z_ij <= lpSum([self.variables['y'][(id_i, r)] *
                                  self.variables['y'][(id_j, r)]
                                  for r in range(1, self.num_runways + 1)]),
                    f"SameRunway_{id_i}_{id_j}_upper"
                )

    def solve(self, time_limit: int = 300, gap: float = 0.01,
              solver_name: str = 'PULP_CBC_CMD', verbose: bool = True) -> Optional[Solution]:
        """
        Solve the MIP model.

        Args:
            time_limit: Maximum solve time in seconds
            gap: Acceptable optimality gap (0.01 = 1%)
            solver_name: Name of solver to use
            verbose: Print solve progress

        Returns:
            Solution object if solved, None otherwise
        """
        if self.model is None:
            self.build_model()

        print(f"\nSolving with {solver_name}...")
        print(f"Time limit: {time_limit}s, Gap: {gap*100}%")

        # Configure solver
        if solver_name == 'PULP_CBC_CMD':
            solver = PULP_CBC_CMD(
                timeLimit=time_limit,
                gapRel=gap,
                msg=1 if verbose else 0
            )
        elif solver_name == 'GUROBI_CMD':
            solver = GUROBI_CMD(
                timeLimit=time_limit,
                mip=gap,
                msg=1 if verbose else 0
            )
        else:
            solver = getSolver(solver_name)

        # Solve
        import time
        start_time = time.time()
        status = self.model.solve(solver)
        solve_time = time.time() - start_time

        print(f"Solve completed in {solve_time:.2f}s")
        print(f"Status: {LpStatus[status]}")

        if status not in [LpStatusOptimal, LpStatusNotSolved]:
            if status == LpStatusInfeasible:
                print("Model is infeasible!")
            return None

        # Extract solution
        solution = self._extract_solution(solve_time, LpStatus[status])

        if solution:
            print(f"Objective value: {solution.objective_value:.2f}")

        return solution

    def _extract_solution(self, solve_time: float, status: str) -> Optional[Solution]:
        """Extract solution from solved model."""
        landing_times = {}
        runway_assignments = {}

        try:
            # Extract landing times
            for a in self.instance.aircraft:
                landing_times[a.id] = value(self.variables['x'][a.id])

            # Extract runway assignments
            if self.num_runways == 1:
                for a in self.instance.aircraft:
                    runway_assignments[a.id] = 1
            else:
                for a in self.instance.aircraft:
                    for r in range(1, self.num_runways + 1):
                        if value(self.variables['y'][(a.id, r)]) > 0.5:
                            runway_assignments[a.id] = r
                            break

            objective_value = value(self.model.objective)

            solution = Solution(
                landing_times=landing_times,
                runway_assignments=runway_assignments,
                objective_value=objective_value,
                solve_time=solve_time,
                status=status,
                gap=0.0  # TODO: Extract actual gap if available
            )

            return solution

        except Exception as e:
            print(f"Error extracting solution: {e}")
            return None


if __name__ == "__main__":
    # Test the model
    from .data_loader import DataLoader

    print("Testing AircraftLandingModel...")

    # Create sample instance
    instance = DataLoader.create_sample_instance(num_aircraft=5)

    # Build and solve model
    model = AircraftLandingModel(instance, num_runways=1)
    solution = model.solve(time_limit=60, verbose=True)

    if solution:
        print(f"\nSolution found!")
        print(f"Landing times:")
        for aid in sorted(solution.landing_times.keys()):
            print(f"  Aircraft {aid}: {solution.landing_times[aid]:.2f}")
        print(f"Total cost: {solution.objective_value:.2f}")
    else:
        print("No solution found!")
