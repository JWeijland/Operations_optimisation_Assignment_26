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

        Big-M is a large number used in optimization to model "either-or" constraints.
        In our case, it's used to enforce that either:
        - Aircraft i lands before aircraft j, OR
        - Aircraft j lands before aircraft i

        The Big-M must be larger than any possible time difference between aircraft,
        so we calculate it as: maximum landing time + maximum separation time + safety buffer
        """
        # Find the latest possible landing time of any aircraft
        max_time = max(aircraft.latest_time for aircraft in self.instance.aircraft)

        # Find the longest separation requirement between any two aircraft
        max_separation = np.max(self.instance.separation_matrix)

        # Big-M = max time + max separation + safety buffer of 100
        big_m_value = max_time + max_separation + 100

        return big_m_value

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
        """
        Create all decision variables for the optimization model.

        We create 4 types of variables:
        1. Landing times (when each aircraft lands)
        2. Early deviations (how much before target time)
        3. Late deviations (how much after target time)
        4. Ordering variables (which aircraft lands first)
        5. Runway assignments (if multiple runways)
        """
        aircraft_list = self.instance.aircraft
        number_of_aircraft = len(aircraft_list)

        # ===== 1. LANDING TIME VARIABLES =====
        # For each aircraft, we have a continuous variable representing when it lands
        # Bounds: must be between earliest possible time and latest possible time
        self.variables['x'] = {}  # 'x' stores the actual landing time
        for aircraft in aircraft_list:
            landing_time_var = LpVariable(
                f"landing_time_{aircraft.id}",
                lowBound=aircraft.appearance_time,  # Can't land before this
                upBound=aircraft.latest_time,       # Can't land after this
                cat='Continuous'
            )
            self.variables['x'][aircraft.id] = landing_time_var

        # ===== 2. EARLY DEVIATION VARIABLES =====
        # How many time units BEFORE the target time the aircraft lands
        # Example: if target is 10 and aircraft lands at 7, early deviation = 3
        self.variables['alpha'] = {}  # 'alpha' = early deviation
        for aircraft in aircraft_list:
            max_early_deviation = aircraft.target_time - aircraft.appearance_time
            early_deviation_var = LpVariable(
                f"early_deviation_{aircraft.id}",
                lowBound=0,                      # Can't be negative
                upBound=max_early_deviation,     # Maximum = target - earliest
                cat='Continuous'
            )
            self.variables['alpha'][aircraft.id] = early_deviation_var

        # ===== 3. LATE DEVIATION VARIABLES =====
        # How many time units AFTER the target time the aircraft lands
        # Example: if target is 10 and aircraft lands at 13, late deviation = 3
        self.variables['beta'] = {}  # 'beta' = late deviation
        for aircraft in aircraft_list:
            max_late_deviation = aircraft.latest_time - aircraft.target_time
            late_deviation_var = LpVariable(
                f"late_deviation_{aircraft.id}",
                lowBound=0,                     # Can't be negative
                upBound=max_late_deviation,     # Maximum = latest - target
                cat='Continuous'
            )
            self.variables['beta'][aircraft.id] = late_deviation_var

        # ===== 4. ORDERING VARIABLES (BINARY) =====
        # For each pair of aircraft (i, j), we have a binary variable:
        # delta[i,j] = 1 if aircraft i lands BEFORE aircraft j
        # delta[i,j] = 0 if aircraft j lands BEFORE aircraft i
        self.variables['delta'] = {}
        for i in range(number_of_aircraft):
            for j in range(i + 1, number_of_aircraft):  # Only create for i < j (avoid duplicates)
                aircraft_i_id = aircraft_list[i].id
                aircraft_j_id = aircraft_list[j].id

                ordering_var = LpVariable(
                    f"ordering_{aircraft_i_id}_before_{aircraft_j_id}",
                    cat='Binary'  # Can only be 0 or 1
                )
                self.variables['delta'][(aircraft_i_id, aircraft_j_id)] = ordering_var

        # ===== 5. RUNWAY ASSIGNMENT VARIABLES (IF MULTIPLE RUNWAYS) =====
        # For each aircraft and each runway, we have a binary variable:
        # y[aircraft, runway] = 1 if aircraft is assigned to that runway
        # y[aircraft, runway] = 0 otherwise
        if self.num_runways > 1:
            self.variables['y'] = {}
            for aircraft in aircraft_list:
                for runway_number in range(1, self.num_runways + 1):
                    runway_assignment_var = LpVariable(
                        f"aircraft_{aircraft.id}_on_runway_{runway_number}",
                        cat='Binary'
                    )
                    self.variables['y'][(aircraft.id, runway_number)] = runway_assignment_var

            # ===== 6. SAME RUNWAY INDICATOR VARIABLES (FOR SEPARATION CONSTRAINTS) =====
            # For each pair of aircraft (i, j), we have a binary variable:
            # z[i,j] = 1 if both aircraft i and j are on the SAME runway
            # z[i,j] = 0 if they are on DIFFERENT runways
            # This is needed because separation is only required on the same runway
            self.variables['z'] = {}
            for i in range(number_of_aircraft):
                for j in range(i + 1, number_of_aircraft):
                    aircraft_i_id = aircraft_list[i].id
                    aircraft_j_id = aircraft_list[j].id

                    same_runway_var = LpVariable(
                        f"same_runway_{aircraft_i_id}_{aircraft_j_id}",
                        cat='Binary'
                    )
                    self.variables['z'][(aircraft_i_id, aircraft_j_id)] = same_runway_var

    def _set_objective(self):
        """
        Set the objective function: minimize total cost.

        The total cost is calculated as:
        - For each aircraft that lands EARLY: early_penalty × early_deviation
        - For each aircraft that lands LATE: late_penalty × late_deviation
        - Sum all these costs across all aircraft

        Example: If aircraft 1 lands 3 minutes early with penalty €50/min,
                 the cost is 3 × 50 = €150
        """
        aircraft_list = self.instance.aircraft

        # Calculate total cost by summing penalties for all aircraft
        total_cost = lpSum([
            # Cost if aircraft lands early
            aircraft.early_penalty * self.variables['alpha'][aircraft.id] +
            # Cost if aircraft lands late
            aircraft.late_penalty * self.variables['beta'][aircraft.id]
            for aircraft in aircraft_list
        ])

        # Set this as the objective to minimize
        self.model += total_cost

    def _add_time_window_constraints(self):
        """
        Add time window constraints: each aircraft must land within its time window.

        For each aircraft:
        - Landing time ≥ Earliest time (can't land before it arrives)
        - Landing time ≤ Latest time (can't land after deadline)

        Note: These constraints are already enforced by variable bounds,
        but we add them explicitly for clarity.
        """
        for aircraft in self.instance.aircraft:
            # Constraint: landing time must be at or after earliest time
            self.model += (
                self.variables['x'][aircraft.id] >= aircraft.appearance_time,
                f"Earliest_time_bound_aircraft_{aircraft.id}"
            )

            # Constraint: landing time must be at or before latest time
            self.model += (
                self.variables['x'][aircraft.id] <= aircraft.latest_time,
                f"Latest_time_bound_aircraft_{aircraft.id}"
            )

    def _add_target_deviation_constraints(self):
        """
        Add target time deviation constraints: link landing time to early/late deviations.

        This constraint ensures that:
        landing_time = target_time - early_deviation + late_deviation

        Examples:
        - If landing at target time (10): 10 = 10 - 0 + 0 ✓
        - If landing 3 minutes early (7): 7 = 10 - 3 + 0 ✓
        - If landing 2 minutes late (12): 12 = 10 - 0 + 2 ✓

        Only one of early_deviation or late_deviation will be non-zero
        (enforced automatically by the optimization).
        """
        for aircraft in self.instance.aircraft:
            # Constraint: landing_time = target_time - early_deviation + late_deviation
            self.model += (
                self.variables['x'][aircraft.id] ==
                aircraft.target_time
                - self.variables['alpha'][aircraft.id]  # Subtract early deviation
                + self.variables['beta'][aircraft.id],   # Add late deviation
                f"Target_deviation_constraint_aircraft_{aircraft.id}"
            )

    def _add_separation_constraints(self):
        """
        Add separation constraints between aircraft.

        SEPARATION REQUIREMENT:
        When two aircraft land on the same runway, there must be enough time between
        their landings for safety (wake turbulence, runway clearing, etc.).

        LOGIC:
        For each pair of aircraft (i, j), we need to enforce:
        - If i lands BEFORE j: landing_time_j ≥ landing_time_i + separation_time_i_to_j
        - If j lands BEFORE i: landing_time_i ≥ landing_time_j + separation_time_j_to_i

        We use the "ordering variable" delta to decide which one applies:
        - delta[i,j] = 1 means i lands BEFORE j
        - delta[i,j] = 0 means j lands BEFORE i

        The Big-M technique is used to activate/deactivate constraints based on delta.
        """
        aircraft_list = self.instance.aircraft
        number_of_aircraft = len(aircraft_list)

        # Loop through all pairs of aircraft
        for i in range(number_of_aircraft):
            for j in range(i + 1, number_of_aircraft):  # Only i < j to avoid duplicates
                # Get aircraft IDs and their separation requirements
                aircraft_i_id = aircraft_list[i].id
                aircraft_j_id = aircraft_list[j].id
                separation_i_before_j = self.instance.get_separation(i, j)  # Time needed if i lands before j
                separation_j_before_i = self.instance.get_separation(j, i)  # Time needed if j lands before i

                # Get the relevant variables
                ordering_variable = self.variables['delta'][(aircraft_i_id, aircraft_j_id)]
                landing_time_i = self.variables['x'][aircraft_i_id]
                landing_time_j = self.variables['x'][aircraft_j_id]

                if self.num_runways == 1:
                    # ===== SINGLE RUNWAY CASE =====
                    # Both aircraft use the same runway, so separation always applies

                    # CONSTRAINT 1: If ordering_variable = 1 (i lands before j)
                    # Then: landing_time_j ≥ landing_time_i + separation_i_before_j
                    # Using Big-M: landing_time_j ≥ landing_time_i + separation - Big_M * (1 - ordering_variable)
                    # When ordering_variable = 1: landing_time_j ≥ landing_time_i + separation (enforced!)
                    # When ordering_variable = 0: landing_time_j ≥ landing_time_i + separation - Big_M (inactive, always true)
                    self.model += (
                        landing_time_j >= landing_time_i + separation_i_before_j - self.big_m * (1 - ordering_variable),
                        f"Separation_if_aircraft_{aircraft_i_id}_before_{aircraft_j_id}"
                    )

                    # CONSTRAINT 2: If ordering_variable = 0 (j lands before i)
                    # Then: landing_time_i ≥ landing_time_j + separation_j_before_i
                    # Using Big-M: landing_time_i ≥ landing_time_j + separation - Big_M * ordering_variable
                    # When ordering_variable = 0: landing_time_i ≥ landing_time_j + separation (enforced!)
                    # When ordering_variable = 1: landing_time_i ≥ landing_time_j + separation - Big_M (inactive, always true)
                    self.model += (
                        landing_time_i >= landing_time_j + separation_j_before_i - self.big_m * ordering_variable,
                        f"Separation_if_aircraft_{aircraft_j_id}_before_{aircraft_i_id}"
                    )

                else:
                    # ===== MULTIPLE RUNWAYS CASE =====
                    # Separation only matters if BOTH aircraft are on the SAME runway
                    # We use the "same runway" variable z to check this

                    same_runway_variable = self.variables['z'][(aircraft_i_id, aircraft_j_id)]

                    # CONSTRAINT 1: If same_runway = 1 AND ordering_variable = 1
                    # Then: landing_time_j ≥ landing_time_i + separation_i_before_j
                    # Using Big-M: constraint is active only when BOTH are 1
                    # Formula: landing_time_j ≥ landing_time_i + separation - Big_M * (2 - same_runway - ordering_variable)
                    # When both = 1: (2 - 1 - 1) = 0, so constraint is fully active
                    # When either = 0: multiplier ≥ 1, so Big_M makes constraint inactive
                    self.model += (
                        landing_time_j >= landing_time_i + separation_i_before_j - self.big_m * (2 - same_runway_variable - ordering_variable),
                        f"Separation_if_aircraft_{aircraft_i_id}_before_{aircraft_j_id}"
                    )

                    # CONSTRAINT 2: If same_runway = 1 AND ordering_variable = 0
                    # Then: landing_time_i ≥ landing_time_j + separation_j_before_i
                    # Formula: landing_time_i ≥ landing_time_j + separation - Big_M * (1 + ordering_variable - same_runway)
                    # When same_runway = 1 and ordering_variable = 0: (1 + 0 - 1) = 0, constraint active
                    # Otherwise: multiplier makes constraint inactive
                    self.model += (
                        landing_time_i >= landing_time_j + separation_j_before_i - self.big_m * (1 + ordering_variable - same_runway_variable),
                        f"Separation_if_aircraft_{aircraft_j_id}_before_{aircraft_i_id}"
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

                # Linearization of: z_ij = OR_r(y_ir AND y_jr)
                # Lower bound: If both on runway r, then z_ij must be 1
                for r in range(1, self.num_runways + 1):
                    y_ir = self.variables['y'][(id_i, r)]
                    y_jr = self.variables['y'][(id_j, r)]

                    self.model += (
                        z_ij >= y_ir + y_jr - 1,
                        f"SameRunway_{id_i}_{id_j}_r{r}_lower"
                    )

                # Upper bound: z_ij can be 1 only if sum over all runways of common assignments >= 1
                # Simpler: z_ij <= sum_r of min(y_ir, y_jr)
                # We approximate with: z_ij * num_runways <= sum_r (y_ir + y_jr)
                # Even simpler: just ensure z_ij = 0 if no common runway exists
                # The lower bound constraints already handle this correctly!

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
