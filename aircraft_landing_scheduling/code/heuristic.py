#!/usr/bin/env python3
"""
Greedy Heuristic for Aircraft Landing Scheduling
Based on the algorithm described in Beasley et al. (2000)
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from .data_loader import ProblemInstance, Aircraft
from .model import Solution


class GreedyHeuristic:
    """
    Greedy constructive heuristic for aircraft landing scheduling.

    Algorithm (from Beasley et al. 2000):
    1. Sort aircraft by target landing time
    2. For each aircraft in sorted order:
       a. Try assigning to each runway
       b. For each runway, find earliest feasible time respecting:
          - Time window [E_i, L_i]
          - Separation from already scheduled aircraft
       c. Calculate cost for each option
       d. Assign to runway with minimum cost
    3. Optionally improve solution by local adjustments

    Time Complexity: O(P² × R) where P = aircraft, R = runways
    """

    def __init__(self, instance: ProblemInstance, num_runways: int = 1):
        """
        Initialize the greedy heuristic.

        Args:
            instance: Problem instance data
            num_runways: Number of available runways
        """
        self.instance = instance
        self.num_runways = num_runways

        # Store solution under construction
        self.landing_times: Dict[int, float] = {}
        self.runway_assignments: Dict[int, int] = {}
        self.runway_schedules: Dict[int, List[int]] = {
            r: [] for r in range(1, num_runways + 1)
        }

    def solve(self, improve: bool = True) -> Solution:
        """
        Execute the greedy heuristic algorithm.

        Args:
            improve: Whether to apply improvement phase

        Returns:
            Solution object
        """
        import time
        start_time = time.time()

        print(f"Running greedy heuristic: {self.instance.num_aircraft} aircraft, "
              f"{self.num_runways} runway(s)")

        # Phase 1: Construct initial solution
        self._construct_solution()

        # Phase 2: Improve solution (optional)
        if improve:
            self._improve_solution()

        solve_time = time.time() - start_time

        # Calculate objective value
        objective_value = self._calculate_objective()

        solution = Solution(
            landing_times=self.landing_times.copy(),
            runway_assignments=self.runway_assignments.copy(),
            objective_value=objective_value,
            solve_time=solve_time,
            status="Heuristic"
        )

        print(f"Greedy heuristic completed in {solve_time:.3f}s")
        print(f"Objective value: {objective_value:.2f}")

        return solution

    def _construct_solution(self):
        """Phase 1: Construct initial solution using greedy approach."""

        # Sort aircraft by target time (key heuristic choice)
        sorted_aircraft = sorted(
            self.instance.aircraft,
            key=lambda a: a.target_time
        )

        # Schedule each aircraft
        for aircraft in sorted_aircraft:
            best_runway = None
            best_time = None
            best_cost = float('inf')

            # Try each runway
            for runway in range(1, self.num_runways + 1):
                # Find earliest feasible time on this runway
                feasible_time = self._find_earliest_feasible_time(
                    aircraft, runway
                )

                if feasible_time is not None:
                    # Calculate cost of landing at this time
                    cost = aircraft.calculate_cost(feasible_time)

                    if cost < best_cost:
                        best_cost = cost
                        best_time = feasible_time
                        best_runway = runway

            # Assign aircraft to best runway
            if best_runway is not None:
                self.landing_times[aircraft.id] = best_time
                self.runway_assignments[aircraft.id] = best_runway
                self.runway_schedules[best_runway].append(aircraft.id)
            else:
                # Fallback: assign to target time on runway 1 (may be infeasible)
                print(f"Warning: No feasible time found for aircraft {aircraft.id}, "
                      f"using target time")
                self.landing_times[aircraft.id] = aircraft.target_time
                self.runway_assignments[aircraft.id] = 1
                self.runway_schedules[1].append(aircraft.id)

    def _find_earliest_feasible_time(
        self,
        aircraft: Aircraft,
        runway: int
    ) -> Optional[float]:
        """
        Find the earliest feasible landing time for an aircraft on a given runway.

        GOAL: Find a landing time that:
        1. Is within the aircraft's time window [earliest, latest]
        2. Respects separation requirements with all already-scheduled aircraft
        3. Is as close as possible to the target time

        ALGORITHM:
        - Start with the target time (ideal case)
        - Check if it conflicts with any already-scheduled aircraft
        - If conflict: move the time forward until no conflicts exist
        - Keep trying until we find a valid time or run out of options

        Args:
            aircraft: The aircraft we want to schedule
            runway: The runway number (1, 2, 3, etc.)

        Returns:
            A valid landing time, or the latest time as fallback
        """
        # Start with target time (this is what the aircraft prefers)
        candidate_time = aircraft.target_time

        # Get list of aircraft IDs that are already scheduled on this runway
        already_scheduled_aircraft = self.runway_schedules[runway]

        # CASE 1: Runway is empty - easy case!
        if not already_scheduled_aircraft:
            # Just use target time if it's after the aircraft arrives
            if candidate_time >= aircraft.appearance_time:
                return candidate_time
            else:
                # Aircraft can't make it to target, land as soon as it arrives
                return aircraft.appearance_time

        # CASE 2: Runway has other aircraft - need to check separations
        # Keep trying different times until we find one that works
        while candidate_time <= aircraft.latest_time:
            time_is_valid = True  # Assume it's valid until proven otherwise

            # Check against each already-scheduled aircraft
            for scheduled_aircraft_id in already_scheduled_aircraft:
                scheduled_landing_time = self.landing_times[scheduled_aircraft_id]
                scheduled_aircraft_index = self._get_aircraft_index(scheduled_aircraft_id)
                current_aircraft_index = self._get_aircraft_index(aircraft.id)

                # SCENARIO A: The scheduled aircraft lands BEFORE our candidate time
                if scheduled_landing_time <= candidate_time:
                    # Calculate required separation time
                    required_separation = self.instance.get_separation(
                        scheduled_aircraft_index, current_aircraft_index
                    )

                    # Check if candidate time is too soon after the scheduled aircraft
                    if candidate_time < scheduled_landing_time + required_separation:
                        # TOO SOON! Move candidate time forward to respect separation
                        candidate_time = scheduled_landing_time + required_separation
                        time_is_valid = False
                        break  # Start checking all aircraft again with new time

                # SCENARIO B: The scheduled aircraft lands AFTER our candidate time
                else:
                    # Calculate required separation time (reversed order)
                    required_separation = self.instance.get_separation(
                        current_aircraft_index, scheduled_aircraft_index
                    )

                    # Check if our candidate time is too close to the scheduled aircraft
                    if scheduled_landing_time < candidate_time + required_separation:
                        # CONFLICT! Our candidate time doesn't leave enough room
                        time_is_valid = False
                        # Try landing just after this aircraft (small offset to avoid exact equality)
                        candidate_time = scheduled_landing_time + 0.01
                        break  # Start checking all aircraft again with new time

            # Did we find a valid time?
            if time_is_valid and candidate_time >= aircraft.appearance_time:
                return candidate_time

            # If not valid, the loop continues with the adjusted candidate_time
            # Safety check to avoid infinite loops
            if not time_is_valid:
                continue  # Try again with the new candidate_time
            else:
                # No conflicts but still need to search? Move forward slightly
                candidate_time += 0.1

        # If we get here, we couldn't find a perfect time within the window
        # Return latest time as fallback (solver will handle any violations)
        return aircraft.latest_time

    def _improve_solution(self):
        """
        Phase 2: Improve solution through local adjustments.

        Tries to shift aircraft landing times closer to target times
        while maintaining feasibility.
        """
        improved = True
        iterations = 0
        max_iterations = 10

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            # Try to improve each aircraft
            for aircraft in self.instance.aircraft:
                old_time = self.landing_times[aircraft.id]
                old_cost = aircraft.calculate_cost(old_time)

                # Try moving towards target time
                new_time = self._try_improve_time(aircraft)

                if new_time is not None and new_time != old_time:
                    new_cost = aircraft.calculate_cost(new_time)

                    if new_cost < old_cost:
                        # Check feasibility with other aircraft
                        if self._is_time_feasible(aircraft, new_time):
                            self.landing_times[aircraft.id] = new_time
                            improved = True

        if iterations > 0:
            print(f"  Improvement phase: {iterations} iterations")

    def _try_improve_time(self, aircraft: Aircraft) -> Optional[float]:
        """Try to find a better landing time closer to target."""
        current_time = self.landing_times[aircraft.id]
        target_time = aircraft.target_time

        # If already at target, no improvement possible
        if abs(current_time - target_time) < 0.01:
            return None

        # Try to move towards target
        if current_time < target_time:
            # Currently early, try moving later
            new_time = min(current_time + 1.0, target_time)
        else:
            # Currently late, try moving earlier
            new_time = max(current_time - 1.0, target_time)

        # Ensure within time window
        new_time = max(aircraft.appearance_time,
                      min(new_time, aircraft.latest_time))

        return new_time

    def _is_time_feasible(self, aircraft: Aircraft, candidate_time: float) -> bool:
        """
        Check if a candidate landing time is feasible.

        Args:
            aircraft: Aircraft being scheduled
            candidate_time: Proposed landing time

        Returns:
            True if feasible, False otherwise
        """
        # Check time window
        if candidate_time < aircraft.appearance_time or \
           candidate_time > aircraft.latest_time:
            return False

        # Check separation with other aircraft on same runway
        runway = self.runway_assignments[aircraft.id]
        current_idx = self._get_aircraft_index(aircraft.id)

        for scheduled_id in self.runway_schedules[runway]:
            if scheduled_id == aircraft.id:
                continue

            scheduled_time = self.landing_times[scheduled_id]
            scheduled_idx = self._get_aircraft_index(scheduled_id)

            if scheduled_time <= candidate_time:
                # Scheduled aircraft lands first
                required_sep = self.instance.get_separation(
                    scheduled_idx, current_idx
                )
                if candidate_time < scheduled_time + required_sep:
                    return False
            else:
                # Current aircraft would land first
                required_sep = self.instance.get_separation(
                    current_idx, scheduled_idx
                )
                if scheduled_time < candidate_time + required_sep:
                    return False

        return True

    def _calculate_objective(self) -> float:
        """Calculate total cost of current solution."""
        total_cost = 0.0

        for aircraft in self.instance.aircraft:
            landing_time = self.landing_times[aircraft.id]
            cost = aircraft.calculate_cost(landing_time)
            total_cost += cost

        return total_cost

    def _get_aircraft_index(self, aircraft_id: int) -> int:
        """Get 0-indexed position of aircraft in instance list."""
        for idx, aircraft in enumerate(self.instance.aircraft):
            if aircraft.id == aircraft_id:
                return idx
        return -1


class MultiStartGreedy:
    """
    Multi-start greedy heuristic that tries different sorting strategies
    and returns the best solution found.
    """

    def __init__(self, instance: ProblemInstance, num_runways: int = 1):
        self.instance = instance
        self.num_runways = num_runways

    def solve(self, num_starts: int = 5) -> Solution:
        """
        Run multiple greedy heuristics with different strategies.

        Args:
            num_starts: Number of different strategies to try

        Returns:
            Best solution found
        """
        print(f"Running multi-start greedy heuristic ({num_starts} starts)...")

        sorting_strategies = [
            ('target_time', lambda a: a.target_time),
            ('appearance_time', lambda a: a.appearance_time),
            ('time_window', lambda a: a.latest_time - a.appearance_time),
            ('total_penalty', lambda a: a.early_penalty + a.late_penalty),
            ('early_penalty', lambda a: -a.early_penalty),
        ]

        best_solution = None
        best_cost = float('inf')

        for i, (name, sort_key) in enumerate(sorting_strategies[:num_starts]):
            print(f"\n  Start {i+1}/{num_starts}: Sorting by {name}")

            # Sort aircraft using this strategy
            sorted_aircraft = sorted(self.instance.aircraft, key=sort_key)

            # Create temporary instance with sorted aircraft
            temp_instance = ProblemInstance(
                aircraft=sorted_aircraft,
                separation_matrix=self.instance.separation_matrix,
                freeze_time=self.instance.freeze_time
            )

            # Run greedy heuristic
            heuristic = GreedyHeuristic(temp_instance, self.num_runways)
            solution = heuristic.solve(improve=True)

            if solution.objective_value < best_cost:
                best_cost = solution.objective_value
                best_solution = solution
                print(f"    New best: {best_cost:.2f}")

        print(f"\nBest solution: {best_cost:.2f}")
        return best_solution
