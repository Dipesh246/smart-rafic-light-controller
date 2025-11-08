from traffic.models import Intersection, TrafficData, SignalCycle
from django.utils import timezone
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from .constants import DIRECTIONS, LANES, LANE_WEIGHTS, LANE_FLOW_RATES
from .ml_predictor import MLQueuePredictor
import joblib, os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


class DynamicSignalController:
    """
    Weighted Round Robin-based dynamic traffic signal timing.
    """

    def __init__(self, cycle_time: int = 60, min_green: int = 5):
        """
        :param cycle_time: Total duration of one complete cycle (sec)
        :param min_green: Minimum green duration for any direction (sec)
        """
        self.cycle_time = cycle_time
        self.min_green = min_green

    def compute_for_intersection(self, intersection: Intersection):
        """
        Compute and store green times for all directions at one intersection.
        """
        direction_loads = defaultdict(float)

        for direction in DIRECTIONS:
            lane_data = TrafficData.objects.filter(
                intersection=intersection,
                direction=direction,
            ).order_by("-timestamp")[:3]

            # Aggregate by lane
            lane_counts = defaultdict(int)
            for record in lane_data:
                lane_counts[record.lane_type] += record.vehicle_count

            # Weighted load per direction
            total_weighted = sum(
                lane_counts.get(l, 0) * LANE_WEIGHTS.get(l, 1) for l in LANES
            )
            direction_loads[direction] = total_weighted

        total = sum(direction_loads.values())

        if total == 0:
            allocation = {d: round(self.cycle_time / 4, 2) for d in DIRECTIONS}
        else:
            allocation = {
                d: max(round((count / total) * self.cycle_time, 2), self.min_green)
                for d, count in direction_loads.items()
            }

        # Normalize to ensure total doesn’t exceed cycle time
        total_alloc = sum(allocation.values())
        if total_alloc > self.cycle_time:
            factor = self.cycle_time / total_alloc
            allocation = {d: round(v * factor, 2) for d, v in allocation.items()}

        # Log signal cycles (still direction-level)
        for direction, green_time in allocation.items():
            SignalCycle.objects.create(
                intersection=intersection,
                direction=direction,
                green_time=green_time,
                cycle_timestamp=timezone.now(),
            )

        return allocation

    def run_for_all(self):
        """
        Run the dynamic algorithm for all intersections.
        """
        results = {}
        for inter in Intersection.objects.all():
            results[inter.name] = self.compute_for_intersection(inter)
        return results


class QueuePredictorEMA:
    """
    Predicts upcoming traffic load per direction using Exponential Moving Average (EMA).
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha

    def predict_for_intersection(self, intersection: Intersection, limit: int = 10):
        """
        Predict queue size per direction using EMA based on recent data points.
        """
        directions = ["N", "E", "S", "W"]
        lanes = ["straight", "left", "right"]
        predictions = {}

        for d in directions:
            for lane in lanes:
                history = TrafficData.objects.filter(
                    intersection=intersection, direction=d, lane_type=lane
                ).order_by("-timestamp")[:limit]

                if not history:
                    predictions[f"{d}-{lane}"] = 0
                    continue

                values = [
                    h.vehicle_count for h in reversed(history)
                ]  # chronological order
                ema = values[0]
                for v in values[1:]:
                    ema = self.alpha * v + (1 - self.alpha) * ema

                predictions[f"{d}-{lane}"] = round(ema, 2)

        return predictions

    def run_for_all(self):
        results = defaultdict(dict)
        for intersection in Intersection.objects.all():
            results[intersection.name] = self.predict_for_intersection(intersection)
        return results


class QueuePredictor:
    """
    Wrapper combining ML and EMA predictors.
    Prefers ML; falls back to EMA if unavailable.
    """

    def __init__(self, alpha=0.3):
        self.ml = MLQueuePredictor()
        self.ema = QueuePredictorEMA(alpha=alpha)

    def run_for_all(self):
        if self.ml.is_available():
            return self.ml.run_for_all()
        return self.ema.run_for_all()
