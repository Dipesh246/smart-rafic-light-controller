from traffic.models import Intersection, TrafficData, SignalCycle
from django.utils import timezone
from datetime import datetime
import numpy as np
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
    New QueuePredictor: prefer ML predictions, fall back to EMA if ML not available.
    """

    model_path = "ml_models/queue_predictor.pkl"

    def __init__(self, alpha=0.3):
        self.ml = MLQueuePredictor()
        self.ema = QueuePredictorEMA(alpha=alpha)
        self.direction_map = {"N": 0, "E": 1, "S": 2, "W": 3}
        self.lane_map = {"straight": 0, "left": 1, "right": 2}

    def predict_for_intersection(self, intersection, limit=10):
        if self.ml.is_available():
            preds = self.ml.predict_for_intersection(intersection, limit=limit)
            # sanity: if ML returned empty, fallback to EMA
            if preds:
                return preds
        # fallback
        return self.ema.predict_for_intersection(intersection, limit=limit)

    def train_model(self):
        data = TrafficData.objects.all().order_by("-timestamp")[:5000]
        if not data.exists():
            print("⚠️ No data available for training.")
            return

        X, y = [], []

        for d in data:
            # Feature 1: vehicle count
            vehicle_count = d.vehicle_count

            # Feature 2: encoded direction (N=0, E=1, S=2, W=3)
            dir_code = self.direction_map.get(d.direction, -1)

            # Feature 3: encoded lane type (straight=0, left=1, right=2)
            lane_code = self.lane_map.get(d.lane_type, -1)

            # Feature 4: time of day as hour (0–23)
            hour = d.timestamp.hour

            # Feature 5: weekday/weekend binary
            is_weekend = 1 if d.timestamp.weekday() >= 5 else 0

            # Target variable (approximate "queue length")
            # You can later replace this with measured queue length if available
            estimated_queue = vehicle_count * np.random.uniform(0.7, 1.3)

            X.append([vehicle_count, dir_code, lane_code, hour, is_weekend])
            y.append(estimated_queue)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(model, self.model_path)

        print(f"✅ Model trained and saved at {self.model_path}")
        print(f"📊 Mean Absolute Error (validation): {mae:.2f}")

    def run_for_all(self):
        if not os.path.exists(self.model_path):
            print("⚠️ Model not found. Please train it first.")
            return {}

        model = joblib.load(self.model_path)
        predictions = {}

        # Take recent data to simulate real-time prediction
        recent_data = TrafficData.objects.order_by("-timestamp")[:200]

        for record in recent_data:
            features = np.array(
                [
                    record.vehicle_count,
                    self.direction_map.get(record.direction, -1),
                    self.lane_map.get(record.lane_type, -1),
                    record.timestamp.hour,
                    1 if record.timestamp.weekday() >= 5 else 0,
                ]
            ).reshape(1, -1)

            predicted_queue = model.predict(features)[0]

            intersection = record.intersection.name
            key = f"{record.direction}-{record.lane_type}"

            predictions.setdefault(intersection, {})[key] = round(predicted_queue, 2)

        return predictions
