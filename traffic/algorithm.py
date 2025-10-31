from traffic.models import Intersection, TrafficData, SignalCycle
from django.utils import timezone
from collections import defaultdict


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
        directions = ["N", "E", "S", "W"]
        recent_data = {
            d: TrafficData.objects.filter(intersection=intersection, direction=d)
            .order_by("-timestamp")
            .first()
            for d in directions
        }

        # Replace None with zero counts
        vehicle_counts = {
            d: (recent_data[d].vehicle_count if recent_data[d] else 0)
            for d in directions
        }
        total = sum(vehicle_counts.values())

        if total == 0:
            # No traffic — equal minimal timing
            allocation = {d: self.cycle_time / 4 for d in directions}
        else:
            allocation = {
                d: max((count / total) * self.cycle_time, self.min_green)
                for d, count in vehicle_counts.items()
            }

        # Normalize if total exceeds cycle time (after min_green adjustments)
        total_alloc = sum(allocation.values())
        if total_alloc > self.cycle_time:
            factor = self.cycle_time / total_alloc
            allocation = {d: round(t * factor, 2) for d, t in allocation.items()}

        # Store results
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
        for intersection in Intersection.objects.all():
            results[intersection.name] = self.compute_for_intersection(intersection)
        return results


class QueuePredictor:
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
                    intersection=intersection,
                    direction=d,
                    lane_type=lane
                ).order_by("-timestamp")[:limit]

                if not history:
                    predictions[f"{d}-{lane}"] = 0
                    continue

                values = [h.vehicle_count for h in reversed(history)]  # chronological order
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
