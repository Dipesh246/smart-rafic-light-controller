from celery import shared_task
import random
from datetime import datetime
from .models import Intersection, TrafficData, SignalCycle
from .algorithm import QueuePredictor, DynamicSignalController
from .constants import DIRECTIONS, LANES


@shared_task()
def generate_traffic_data():
    """
    Generate realistic lane-based traffic data for each intersection and direction.
    Each direction can have multiple lanes with varying vehicle counts.
    """
    intersections = Intersection.objects.all()

    for inter in intersections:
        for direction in DIRECTIONS:
            num_lanes = random.randint(2, 4)  # each direction has 2–4 lanes
            lane_vehicle_counts = []

            for lane in LANES:
                inflow = random.randint(0, 8)  # vehicles arriving in this time window

                # Get latest data for this lane
                last = (
                    TrafficData.objects.filter(
                        intersection=inter, direction=direction, lane_type=lane
                    )
                    .order_by("-timestamp")
                    .first()
                )
                current_count = last.vehicle_count if last else 0
                new_count = max(current_count + inflow - random.randint(0, 3), 0)
                # simulate slightly fluctuating per-lane load
                TrafficData.objects.create(
                    intersection=inter,
                    direction=direction,
                    lane_type=lane,
                    vehicle_count=new_count,
                    timestamp=datetime.now(),
                )

            # store the average for debugging or logging

    print("Lane-wise traffic updated.")
    return "Lane-based traffic data generated"


@shared_task()
def run_signal_algorithm():
    controller = DynamicSignalController()
    results = controller.run_for_all()
    return results

@shared_task()
def retrain_queue_model():
    predictor = QueuePredictor()
    predictor.ml.train_model()
    return "Model retrained successfully"