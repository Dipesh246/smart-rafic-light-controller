from celery import shared_task
from django.utils import timezone
import random
from collections import defaultdict
from .models import Intersection, TrafficData, SignalCycle
from .algorithm import QueuePredictor, SignalAllocator


DIRECTIONS = ["N", "E", "S", "W"]
LANES = ["straight", "left", "right"]


@shared_task()
def generate_traffic_data():
    intersections = Intersection.objects.all()
    results = defaultdict(dict)
    for inter in intersections:
        for direction in DIRECTIONS:
            for lane in LANES:
                TrafficData.objects.create(
                    intersection=inter,
                    direction=direction,
                    lane_type=lane,
                    vehicle_count=random.randint(5, 40)
                )
    print("Traffic data generated")
    return "Traffic data generated"


@shared_task()
def run_signal_algorithm():
    controller = DynamicSignalController()
    results = controller.run_for_all()
    return results
