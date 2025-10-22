import random
from celery import shared_task
from django.utils import timezone
from .models import TrafficData, Intersection
from .algorithm import DynamicSignalController


@shared_task
def generate_traffic_data():
    directions = ["N", "E", "S", "W"]
    intersections = Intersection.objects.all()

    for inter in intersections:
        for d in directions:
            count = random.randint(0, 40)  # random load simulation
            TrafficData.objects.create(
                intersection=inter,
                direction=d,
                vehicle_count=count,
                timestamp=timezone.now(),
            )
    return "Traffic data generated"


@shared_task
def run_signal_algorithm():
    controller = DynamicSignalController()
    results = controller.run_for_all()
    return results
