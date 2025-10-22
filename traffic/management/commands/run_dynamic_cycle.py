from django.core.management.base import BaseCommand
from traffic.algorithm import DynamicSignalController

class Command(BaseCommand):
    help = "Run the dynamic signal timing algorithm for all intersections"

    def handle(self, *args, **options):
        controller = DynamicSignalController(cycle_time=60, min_green=5)
        results = controller.run_for_all()
        for inter, alloc in results.items():
            self.stdout.write(self.style.SUCCESS(f"{inter}: {alloc}"))
