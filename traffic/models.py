from django.db import models
from django.contrib.postgres.fields import JSONField

LANE_CHOICES = [
    ("straight", "Straight"),
    ("left", "Left Turn"),
    ("right", "Right Turn"),
]


class Intersection(models.Model):
    name = models.CharField(max_length=100, unique=True)
    location = models.CharField(max_length=200, blank=True, null=True)

    def __str__(self):
        return self.name


class TrafficData(models.Model):
    intersection = models.ForeignKey(Intersection, on_delete=models.CASCADE)
    direction = models.CharField(
        max_length=10,
        choices=[
            ("N", "North"),
            ("E", "East"),
            ("S", "South"),
            ("W", "West"),
        ],
    )
    vehicle_count = models.PositiveIntegerField(default=0)
    lane_type = models.CharField(max_length=10, choices=LANE_CHOICES)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.intersection.name} | {self.direction} | {self.lane_type}: {self.vehicle_count}"


class SignalCycle(models.Model):
    intersection = models.ForeignKey(Intersection, on_delete=models.CASCADE)
    direction = models.CharField(max_length=10)
    green_time = models.FloatField(help_text="Green signal time in seconds")
    cycle_timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.intersection.name} - {self.direction} ({self.green_time}s)"
