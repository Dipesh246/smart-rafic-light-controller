from django.contrib import admin
from .models import Intersection, TrafficData, SignalCycle

admin.site.register(Intersection)
admin.site.register(TrafficData)
admin.site.register(SignalCycle)
