from django.shortcuts import render
from .models import TrafficData, SignalCycle

def dashboard_view(request):
    latest_data = TrafficData.objects.all()[:20]
    return render(request, 'dashboard.html', {'traffic_data': latest_data})
