import json
from django.http import JsonResponse
from django.utils.formats import date_format
from django.utils.safestring import mark_safe
from django.shortcuts import render
from traffic.algorithm import DynamicSignalController, QueuePredictor
from traffic.models import SignalCycle


def dashboard_view(request):
    controller = DynamicSignalController()
    predictor = QueuePredictor()

    results = controller.run_for_all()
    predictions = predictor.run_for_all()

    # Serialize predictions to JSON
    predictions_json = mark_safe(json.dumps(predictions))

    latest_cycles = SignalCycle.objects.order_by("-cycle_timestamp")[:20]

    return render(
        request,
        "dashboard.html",
        {
            "results": results,
            "predictions_json": predictions_json,
            "cycles": latest_cycles,
        },
    )


def dashboard_data_api(request):
    controller = DynamicSignalController()
    predictor = QueuePredictor()

    results = controller.run_for_all()
    predictions = predictor.run_for_all()
    latest_cycles = SignalCycle.objects.order_by("-cycle_timestamp")[:20]

    data = {
        "results": results,
        "predictions": predictions,
        "cycles": [
            {
                "intersection": c.intersection.name,
                "direction": c.direction,
                "green_time": round(c.green_time, 2),
                "timestamp": date_format(c.cycle_timestamp, "N j, Y, P"), 
            }
            for c in latest_cycles
        ],
    }
    return JsonResponse(data)
