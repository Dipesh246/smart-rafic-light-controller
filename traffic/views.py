import json
import random, math
from django.http import JsonResponse
from django.utils.formats import date_format
from django.utils.safestring import mark_safe
from django.shortcuts import render
from django.utils.timezone import now
from django.db.models import Avg, Sum
from traffic.algorithm import DynamicSignalController, QueuePredictor
from traffic.models import SignalCycle, MLTrainingLog, Intersection, TrafficData


def dashboard_view(request):
    controller = DynamicSignalController()
    predictor = QueuePredictor()

    results = controller.run_for_all()
    predictions = predictor.run_for_all()

    # Serialize predictions safely
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

    ml_preds = predictor.ml.run_for_all() if predictor.ml.is_available() else {}
    ema_preds = predictor.ema.run_for_all()

    predictions = predictor.run_for_all()

    # Optional: simulate live fluctuations
    for inter, lanes in predictions.items():
        for lane, count in lanes.items():
            delta = random.randint(-3, 3)
            predictions[inter][lane] = max(count + delta, 0)

    latest_cycles = SignalCycle.objects.order_by("-cycle_timestamp")[:20]

    data = {
        "results": results,
        "predictions": predictions,
        "ml_predictions": ml_preds,
        "ema_predictions": ema_preds,
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


def training_metrics_api(request):
    logs = MLTrainingLog.objects.filter(status="success").order_by("-started_at")[:10][
        ::-1
    ]
    data = {
        "timestamps": [log.started_at.strftime("%H:%M") for log in logs],
        "mae": [round(log.mae, 3) if log.mae is not None else None for log in logs],
        "r2": [round(log.r2, 3) if log.r2 is not None else None for log in logs],
    }
    return JsonResponse(data)


def signal_state_api(request):
    data = {}
    current_time = now()

    for inter in Intersection.objects.all():
        inter_data = {}
        latest_cycle = (
            SignalCycle.objects.filter(intersection=inter)
            .order_by("-cycle_timestamp")
            .first()
        )

        if not latest_cycle:
            continue

        for direction in ["N", "E", "S", "W"]:
            if direction == latest_cycle.direction:
                elapsed = (current_time - latest_cycle.cycle_timestamp).total_seconds()
                remaining = max(round(latest_cycle.green_time - elapsed, 1), 0)
                state = "green" if remaining > 0 else "red"
            else:
                state = "red"
                remaining = 0

            inter_data[direction] = {
                "state": state,
                "remaining_time": remaining,
            }

        data[inter.name] = inter_data

    return JsonResponse(data)


def flow_stats_api(request):
    data = {}
    current_time = now()

    for inter in Intersection.objects.all():
        stats = (
            TrafficData.objects.filter(intersection=inter)
            .order_by("-timestamp")[:30]
            .aggregate(
                avg_queue=Avg("vehicle_count"),
                total_passed=Sum("vehicle_count"),
            )
        )

        avg_queue = round(stats["avg_queue"] or 0, 2)
        total_passed = int(stats["total_passed"] or 0)

        data[inter.name] = {
            "vehicles_passed": total_passed,
            "avg_queue_length": avg_queue,
            "timestamp": current_time.isoformat(),
        }

    return JsonResponse(data)


def historical_logs_api(request):
    logs = SignalCycle.objects.select_related("intersection").order_by(
        "-cycle_timestamp"
    )[:100]

    data = [
        {
            "intersection": log.intersection.name,
            "direction": log.direction,
            "green_time": log.green_time,
            "timestamp": log.cycle_timestamp.isoformat(),
            "vehicles_passed": getattr(log, "vehicles_passed", random.randint(20, 100)),
        }
        for log in logs
    ]

    return JsonResponse(data, safe=False)


def synchronization_status_api(request):

    # ---- CONFIG - tune this to match real geography ----
    # groups: ordered list along the corridor (upstream→downstream)
    groups = {
        "Group A": ["Jadibuti", "Koteshwor", "Tinkune", "Baneshwor"],
        "Group B": ["Gwarko", "Koteshwor"],  # Gwarko -> Koteshwor link
    }

    # pairwise distances in meters between connected nodes (bidirectional)
    # NOTE: replace with measured distances or geodesic calculations in production
    distances_m = {
        ("Jadibuti", "Koteshwor"): 300,
        ("Koteshwor", "Tinkune"): 600,
        ("Tinkune", "Baneshwor"): 800,
        ("Gwarko", "Koteshwor"): 400,
    }

    # average assumed progression speed (km/h) for the platoon -> convert to m/s
    avg_speed_kmph = 30.0
    speed_m_s = avg_speed_kmph * 1000.0 / 3600.0

    output_groups = []
    for gname, nodes in groups.items():
        # pick master: prefer Koteshwor if present, else middle node
        master = "Koteshwor" if "Koteshwor" in nodes else nodes[len(nodes) // 2]

        # build cumulative distances from master
        # find index of master
        try:
            idx_master = nodes.index(master)
        except ValueError:
            idx_master = 0
            master = nodes[0]

        # compute distance from master for each node (positive = downstream of master, negative = upstream)
        cumulative = {}
        # handle nodes to the right (indices > master)
        cum = 0.0
        for i in range(idx_master, len(nodes) - 1):
            a = nodes[i]
            b = nodes[i + 1]
            d = distances_m.get((a, b)) or distances_m.get((b, a)) or 0
            cum += d
            cumulative[nodes[i + 1]] = cum
        # handle nodes to the left (indices < master)
        cum = 0.0
        for i in range(idx_master, 0, -1):
            a = nodes[i]
            b = nodes[i - 1]
            d = distances_m.get((a, b)) or distances_m.get((b, a)) or 0
            cum += d
            cumulative[nodes[i - 1]] = -cum
        # master distance 0
        cumulative[master] = 0.0

        rows = []
        for node in nodes:
            dist_m = float(abs(cumulative.get(node, 0.0)))
            travel_time_s = None
            offset_s = None
            if speed_m_s > 0:
                travel_time_s = round(dist_m / speed_m_s, 1)
                # offset is time difference from master; negative => arrives earlier than master
                offset_s = int(round(cumulative.get(node, 0.0) / speed_m_s))
            else:
                travel_time_s = None
                offset_s = None

            # best-effort: if the intersection exists in DB, show it; else still return config row
            exists = Intersection.objects.filter(name__iexact=node).exists()
            rows.append(
                {
                    "intersection": node,
                    "exists": exists,
                    "distance_m": int(dist_m),
                    "travel_time_s": travel_time_s,
                    "offset_s": offset_s,
                }
            )

        output_groups.append(
            {
                "name": gname,
                "master": master,
                "rows": rows,
            }
        )

    return JsonResponse({"groups": output_groups, "generated_at": now().isoformat()})
