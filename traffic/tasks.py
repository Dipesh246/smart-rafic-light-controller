from celery import shared_task
import os, sys, subprocess, traceback, time, random, pytz
from datetime import datetime
from django.utils import timezone
from .models import Intersection, TrafficData, SignalCycle, MLTrainingLog
from .algorithm import QueuePredictor, DynamicSignalController
from .constants import DIRECTIONS, LANES
from traffic.utils.ml_lock import ml_training_lock


def detect_mode():
    """
    Detects mode based on REAL TIME.
    - Peak hours only on weekdays (Sunday–Friday)
    - 09–12 and 17–20
    """
    local_tz = pytz.timezone("Asia/Kathmandu")
    now = timezone.now().astimezone(local_tz)

    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour

    if weekday != 6 and (9 <= hour <= 12 or 17 <= hour <= 20):
        return "peak"
    return "normal"


@shared_task()
def generate_traffic_data():
    """
    Generate realistic, bounded lane-based traffic data for each intersection.
    Keeps vehicle counts within [0, 50] to prevent unrealistic growth.
    """

    mode = detect_mode()
    print(f"🚦 Traffic Generation Mode: {mode}")

    intersections = Intersection.objects.all()
    timestamp = datetime.now()

    for inter in intersections:
        for direction in DIRECTIONS:
            for lane in LANES:
                # Simulate inflow and outflow
                if mode == "peak":
                    inflow = random.randint(4, 12)
                    outflow = random.randint(0, 5)
                else:
                    inflow = random.randint(0, 6)
                    outflow = random.randint(0, 4)

                # Retrieve latest data for this direction-lane
                last = (
                    TrafficData.objects.filter(
                        intersection=inter,
                        direction=direction,
                        lane_type=lane,
                        mode=mode,
                    )
                    .order_by("-timestamp")
                    .first()
                )

                current_count = last.vehicle_count if last else random.randint(0, 10)
                new_count = current_count + inflow - outflow

                # ✅ Clamp to realistic range (0–50)
                max_cap = 100 if mode == "peak" else 50
                new_count = max(0, min(new_count, max_cap))

                TrafficData.objects.create(
                    intersection=inter,
                    direction=direction,
                    lane_type=lane,
                    vehicle_count=new_count,
                    timestamp=timestamp,
                    mode=mode,
                )

    print("✅ Lane-wise traffic data updated (bounded 0–50).")
    return "Lane-based traffic data generated"


@shared_task()
def run_signal_algorithm():
    mode = detect_mode()
    controller = DynamicSignalController()
    results = controller.run_for_all()

    print(f"🚦 Signal Algorithm executed in {mode} mode")
    return results


@shared_task(bind=True, name="traffic.tasks.retrain_queue_model")
def retrain_queue_model(self):
    """
    Production-safe retraining task.
    - Prevents concurrent runs
    - Logs results in MLTrainingLog
    """
    mode = detect_mode()
    print(f"📘 ML Retraining for Mode: {mode}")

    log = MLTrainingLog.objects.create(status="running")
    start_time = time.time()

    project_root = os.path.dirname(os.path.dirname(__file__))
    script_path = os.path.join(project_root, "scripts", "train_queue_model.py")

    try:
        with ml_training_lock():
            print("🚀 Starting ML model retraining...")

            result = subprocess.run(
                [sys.executable, script_path, "--mode", mode],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            # parse metrics from stdout (optional)
            mae = None
            r2 = None
            for line in result.stdout.splitlines():
                if "MAE:" in line:
                    mae = float(line.split("MAE:")[1].split()[0])
                if "R2:" in line.upper():
                    r2 = float(line.split("R2:")[1].split()[0])

            log.status = "success"
            log.mae = mae
            log.r2 = r2
            log.completed_at = timezone.now()
            log.save()

            print(f"✅ Retraining finished in {time.time()-start_time:.1f}s")
            print(f"📊 MAE={mae:.3f}, R²={r2:.3f}")

    except RuntimeError as e:
        log.status = "failed"
        log.error_message = str(e)
        log.completed_at = timezone.now()
        log.save()
        print(f"⚠️ Skipped retraining: {e}")

    except subprocess.CalledProcessError as e:
        log.status = "failed"
        log.error_message = e.stderr
        log.completed_at = timezone.now()
        log.save()
        print(f"❌ Training script failed: {e.stderr}")

    except Exception as e:
        log.status = "failed"
        log.error_message = traceback.format_exc()
        log.completed_at = timezone.now()
        log.save()
        print(f"🔥 Unexpected training error: {e}")
