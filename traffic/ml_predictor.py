# traffic/ml_predictor.py
import os
import joblib
import numpy as np
from datetime import datetime
from collections import defaultdict

from django.utils import timezone
from traffic.models import Intersection, TrafficData

# path to artifacts (adjust if different)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # project/traffic
ARTIFACT_DIR = os.path.join(BASE_DIR, "traffic\ml_artifacts")


class MLQueuePredictor:
    def __init__(self, mode="normal"):
        # try to load artifacts; if missing, mark unavailable
        self.mode = mode
        self.model = None
        self.le_inter = None
        self.le_dirlane = None
        try:
            model_path = os.path.join(ARTIFACT_DIR, mode, f"rf_queue_model_{mode}.joblib")
            le_inter_path = os.path.join(ARTIFACT_DIR, mode, f"le_intersection_{mode}.joblib")
            le_dirlane_path = os.path.join(ARTIFACT_DIR, mode, f"le_dirlane_{mode}.joblib")

            self.model = joblib.load(model_path)
            self.le_inter = joblib.load(le_inter_path)
            self.le_dirlane = joblib.load(le_dirlane_path)
            print(f"✅ Loaded ML model for mode '{mode}'")

        except Exception as e:
            # artifacts missing -> fallback behavior
            print("MLQueuePredictor init: could not load artifacts:", e)
            self.model = None

    def is_available(self):
        return self.model is not None

    def _build_feature_row(self, intersection_name, dir_lane, vehicle_count, ts=None):
        """
        Build numeric feature vector consistent with training.
        """
        if ts is None:
            ts = timezone.localtime(timezone.now())
        hour = ts.hour
        dayofweek = ts.weekday()
        # Encode categorical using label encoders
        try:
            inter_enc = int(self.le_inter.transform([intersection_name])[0])
        except Exception:
            # unknown intersection -> map to a fallback integer (mean)
            inter_enc = -1
        try:
            dirlane_enc = int(self.le_dirlane.transform([dir_lane])[0])
        except Exception:
            dirlane_enc = -1

        return [vehicle_count, hour, dayofweek, inter_enc, dirlane_enc]

    def predict_for_intersection(self, intersection: Intersection, limit=10):
        """
        Predicts next vehicle count per direction-lane key for an intersection.
        Clamps predictions to [0, 50] to avoid outliers.
        """
        if not self.is_available():
            return {}

        directions = ["N", "E", "S", "W"]
        lanes = ["straight", "left", "right"]
        preds = {}

        for d in directions:
            for lane in lanes:
                key = f"{d}-{lane}"

                # Get latest observed value
                last = (
                    TrafficData.objects.filter(
                        intersection=intersection, direction=d, lane_type=lane, mode=self.mode
                    )
                    .order_by("-timestamp")
                    .first()
                )

                vc = last.vehicle_count if last else 0
                ts = last.timestamp if last else timezone.localtime(timezone.now())

                feat = self._build_feature_row(intersection.name, key, vc, ts)
                X = np.array([feat], dtype=float)

                try:
                    pred = float(self.model.predict(X)[0])
                except Exception:
                    # Fallback if prediction fails
                    pred = vc

                # ✅ Clamp to realistic range
                preds[key] = round(max(0, min(pred, 50)), 2)

        return preds


    def run_for_all(self):
        results = defaultdict(dict)
        for inter in Intersection.objects.all():
            results[inter.name] = self.predict_for_intersection(inter)
        return results
