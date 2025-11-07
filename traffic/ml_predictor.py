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
ARTIFACT_DIR = os.path.join(BASE_DIR, "ml_artifacts")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "rf_queue_model.joblib")
LE_INTER_PATH = os.path.join(ARTIFACT_DIR, "le_intersection.joblib")
LE_DIRLANE_PATH = os.path.join(ARTIFACT_DIR, "le_dirlane.joblib")


class MLQueuePredictor:
    def __init__(self):
        # try to load artifacts; if missing, mark unavailable
        self.model = None
        self.le_inter = None
        self.le_dirlane = None
        try:
            self.model = joblib.load(MODEL_PATH)
            self.le_inter = joblib.load(LE_INTER_PATH)
            self.le_dirlane = joblib.load(LE_DIRLANE_PATH)
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
        Predicts next vehicle count per 'direction-lane' key for an intersection.
        Returns dict like {'N-straight': 12.34, ...}
        """
        if not self.is_available():
            return {}

        # gather the latest observed vehicle_count per dir-lane
        directions = ["N", "E", "S", "W"]
        lanes = ["straight", "left", "right"]
        preds = {}

        for d in directions:
            for lane in lanes:
                key = f"{d}-{lane}"
                # get last observed vehicle_count for this dir+lane
                last = TrafficData.objects.filter(
                    intersection=intersection,
                    direction=d,
                    lane_type=lane
                ).order_by("-timestamp").first()

                if last:
                    vc = last.vehicle_count
                    ts = last.timestamp
                else:
                    vc = 0
                    ts = timezone.localtime(timezone.now())

                feat = self._build_feature_row(intersection.name, key, vc, ts)
                # handle unknown encoding: model expects numeric; we set -1 for unknowns
                X = np.array([feat], dtype=float)
                # predict
                try:
                    pred = float(self.model.predict(X)[0])
                    if pred < 0:
                        pred = 0.0
                except Exception as e:
                    # prediction failed -> fallback to current count
                    pred = vc
                preds[key] = round(pred, 2)

        return preds

    def run_for_all(self):
        results = defaultdict(dict)
        for inter in Intersection.objects.all():
            results[inter.name] = self.predict_for_intersection(inter)
        return results
