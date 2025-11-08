# scripts/train_queue_model.py
import os, sys
import django
import joblib
import pandas as pd
from datetime import timedelta
from django.utils import timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ------------- Django setup (adjust module path) -------------
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "config.settings"
)  # change if your settings module differs
django.setup()

from traffic.models import TrafficData  # import after django.setup()

# ------------- Config -------------
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "traffic", "ml_artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ------------- Load data from DB -------------
cutoff = timezone.now() - timedelta(days=1)
qs = TrafficData.objects.filter(timestamp__gte=cutoff).values(
    "intersection__name", "direction", "lane_type", "vehicle_count", "timestamp"
)
df = pd.DataFrame(list(qs))
if df.empty:
    raise SystemExit("No TrafficData in DB. Generate some data first.")

# normalize column names
df.rename(columns={"intersection__name": "intersection"}, inplace=True)
df = df.sort_values(["intersection", "direction", "lane_type", "timestamp"])

# create combined feature 'dir_lane'
df["dir_lane"] = df["direction"].astype(str) + "-" + df["lane_type"].astype(str)

# create time features
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["dayofweek"] = df["timestamp"].dt.dayofweek

# target: next vehicle_count for same intersection+dir_lane
df["next_count"] = df.groupby(["intersection", "dir_lane"])["vehicle_count"].shift(-1)
df = df.dropna(subset=["next_count"])

# optionally limit rows for speed
# df = df.tail(20000)

# features
X = df[["vehicle_count", "hour", "dayofweek", "intersection", "dir_lane"]].copy()
y = df["next_count"].astype(float)

# encode categorical fields
le_inter = LabelEncoder()
X["intersection_enc"] = le_inter.fit_transform(X["intersection"])

le_dirlane = LabelEncoder()
X["dirlane_enc"] = le_dirlane.fit_transform(X["dir_lane"])

# final feature matrix
X_final = X[["vehicle_count", "hour", "dayofweek", "intersection_enc", "dirlane_enc"]]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# train
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("Training RandomForestRegressor...")
model.fit(X_train, y_train)

# eval
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:.3f}   R2: {r2:.3f}")

# save artifacts
model_path = os.path.join(ARTIFACT_DIR, "rf_queue_model.joblib")
le_inter_path = os.path.join(ARTIFACT_DIR, "le_intersection.joblib")
le_dirlane_path = os.path.join(ARTIFACT_DIR, "le_dirlane.joblib")

joblib.dump(model, model_path)
joblib.dump(le_inter, le_inter_path)
joblib.dump(le_dirlane, le_dirlane_path)

print("Saved model and encoders to:", ARTIFACT_DIR)
