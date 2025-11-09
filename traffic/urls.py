from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard_view, name="dashboard"),
    path("dashboard-data/", views.dashboard_data_api, name="dashboard_data_api"),
    path("training-metrics/", views.training_metrics_api, name="training_metrics_api"),
    path("signal-state/", views.signal_state_api, name="signal_state"),
    path("flow-stats/", views.flow_stats_api, name="flow_stats"),
    path("historical-logs/", views.historical_logs_api, name="historical_logs"),
    path("sync-status/", views.synchronization_status_api, name="sync_status"),
]
