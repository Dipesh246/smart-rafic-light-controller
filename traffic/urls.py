from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard_view, name='dashboard'),
    path("dashboard-data/", views.dashboard_data_api, name="dashboard_data_api"),
]
