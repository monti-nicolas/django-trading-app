from django.urls import path
from . import views

app_name = 'trading_app'

urlpatterns = [
    # Main dashboard view
    path('', views.DashboardView.as_view(), name='index'),

    # Update data endpoint (HTMX)
    path('update/', views.UpdateDataView.as_view(), name='update_data'),

    # Get ticker details (HTMX)
    path('ticker/<str:ticker>/', views.TickerDetailView.as_view(), name='ticker_detail'),

    # API endpoints for charts data
    path('api/chart-data/<str:ticker>/', views.get_chart_data, name='chart_data'),
]