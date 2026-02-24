from django.contrib import admin
from .models import DataUpdateLog


@admin.register(DataUpdateLog)
class DataUpdateLogAdmin(admin.ModelAdmin):
    list_display = ['timestamp', 'status', 'duration_seconds', 'message']
    list_filter = ['status', 'timestamp']
    readonly_fields = ['timestamp', 'status', 'message', 'duration_seconds']

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False