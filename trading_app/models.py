from django.db import models


class DataUpdateLog(models.Model):
    """
    Track data update operations.
    """
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=[
        ('started', 'Started'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ])
    message = models.TextField(blank=True)
    duration_seconds = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']
        verbose_name = 'Data Update Log'
        verbose_name_plural = 'Data Update Logs'

    def __str__(self):
        return f"{self.status.upper()} - {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"