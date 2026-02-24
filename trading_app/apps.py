from django.apps import AppConfig


class TradingAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "trading_app"
    verbose_name = 'Stock Dashboard'

    # ============================================================================
    # FILE: dashboard/__init__.py
    # ============================================================================
    """
    Dashboard app for stock forecasting and analysis.
    """

    default_app_config = 'dashboard.apps.DashboardConfig'

    # ============================================================================
    # FILE: dashboard/apps.py
    # ============================================================================
    """
    Dashboard app configuration.
    """

    from django.apps import AppConfig

    class DashboardConfig(AppConfig):
        default_auto_field = 'django.db.models.BigAutoField'
        name = 'dashboard'
        verbose_name = 'Stock Dashboard'

        def ready(self):
            """
            Code to run when the app is ready.
            Initialize Hugging Face login.
            """
            import os
            from django.conf import settings
            from huggingface_hub import login

            # Login to Hugging Face
            if settings.HUGGING_FACE_TOKEN:
                try:
                    login(token=settings.HUGGING_FACE_TOKEN)
                except Exception as e:
                    print(f"Warning: Failed to login to Hugging Face: {e}")

            # Set environment variable for file watcher
            os.environ['STREAMLIT_WATCHER_TYPE'] = 'none'
