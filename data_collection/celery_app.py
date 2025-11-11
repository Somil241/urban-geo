"""
Optional Celery configuration for scheduled data collection.
"""
from celery import Celery
from celery.schedules import crontab

# Create Celery app
celery_app = Celery(
    'urban_geo',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Configure Celery
celery_app.conf.update(
    timezone='UTC',
    enable_utc=True,
)

# Schedule tasks
celery_app.conf.beat_schedule = {
    'collect-traffic-data': {
        'task': 'data_collection.collect_data.scheduled_collection',
        'schedule': crontab(minute='*/15'),  # Every 15 minutes
    },
}

@celery_app.task
def scheduled_collection():
    """Celery task for scheduled data collection."""
    from data_collection.collect_data import scheduled_collection as collect
    return collect()

