import os
from celery import Celery
from .config import settings

broker_url = settings.REDIS_URL
backend_url = settings.REDIS_BACKEND

celery_app = Celery("video_backend", broker=broker_url, backend=backend_url)

# Ensure tasks in app.tasks are registered when the worker starts
celery_app.autodiscover_tasks(["app.tasks"])

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_default_queue="celery",
    task_default_exchange="celery",
    task_default_routing_key="celery",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_send_task_events=True,
    task_send_sent_event=True,
)
