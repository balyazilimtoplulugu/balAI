# gunicorn.conf.py
timeout = 120  # Increase timeout to 120 seconds
workers = 1    # Use only one worker to reduce memory usage
worker_class = 'sync'
max_requests = 1000  # Restart worker after 1000 requests to prevent memory leaks
max_requests_jitter = 50  # Add some randomness to prevent all workers restarting at once