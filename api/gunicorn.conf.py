# Gunicorn configuration file
import multiprocessing

print("Gunicorn 2 config loaded")
max_requests = 1000
max_requests_jitter = 50

log_file = "-"

bind = "0.0.0.0:8000"

worker_class = "uvicorn.workers.UvicornWorker"
workers = multiprocessing.cpu_count()
print(workers)
