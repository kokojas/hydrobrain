"""WSGI entry point for PythonAnywhere's standard web worker."""
import os
import sys
from pathlib import Path

# Limit numerical libraries before importing numpy/scikit-learn.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('HYDRO_DATA', str(Path.home() / 'hydrobrain-data'))
os.chdir(ROOT)

from a2wsgi import ASGIMiddleware
from server import app

# uWSGI can import this module before forking its worker. Event-loop threads
# cannot survive a fork, so create the adapter in the process serving a request.
_adapter = None
_adapter_pid = None


def application(environ, start_response):
    global _adapter, _adapter_pid
    pid = os.getpid()
    if _adapter is None or _adapter_pid != pid:
        _adapter = ASGIMiddleware(app, wait_time=5)
        _adapter_pid = pid
    return _adapter(environ, start_response)
