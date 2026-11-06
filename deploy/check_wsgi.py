"""Verify WSGI responses before and after a fork (uWSGI deployment contract)."""
import io
import json
import multiprocessing
from wsgiref.util import setup_testing_defaults

from deploy.pythonanywhere_wsgi import application


def request():
    environ = {}
    setup_testing_defaults(environ)
    environ.update(PATH_INFO='/healthz', REQUEST_METHOD='GET', **{'wsgi.input': io.BytesIO()})
    status = []
    result = application(environ, lambda value, headers, exc_info=None: status.append(value))
    try:
        payload = json.loads(b''.join(result))
    finally:
        if hasattr(result, 'close'):
            result.close()
    assert status == ['200 OK'], status
    assert payload['status'] == 'ok' and payload['ml_available'], payload
    return payload


def child(conn):
    conn.send(request())
    conn.close()


if __name__ == '__main__':
    print('Parent WSGI request:', request(), flush=True)
    ctx = multiprocessing.get_context('fork')
    receive, send = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=child, args=(send,))
    worker.start()
    try:
        assert receive.poll(15), 'WSGI request hung after fork'
        print('Child WSGI request:', receive.recv(), flush=True)
        worker.join(5)
        assert worker.exitcode == 0, worker.exitcode
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join()
    print('WSGI fork regression check passed', flush=True)
