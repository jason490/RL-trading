"""Pinned SDK transport extension: bounded network waits and read-only requests."""
from requests import Session
import threading
import time


class RequestPacer:
    """One process-wide quota, shared by clients, pagination and SDK retries."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.next_request = 0.0
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            while self.clock() < self.next_request:
                self.sleep(self.next_request - self.clock())
            # Slightly below 180/minute; no initial burst allowance.
            self.next_request = self.clock() + 0.34

    def cooldown(self, seconds):
        with self.lock:
            self.next_request = max(self.next_request, self.clock() + seconds)


_PACER = RequestPacer()

class BoundedSession(Session):
    def __init__(self, pacer=None):
        super().__init__()
        self.pacer = pacer if pacer is not None else _PACER

    def request(self, method, url, **kwargs):
        if method.upper() != 'GET':
            raise ValueError('historical-data transport only supports GET')
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (5, 30)
        self.pacer.wait()
        response = super().request(method, url, **kwargs)
        if response.status_code == 429:
            # A full quota window before another attempt. Fail on longer server
            # cooldowns rather than retry before the requested time.
            try:
                seconds = max(60.0, float(response.headers.get('Retry-After', 60)))
            except (TypeError, ValueError):
                seconds = 60.0
            if seconds > 60:
                raise RuntimeError('Alpaca rate limit requires a longer cooldown; stop downloader')
            self.pacer.cooldown(seconds)
        return response
