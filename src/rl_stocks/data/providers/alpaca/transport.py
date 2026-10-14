"""Pinned SDK transport extension: bounded network waits and read-only requests."""
from requests import Session

class BoundedSession(Session):
    def request(self, method, url, **kwargs):
        if method.upper() != 'GET':
            raise ValueError('historical-data transport only supports GET')
        if kwargs.get('timeout') is None:
            kwargs['timeout'] = (5, 30)
        return super().request(method, url, **kwargs)
