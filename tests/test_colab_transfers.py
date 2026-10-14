from pathlib import Path
import tempfile
import unittest
from rl_stocks.runtime.colab.transfers import download_binary,TransferError


class Response:
    status_code=200
    headers={'Content-Length':'6'}
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def iter_content(self,chunk_size):yield b'abc';yield b'def'


class TransferTests(unittest.TestCase):
    def test_streamed_atomic_file_and_bounded_authenticated_request(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'file.bin'
            def get(url,**kwargs):
                self.assertEqual(url,'https://example.invalid/files/content/file.bin')
                self.assertEqual(kwargs['timeout'],(10,15));self.assertTrue(kwargs['stream'])
                self.assertFalse(kwargs['allow_redirects'])
                return Response()
            download_binary('https://example.invalid','private','/content/file.bin',destination,get)
            self.assertEqual(destination.read_bytes(),b'abcdef')
            self.assertFalse(destination.with_name('file.bin.partial').exists())

    def test_failure_keeps_existing_file_and_never_exposes_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'file.bin';destination.write_bytes(b'previous')
            def get(*args,**kwargs):raise ValueError('private-token-in-request')
            with self.assertRaises(TransferError) as caught:
                download_binary('https://example.invalid','private','/content/file.bin',destination,get)
            self.assertNotIn('private',str(caught.exception))
            self.assertEqual(destination.read_bytes(),b'previous')
            response=Response();response.headers={'Content-Length':'100'}
            with self.assertRaises(TransferError):
                download_binary('https://example.invalid','private','/content/file.bin',destination,lambda *a,**k:response)
            self.assertEqual(destination.read_bytes(),b'previous')
