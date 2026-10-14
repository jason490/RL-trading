"""Official CLI entry point with writable logging; authentication is unchanged."""
from pathlib import Path
import logging
import os
import sys
from colab_cli import cli
from colab_cli.state import SettingsStore
from colab_cli.history import HistoryLogger

root = Path(__file__).resolve().parents[1]
state = root / '.runtime'
state.mkdir(exist_ok=True)
os.chmod(state, 0o700)

def setup_logging(log_to_stderr):
    # Library DEBUG logging includes HTTP details. Keep only warnings locally.
    handler = logging.FileHandler(state / 'colab.log')
    os.chmod(state / 'colab.log', 0o600)
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)

cli.setup_logging = setup_logging
cli.state._settings_store = SettingsStore(str(state / 'settings.json'))
cli.state._history = HistoryLogger(str(state / 'history'))
sys.path.insert(0,str(root/'src'))
from colab_cli.contents import ContentsClient
from rl_stocks.runtime.colab.transfers import download_binary

def stream_download(self, remote_path, local_path):
    download_binary(self.base_url,self.token,remote_path,local_path)

ContentsClient.download=stream_download
if __name__ == '__main__':
    if sys.argv[1:]==['usage']:
        sys.path.insert(0,str(root/'src'))
        from rl_stocks.runtime.colab.control import process_stop_request
        if process_stop_request(root):sys.exit(1)
    cli.app(args=['--config', str(state / 'sessions.json'), *sys.argv[1:]])
