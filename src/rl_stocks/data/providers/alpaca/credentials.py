"""Read data credentials without executing dotenv contents or changing the environment."""
import os
from pathlib import Path
import shlex


def credentials(env_file=".env"):
    pairs = (("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
             ("ALPACA_API_KEY", "ALPACA_SECRET_KEY"))
    # Use a complete pair from one source; never mix credentials across sources.
    for key, secret in pairs:
        if os.environ.get(key) and os.environ.get(secret):
            return os.environ[key], os.environ[secret]
    values = {}
    path = Path(env_file)
    if path.is_file():
        names = {name for pair in pairs for name in pair}
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].lstrip()
            name, sep, value = line.partition("=")
            name = name.strip()
            if not sep or name not in names:
                continue
            try:
                tokens = shlex.split(value, comments=True)
            except ValueError:
                raise ValueError("Malformed Alpaca credential assignment in .env") from None
            if len(tokens) != 1:
                raise ValueError("Alpaca credentials must be single nonempty values")
            values[name] = tokens[0]
    for key, secret in pairs:
        if values.get(key) and values.get(secret):
            return values[key], values[secret]
    raise ValueError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY in the environment or .env")
