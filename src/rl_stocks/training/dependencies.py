"""Verify the reviewed research release set before neural training."""
import importlib.metadata
import sys

EXPECTED = {'numpy':'2.2.6','torch':'2.8.0','gymnasium':'1.2.0',
            'stable-baselines3':'2.7.0','sb3-contrib':'2.7.0','pandas':'2.3.2','plotly':'6.3.0'}


def verify_environment():
    if sys.version_info[:2] != (3,13):
        raise RuntimeError('research training requires Colab-matched Python 3.13')
    actual = {package:importlib.metadata.version(package) for package in EXPECTED}
    mismatches = {package:version for package,version in actual.items()
                  if version.split('+')[0] != EXPECTED[package]}
    if mismatches:
        raise RuntimeError(f'dependency versions differ from reviewed pins: {mismatches}')
    return actual
