"""Read-only task execution provenance, including allowlisted CPU math selectors."""
import hashlib
import os
import platform
from pathlib import Path

import numpy as np
import torch

from rl_stocks.core.execution_provenance import execution_backend


MATH_SELECTORS = {
    'ATEN_CPU_CAPABILITY': {'default', 'avx2', 'avx512'},
    'MKL_CBWR': {'AUTO', 'COMPATIBLE', 'SSE2', 'SSE4_2', 'AVX', 'AVX2', 'AVX512', 'AVX2,STRICT'},
    'MKL_ENABLE_INSTRUCTIONS': {'SSE4_2', 'AVX', 'AVX2', 'AVX512', 'AVX512_MIC', 'AVX512_MIC_E1'},
}


def task_execution_backend(device):
    value = execution_backend(device)
    value['cpu_math_environment'] = {
        name: None if os.environ.get(name) is None else
        os.environ[name] if os.environ[name] in allowed else 'unrecognized'
        for name, allowed in MATH_SELECTORS.items()}
    value['libc'] = list(platform.libc_ver())
    value['numpy_version'] = np.__version__
    value['torch_build_config_sha256'] = hashlib.sha256(torch.__config__.show().encode()).hexdigest()
    value['cpu_model'] = None
    try:
        with Path('/proc/cpuinfo').open() as stream:
            for index, line in enumerate(stream):
                if index >= 100:
                    break
                key, separator, data = line.partition(':')
                if separator and key.strip() == 'model name':
                    value['cpu_model'] = data.strip()[:200]
                    break
    except OSError:
        pass
    return value
