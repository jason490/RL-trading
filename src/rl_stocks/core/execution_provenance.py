"""Read-only numerical execution settings; never change backend flags or RNG."""
import os
import platform


def execution_backend(device):
    import torch

    device = torch.device(device)
    thread_environment = {}
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        value = os.environ.get(name)
        # Capture only thread counts, never arbitrary environment content.
        thread_environment[name] = (None if value is None else int(value)
                                    if value.isascii() and value.isdecimal() and len(value) <= 9
                                    else 'non-numeric')
    result = dict(
        schema='numerical-execution-v1',
        selected_device=str(device),
        machine=platform.machine(),
        torch_version=str(torch.__version__),
        cpu_capability=torch.backends.cpu.get_cpu_capability(),
        visible_logical_cpus=os.cpu_count(),
        affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        torch_intraop_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        thread_environment=thread_environment,
        default_dtype=str(torch.get_default_dtype()),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        mkldnn_enabled=torch.backends.mkldnn.enabled,
        mkldnn_deterministic=torch.backends.mkldnn.deterministic,
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_enabled=torch.backends.cudnn.enabled,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        gpu_name=None,
    )
    if device.type == 'cuda':
        result['gpu_name'] = torch.cuda.get_device_name(device)
    return result
