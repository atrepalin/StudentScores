from typing import Literal

import pynvml


def is_cuda_available():
    """
    Check if CUDA is available

    Returns:
        bool
    """
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        if device_count == 0:
            return False

        pynvml.nvmlShutdown()
        return True
    except pynvml.NVMLError:
        print("CUDA/NVML is not available or no NVIDIA driver found.")
        return False


def core_factory(device: Literal["cuda", "cpu"]):
    """
    Create core based on device type and cuml availability

    Args:
        device (Literal["cuda", "cpu"]): "cuda" or "cpu"

    Returns:
        core: sklearn or cuml
    """
    try:
        if device == "cuda":
            import cuml as core
            import cuml.accel

            cuml.accel.install()
        else:
            raise RuntimeError("Backend mismatch")
    except:
        import sklearn as core

    return core
