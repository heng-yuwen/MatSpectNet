from .spectral_recovery import SpectralRecovery
from .matspectnet import MatSpectNet

__all__ = {"SpectralRecovery": SpectralRecovery,
           "MatSpectNet": MatSpectNet}

def get_experiment(experiment_name):
    assert experiment_name in __all__, f"The experiment {experiment_name} does not exist!"
    return __all__[experiment_name]
