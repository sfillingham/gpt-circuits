from config.sae.models import SAEVariant


class StandardExperiment:
    """
    Base setup for "Standard" experiment.
    """

    experiment_name: str = "all-layers"
    sae_variant: SAEVariant = SAEVariant.STANDARD
    n_features: tuple[int, ...] = tuple(64 * n for n in (8, 8, 32, 64, 64))

    # Sweep parameters
    num_sweeps = 1
    num_sweep_steps = 8

    # Regularization parameters
    regularization_coefficient: float = 10000.0
    regularization_max_steps = 15000

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.00009, 0.0002, 0.0006, 0.0010, 0.0032)
    sweep_normal_ending_coefficients = (0.00025, 0.0020, 0.0050, 0.0050, 0.0120)


class RegularizeAllLayersExperiment(StandardExperiment):
    """
    Use all layers when regularizing.
    """

    regularization_l1_coefficients = (0.01, 0.03, 0.05, 0.06, 0.07)  # Targets l0s ~ 10
    regularization_trainable_layers = (0, 1, 2, 3, 4)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.00006, 0.00006, 0.00012, 0.0003, 0.0005)
    sweep_regularized_ending_coefficients = (0.00024, 0.00050, 0.00110, 0.0012, 0.0017)


class RegularizeLastLayersExperiment(StandardExperiment):
    """
    Only regularize weights using layers 3 and 4.
    """

    experiment_name: str = "last-layers"
    regularization_l1_coefficients = (0.0, 0.0, 0.0, 0.03, 0.03)  # Targets l0s ~ 10
    regularization_trainable_layers = (3, 4)

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.00005, 0.00012, 0.00030, 0.00035, 0.00050)
    sweep_regularized_ending_coefficients = (0.00025, 0.00090, 0.00250, 0.00135, 0.00160)
