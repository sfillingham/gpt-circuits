from config.sae.models import SAEVariant


class StandardExperimentSetup:
    """
    Setup for "Standard" experiment.
    """

    experiment_name: str = "standard"
    sae_variant: SAEVariant = SAEVariant.STANDARD
    n_features: tuple[int, ...] = tuple(64 * n for n in (8, 8, 32, 64, 64))

    # Regularization parameters
    regularization_coefficient: float = 10000.0
    regularization_l1_coefficients = (0.01, 0.03, 0.05, 0.06, 0.07)  # Targets l0s ~ 10
    regularization_max_steps = 15000
    regularization_trainable_layers = (0, 1, 2, 3, 4)

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.00015, 0.0002, 0.0010, 0.0010, 0.0040)
    sweep_normal_ending_coefficients = (0.0016, 0.0030, 0.0100, 0.0100, 0.0120)
    num_normal_sweeps = 1
    num_normal_steps = 8

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.0001, 0.0001, 0.0003, 0.0004, 0.0008)
    sweep_regularized_ending_coefficients = (0.0008, 0.0015, 0.0030, 0.0030, 0.0020)
    num_regularized_sweeps = 1
    num_regularized_steps = 8


class LayersExperimentSetup(StandardExperimentSetup):
    """
    Setup for experiment that regularizes on select layers.
    """

    experiment_name: str = "layers"
    regularization_l1_coefficients = (0.0, 0.0, 0.0, 0.03, 0.03)  # Targets l0s ~ 10
    regularization_trainable_layers = (3, 4)

    # Regularization parameters
    regularization_max_steps = 10000

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.00006, 0.0002, 0.0006, 0.0010, 0.0026)
    sweep_normal_ending_coefficients = (0.00028, 0.0018, 0.0050, 0.0050, 0.0100)
    num_normal_sweeps = 1
    num_normal_steps = 8

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.00005, 0.00012, 0.00030, 0.00035, 0.00050)
    sweep_regularized_ending_coefficients = (0.00025, 0.00090, 0.00250, 0.00150, 0.00160)
    num_regularized_sweeps = 1
    num_regularized_steps = 8


class GatedExperimentSetup:
    """
    Setup for "Gated" experiment.
    """

    experiment_name: str = "gated"
    sae_variant: SAEVariant = SAEVariant.GATED
    n_features: tuple[int, ...] = tuple(64 * n for n in (8, 8, 32, 64, 64))

    # Regularization parameters
    regularization_coefficient: float = 100.0
    regularization_l1_coefficients = (0.11, 0.19, 1.3, 3.8, 4.1)  # Targets l0s ~ 10
    regularization_max_steps = 15000
    regularization_trainable_layers = (0, 1, 2, 3, 4)

    # Sweep range for SAE training on model using normal weights
    # The following coefficients yield l0s ~ 10.0: (0.4, 0.8, 7.5, 23.0, 40.0)
    sweep_normal_starting_coefficients = (0.2, 0.3, 2.5, 10.0, 18.0)
    sweep_normal_ending_coefficients = (1.0, 4.0, 20.0, 55.0, 90.0)
    num_normal_sweeps = 1
    num_normal_steps = 16

    # Sweep range for SAE training on model using regularized weights
    # The following coefficients yield l0s ~ 10.0: (0.15, 0.33, 1.5, 3.8, 4.1)
    sweep_regularized_starting_coefficients = (0.04, 0.1, 0.8, 2.0, 2.0)
    sweep_regularized_ending_coefficients = (0.80, 1.4, 5.0, 9.0, 9.0)
    num_regularized_sweeps = 1
    num_regularized_steps = 16


class GatedV2ExperimentSetup:
    """
    Setup for "Gated V2" experiment.
    """

    experiment_name: str = "gated_v2"
    sae_variant: SAEVariant = SAEVariant.GATED_V2
    n_features: tuple[int, ...] = tuple(64 * n for n in (8, 8, 32, 64, 64))

    # Regularization parameters
    regularization_coefficient: float = 100.0
    regularization_l1_coefficients = (0.11, 0.19, 1.3, 3.8, 4.1)  # Targets l0s ~ 10
    regularization_max_steps = 15000
    regularization_trainable_layers = (0, 1, 2, 3, 4)

    # Sweep range for SAE training on model using normal weights
    # The following coefficients yield l0s ~ 10.0: (0.4, 0.8, 7.5, 23.0, 40.0)
    sweep_normal_starting_coefficients = (0.2, 0.3, 2.5, 10.0, 18.0)
    sweep_normal_ending_coefficients = (1.0, 4.0, 20.0, 55.0, 90.0)
    num_normal_sweeps = 1
    num_normal_steps = 16

    # Sweep range for SAE training on model using regularized weights
    # The following coefficients yield l0s ~ 10.0: (0.15, 0.33, 1.5, 3.8, 4.1)
    sweep_regularized_starting_coefficients = (0.04, 0.1, 0.8, 2.0, 2.0)
    sweep_regularized_ending_coefficients = (0.80, 1.4, 5.0, 9.0, 9.0)
    num_regularized_sweeps = 1
    num_regularized_steps = 16
