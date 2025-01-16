from config.sae.models import SAEVariant


class StandardExperimentSetup:
    """
    Setup for "Standard" experiment.
    """

    experiment_name: str = "standard"
    sae_variant: SAEVariant = SAEVariant.STANDARD
    n_features: tuple[int, ...] = tuple(64 * n for n in (8, 8, 32, 64, 64))

    # Regularization parameters
    regularization_coefficient: float = 1.0
    regularization_loss_coefficients = (0.0002, 0.0010, 0.0035, 0.0040, 0.0065)  # Targets l0s ~ 10
    regularization_max_steps = 15000

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.0001, 0.0002, 0.0006, 0.0010, 0.0030)
    sweep_normal_ending_coefficients = (0.0020, 0.0040, 0.0100, 0.0100, 0.0200)
    num_normal_sweeps = 1
    num_normal_steps = 8

    # Sweep range for SAE training on model using regularized weights
    # The following coefficients yield l0s ~ 10.0: (0.15, 0.33, 1.5, 3.8, 4.1)
    sweep_regularized_starting_coefficients = (0.0001, 0.0003, 0.0010, 0.0015, 0.0030)
    sweep_regularized_ending_coefficients = (0.0010, 0.0050, 0.0100, 0.0100, 0.0150)
    num_regularized_sweeps = 1
    num_regularized_steps = 8


class StandardV2ExperimentSetup:
    """
    Setup for "Standard V2" experiment.
    """

    experiment_name: str = "standard_v2"
    sae_variant: SAEVariant = SAEVariant.STANDARD_V2
    n_features: tuple[int, ...] = tuple(64 * n for n in (8, 8, 32, 64, 64))

    # Regularization parameters
    regularization_coefficient: float = 10.0
    regularization_loss_coefficients = (0.0003, 0.0006, 0.0018, 0.0020, 0.0030)  # Targets l0s ~ 10
    regularization_max_steps = 15000

    # Sweep range for SAE training on model using normal weights
    sweep_normal_starting_coefficients = (0.0001, 0.0002, 0.0006, 0.0010, 0.0030)
    sweep_normal_ending_coefficients = (0.0020, 0.0040, 0.0100, 0.0100, 0.0200)
    num_normal_sweeps = 1
    num_normal_steps = 8

    # Sweep range for SAE training on model using regularized weights
    sweep_regularized_starting_coefficients = (0.0001, 0.0002, 0.0005, 0.0008, 0.0015)
    sweep_regularized_ending_coefficients = (0.0010, 0.0020, 0.0050, 0.0060, 0.0070)
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
    regularization_loss_coefficients = (0.11, 0.19, 1.3, 3.8, 4.1)  # Targets l0s ~ 10
    regularization_max_steps = 15000

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
    regularization_loss_coefficients = (0.11, 0.19, 1.3, 3.8, 4.1)  # Targets l0s ~ 10
    regularization_max_steps = 15000

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
