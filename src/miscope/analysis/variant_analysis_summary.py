from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from miscope.families.variant import Variant



_FIRST_DESCENT_TRAIN_LOSS_THRESHOLD = 1.0e-6
_FIRST_MOVER_COUNT = 40
_TOTAL_NEURON_COUNT_OVER_THRESHOLD = 100
_SECOND_DESCENT_TEST_LOSS_THRESHOLD = 1.0e-6
_SECOND_DESCENT_ONSET_DIFF_THRESHOLD: float = 0.8

_EMBED_FRAC_EXPLAINED_BY_FREQUENCY = 0.3
_ATTENTION_FRAC_EXPLAINED_BY_FREQUENCY_FLOOR = 0.7
_NEURON_FRAC_EXPLAINED_BY_FREQUENCY = 0.7
_SPECIALIZATION_FLOOR: float = 0.10

_EARLY_SECOND_DESCENT_EPOCH: int = 9000
_LATE_SECOND_DESCENT_EPOCH: int = 12000
_SUCCESSFUL_TEST_LOSS_THRESHOLD: float = 1.0e-5
_REBOUND_TEST_LOSS_THRESHOLD: float = 0.2

@dataclass
class VariantAnalysisData:
    variant: Variant
    available_checkpoints: list[Any] = field(default_factory=list)
    losses_loaded: bool = False
    train_losses: list[Any] = field(default_factory=list)
    test_losses: list[Any] = field(default_factory=list)
    available_checkpoints: list[Any] = field(default_factory=list)
    
    neurons_loaded: bool = False
    neurons_checkpoints: list[Any] = field(default_factory=list)
    neurons_dominant_frequencies: np.ndarray | None = None
    neurons_frequency_specialization: np.ndarray | None = None
    neurons_commitment_epochs: np.ndarray | None = None
    
    effective_dimensionality_loaded: bool = False
    effective_dimensionality_pr_epochs: list[Any] = field(default_factory=list)
    effective_dimensionality_pr_w_in: list[Any] = field(default_factory=list)
    effective_dimensionality_pr_w_out: list[Any] = field(default_factory=list)

    geometry_loaded: bool = False
    geometry_checkpoints: list[Any] = field(default_factory=list)
    geometry_resid_post_circularity: list[Any] = field(default_factory=list)
    geometry_resid_post_fisher_mean: list[Any] = field(default_factory=list)

    def load(self):
        self.available_checkpoints = self.variant.get_available_checkpoints()

    def load_loss_data(self):
        self.train_losses = list(self.variant.metadata["train_losses"])
        self.test_losses = list(self.variant.metadata["test_losses"])
        self.losses_loaded = True

    def load_neuron_data(self):
        neuron_dynamics_data = self.variant.artifacts.load_cross_epoch("neuron_dynamics")
        self.neurons_checkpoints = list(neuron_dynamics_data["epochs"])
        self.neurons_dominant_frequencies = neuron_dynamics_data["dominant_freq"]  # (n_epochs, d_mlp)
        self.neurons_frequency_specialization = neuron_dynamics_data["max_frac"]  # (n_epochs, d_mlp)
        self.neurons_commitment_epochs = neuron_dynamics_data["commitment_epochs"]  # (d_mlp,)
        self.neurons_loaded = True
    
    def load_effective_dimensionality_data(self):
        effective_dimensionality_data = self.variant.artifacts.load_summary("effective_dimensionality")
        self.effective_dimensionality_pr_epochs = list(effective_dimensionality_data["epochs"])
        self.effective_dimensionality_pr_w_in = list(effective_dimensionality_data["pr_W_in"])
        self.effective_dimensionality_pr_w_out = list(effective_dimensionality_data["pr_W_out"])
        self.effective_dimensionality_loaded = True

    def load_geometry(self):
        geometry_data = self.variant.artifacts.load_summary("repr_geometry")

        self.geometry_resid_post_circularity = list(geometry_data["resid_post_circularity"])
        self.geometry_resid_post_fisher_mean = list(geometry_data["resid_post_fisher_mean"])
        self.geometry_loaded = True

        
class VariantAnalysisSummary:
    variant: Variant
    analysis_data: VariantAnalysisData
    summary_data: dict[str, Any]

    def __init__(self, variant: Variant):
        self.variant = variant
        self.analysis_data = VariantAnalysisData(variant)
        self.analysis_data.load()
        self.summary_data = {}

    def analyze(self) -> None:
        self._load_metrics()
        self._write_summary()

    def _get_nearest_checkpoint_epoch(self, epoch: int) -> int:

        nearest_checkpoint_epoch = 0

        available_checkpoints = self.variant.get_available_checkpoints()
        if available_checkpoints:
            distances = [abs(e - epoch) for e in available_checkpoints]
            nearest_checkpoint_epoch = available_checkpoints[distances.index(min(distances))]

        return nearest_checkpoint_epoch

    def _get_nearest_checkpoint_epoch_index(self, epoch: int) -> int:

        nearest_checkpoint_epoch_index = 0

        available_checkpoints = self.variant.get_available_checkpoints()
        if available_checkpoints:
            distances = [abs(e - epoch) for e in available_checkpoints]
            nearest_checkpoint_epoch_index = distances.index(min(distances))

        return nearest_checkpoint_epoch_index

    def _get_frequency_bands(self, frequencies: list[int], prime: int) -> list[str]:
        """Classify a frequency into low / mid / high band relative to prime."""
        bands = []
        for freq in frequencies:
            if freq <= prime // 4:
                bands.append("low")
            elif freq > 3 * prime // 8:
                bands.append("high")
            else:
                bands.append("mid")

        return list(set(bands))

    def _get_learned_frequencies(self, epoch_index: int) -> list[int]:

        if not self.analysis_data.neurons_loaded:
            self.analysis_data.load_neuron_data()

        frequencies_over_threshold = []
        dominant_freq = self.analysis_data.neurons_dominant_frequencies  # (n_epochs, d_mlp)
        max_frac = self.analysis_data.neurons_frequency_specialization  # (n_epochs, d_mlp)

        if max_frac is not None and dominant_freq is not None:
            neuron_fracs = max_frac[epoch_index, :]
            # list of neurons over threshold
            neurons_over_threshold = list(set(i for i, x in enumerate(neuron_fracs) if x >= _NEURON_FRAC_EXPLAINED_BY_FREQUENCY))
            # total count of neurons over threshold
            #count_neurons_over_threshold = len(neurons_over_threshold)
            # frequencies over threshold
            frequencies_over_threshold = list(set(frequency_idx + 1 for frequency_idx in dominant_freq[epoch_index, neurons_over_threshold]))

        return frequencies_over_threshold

    def _get_variant_preformance_classification(self) -> tuple[str, list[str]]:
        """Classify a variant's failure mode from its metrics.

        Returns a (category, reasons) tuple. Reasons list the specific rules that
        fired — classification is fully auditable.

        Categories (in priority order):
            no_grokking: never crossed the grokking threshold during training
            degraded_recovery: entered second descent, test loss climbed back
            degraded: high final test loss, never properly descended
            late_grokker: grokked but significantly past the expected window
            healthy: grokked on time, final loss acceptable

        Args:
            metrics: Dict as returned by compute_variant_metrics.
            rules: ClassificationRules instance. If None, uses defaults.

        Returns:
            Tuple of (category_str, reasons_list).
        """
        reasons: list[str] = []
        second_descent_onset_epoch = self.summary_data["second_descent_onset_epoch"]
        final_test_loss = self.summary_data["test_loss_final"]
        #band_count = metrics.get("frequency_band_count")
        min_test_loss = self.summary_data["test_loss_min"]
        post_descent_rebound = (final_test_loss - min_test_loss) >= _REBOUND_TEST_LOSS_THRESHOLD

        # 1. no_grokking: never crossed threshold
        if second_descent_onset_epoch is None or second_descent_onset_epoch == 0:
            reasons.append(f"test loss never dropped more than {_SECOND_DESCENT_ONSET_DIFF_THRESHOLD} between epochs")
            return "no_second_descent", reasons

        # 2. degraded_recovery: entered second descent, test loss climbed back
        if (
            second_descent_onset_epoch is not None 
            and second_descent_onset_epoch > 0
            and post_descent_rebound is True
            and (final_test_loss is None or final_test_loss > _SUCCESSFUL_TEST_LOSS_THRESHOLD)
        ):
            reasons.append(
                f"second_descent_onset={second_descent_onset_epoch}, post-descent recovery detected"
            )
            if final_test_loss is not None:
                reasons.append(f"final_test_loss={final_test_loss:.6f} > {_SUCCESSFUL_TEST_LOSS_THRESHOLD}")
            return "degraded_rebound", reasons

        # 3. degraded: high final loss, never properly descended
        if final_test_loss is not None and final_test_loss > _SUCCESSFUL_TEST_LOSS_THRESHOLD:
            reasons.append(f"final_test_loss={final_test_loss:.6f} > {_SUCCESSFUL_TEST_LOSS_THRESHOLD}")
            return "degraded", reasons

        # 4. late_grokker: grokked but past expected window
        if second_descent_onset_epoch > _LATE_SECOND_DESCENT_EPOCH:
            reasons.append(f"second_descent_onset={second_descent_onset_epoch} > {_LATE_SECOND_DESCENT_EPOCH}")
            return "late_grokker", reasons

        # 5. healthy
        reasons.append("grokking onset on time, final loss acceptable")
        return "healthy", reasons

    def _load_train_test_loss_metrics(self) -> None:
        
        if not self.analysis_data.losses_loaded:
            self.analysis_data.load_loss_data()

        train_losses = self.analysis_data.train_losses
        test_losses = self.analysis_data.test_losses

        train_loss_min = min(train_losses)
        train_loss_min_epoch = train_losses.index(train_loss_min)
        test_loss_min = min(test_losses)
        test_loss_min_epoch = test_losses.index(test_loss_min)
        test_loss_max = max(test_losses)
        test_loss_max_epoch = test_losses.index(test_loss_max)

        train_loss_threshold_first_epoch = next((i for i, x in enumerate(train_losses) if x <= _FIRST_DESCENT_TRAIN_LOSS_THRESHOLD), -1)
        test_loss_threshold_first_epoch = next((i for i, x in enumerate(test_losses) if x <= _SECOND_DESCENT_TEST_LOSS_THRESHOLD), -1)

        train_loss_final = float(train_losses[-1])
        test_loss_final = float(test_losses[-1])

        # Second descent onset: first epoch after peak where descent_fraction >= threshold
        second_descent_onset_epoch = None
        for i in range(test_loss_max_epoch, len(test_losses)):
            descent_fraction = (test_loss_max - test_losses[i]) / test_loss_max
            if descent_fraction >= _SECOND_DESCENT_ONSET_DIFF_THRESHOLD:
                second_descent_onset_epoch = i
                break

        # store metrics
        self.summary_data["train_loss_min"] = train_loss_min
        self.summary_data["train_loss_min_epoch"] = train_loss_min_epoch
        self.summary_data["train_loss_threshold_first_epoch"] = train_loss_threshold_first_epoch
        self.summary_data["train_loss_final"] = train_loss_final
        self.summary_data["test_loss_min"] = test_loss_min
        self.summary_data["test_loss_min_epoch"] = test_loss_min_epoch
        self.summary_data["test_loss_max"] = test_loss_max
        self.summary_data["test_loss_max_epoch"] = test_loss_max_epoch
        self.summary_data["test_loss_threshold_first_epoch"] = test_loss_threshold_first_epoch
        self.summary_data["test_loss_final"] = test_loss_final
        self.summary_data["second_descent_onset_epoch"] = second_descent_onset_epoch

    
    def _load_neuron_threshold_key_epochs(self) -> None:

        if not self.analysis_data.neurons_loaded:
            self.analysis_data.load_neuron_data()

        first_mover_epoch = -1
        first_mover_frequency = -1
        first_mover_frequency_count_threshold_epoch = -1
        total_neurons_over_specialization_threshold_epoch = -1

        epochs = self.analysis_data.neurons_checkpoints
        dominant_freq = self.analysis_data.neurons_dominant_frequencies  # (n_epochs, d_mlp)
        max_frac = self.analysis_data.neurons_frequency_specialization  # (n_epochs, d_mlp)

        if max_frac is not None and dominant_freq is not None:
            for epoch_idx, neuron_fracs in enumerate(max_frac):
                # list of neurons over threshold
                neurons_over_threshold = list(set(i for i, x in enumerate(neuron_fracs) if x >= _NEURON_FRAC_EXPLAINED_BY_FREQUENCY))
                # total count of neurons over threshold
                count_neurons_over_threshold = len(neurons_over_threshold)
                # frequencies over threshold
                frequencies_over_threshold = list(set(frequency_idx + 1 for frequency_idx in dominant_freq[epoch_idx, neurons_over_threshold]))

                # capture first mover frequency
                if len(frequencies_over_threshold) > 0 and first_mover_frequency == -1:
                    first_mover_epoch = epochs[epoch_idx]
                    first_mover_frequency = frequencies_over_threshold[0]

                # get counts for all neurons specializing in first_mover_frequency
                if first_mover_frequency > -1 and first_mover_frequency_count_threshold_epoch == -1:
                    total_first_mover_count = sum((frequency_idx + 1)==first_mover_frequency for frequency_idx in dominant_freq[epoch_idx, neurons_over_threshold])
                    if total_first_mover_count >= _FIRST_MOVER_COUNT:
                        first_mover_frequency_count_threshold_epoch = epochs[epoch_idx]

                if count_neurons_over_threshold >= _TOTAL_NEURON_COUNT_OVER_THRESHOLD and total_neurons_over_specialization_threshold_epoch == -1: 
                    total_neurons_over_specialization_threshold_epoch = epochs[epoch_idx]

        self.summary_data["first_mover_epoch"] = first_mover_epoch
        self.summary_data["first_mover_frequency"] = first_mover_frequency
        self.summary_data["first_mover_frequency_count_threshold_epoch"] = first_mover_frequency_count_threshold_epoch
        self.summary_data["total_neurons_over_specialization_threshold_epoch"] = total_neurons_over_specialization_threshold_epoch

    def _load_effective_dimensionality_key_epochs(self) -> None:

        if not self.analysis_data.effective_dimensionality_loaded:
            self.analysis_data.load_effective_dimensionality_data()

        epochs = self.analysis_data.effective_dimensionality_pr_epochs
        W_In = self.analysis_data.effective_dimensionality_pr_w_in
        W_Out = self.analysis_data.effective_dimensionality_pr_w_out
        effective_dimensionality_cross_over_epoch = -1

        # Skip initial period where W_out ≈ W_in at random init; only detect the
        # meaningful crossover after W_out has first risen clearly above W_in.
        w_out_rose_above_w_in = False
        n_epochs = len(epochs)
        for epoch_idx in range(n_epochs):
            w_out_value = W_Out[epoch_idx]
            w_in_value = W_In[epoch_idx]

            if not w_out_rose_above_w_in:
                if w_out_value > w_in_value:
                    w_out_rose_above_w_in = True
                continue

            if w_out_value <= w_in_value:
                effective_dimensionality_cross_over_epoch = epochs[epoch_idx]
                break

        self.summary_data["effective_dimensionality_cross_over_epoch"] = effective_dimensionality_cross_over_epoch

    # For a given variant, capture start and end epochs for pre-defined windows in training
    # This might be ModuloAddition-specific
    def _load_window_ranges(self) -> None:

        # First Descent:
        #   Marked by the window between epoch 0 and the first epoch where
        #       Train Loss <= FIRST_DESCENT_TRAIN_LOSS_THRESHOLD
        #
        # Plateau:
        #   Marked by the window between the end of First Descent and
        #       Second Descent Onset
        # Cascade:
        #   Marked by the window between the emergence of the First Mover
        #       and one of these indicators:
        #           1. First Frequency to surpass Neuron Count of First Mover
        #           2. First epoch when W_out crosses below W_in *
        #       First Mover emergence is when the first frequency 
        #           reaches >= FIRST_MOVER_COUNT and >= NEURON_FRAC_EXPLAINED_BY_FREQUENCY
        # Second Descent:
        #   Marked by the window between second descent onset 
        #       and test loss < SECOND_DESCENT_TEST_LOSS_THRESHOLD OR test loss = min test loss (for rebounders)
        #       Second descent onset = first epoch where test loss <= SECOND_DESCENT_TEST_LOSS_THRESHOLD
        # Final:
        #   Marked by window between Second Descent end and training end. 
        #       Models that never fully descend may not have a Final window
        #
        # Metrics needed to capture windows:
        #   First epoch below FIRST_DESCENT_TRAIN_LOSS_THRESHOLD
        #   First epoch where a frequency reaches >= FIRST_MOVER_COUNT and >= NEURON_FRAC_EXPLAINED_BY_FREQUENCY
        #   Epoch where W_Out effective dimensionality <= W_In effective dimensionality
        #   First epoch where test loss below SECOND_DESCENT_ONSET_DIFF_THRESHOLD
        #   Epoch with min test loss
        train_loss_threshold_first_epoch_nearest = 0
        second_descent_onset_epoch_nearest = 0
        test_loss_threshold_first_epoch_nearest = 0

        first_descent_window = {"start_epoch": 0, "end_epoch": 0}
        plateau_window = {"start_epoch": 0, "end_epoch": 0}
        cascade_window = {"start_epoch": 0, "end_epoch": 0}
        second_descent_window = {"start_epoch": 0, "end_epoch": 0}
        final_window = {"start_epoch": 0, "end_epoch": 0}

        # First Descent
        if self.summary_data["train_loss_threshold_first_epoch"]:
            train_loss_threshold_first_epoch_nearest = self._get_nearest_checkpoint_epoch(int(self.summary_data["train_loss_threshold_first_epoch"]))
            first_descent_window = {"start_epoch": 0, "end_epoch": train_loss_threshold_first_epoch_nearest}
        
        # Second Descent
        if self.summary_data["second_descent_onset_epoch"] and self.summary_data["train_loss_threshold_first_epoch"]:
            second_descent_onset_epoch_nearest = self._get_nearest_checkpoint_epoch(int(self.summary_data["second_descent_onset_epoch"]))
            test_loss_threshold_first_epoch_nearest = self._get_nearest_checkpoint_epoch(int(self.summary_data["test_loss_threshold_first_epoch"]))
            second_descent_window = {"start_epoch": second_descent_onset_epoch_nearest, "end_epoch": test_loss_threshold_first_epoch_nearest}

        # Plateau
        plateau_window = {"start_epoch": first_descent_window["end_epoch"], "end_epoch": second_descent_window["start_epoch"]}

        # Cascade
        if self.summary_data["first_mover_frequency_count_threshold_epoch"] and self.summary_data["effective_dimensionality_cross_over_epoch"]:
            cascade_window = {"start_epoch": int(self.summary_data["first_mover_frequency_count_threshold_epoch"]), "end_epoch": int(self.summary_data["effective_dimensionality_cross_over_epoch"])}

        # Final
        final_window = {"start_epoch": second_descent_window["end_epoch"], "end_epoch": self.variant.get_available_checkpoints()[-1]}
        
        self.summary_data["first_descent_window"] = first_descent_window
        self.summary_data["plateau_window"] = plateau_window
        self.summary_data["cascade_window"] = cascade_window
        self.summary_data["second_descent_window"] = second_descent_window
        self.summary_data["final_window"] = final_window

    # For each window, gather metrics
    def _load_window_metrics(self, window_name: str) -> None:
        
        prime = self.variant.params["prime"]

        # load data
        if not self.analysis_data.losses_loaded:
            self.analysis_data.load_loss_data()

        train_losses = self.analysis_data.train_losses
        test_losses = self.analysis_data.test_losses

        if not self.analysis_data.effective_dimensionality_loaded:
            self.analysis_data.load_effective_dimensionality_data()

        if not self.analysis_data.geometry_loaded:
            self.analysis_data.load_geometry()

        if self.summary_data[f"{window_name}_window"]:
            start_epoch = self.summary_data[f"{window_name}_window"]["start_epoch"]
            end_epoch = self.summary_data[f"{window_name}_window"]["end_epoch"]
            start_epoch_index = self._get_nearest_checkpoint_epoch_index(start_epoch)
            end_epoch_index = self._get_nearest_checkpoint_epoch_index(end_epoch)

            window_metrics = self.summary_data[f"{window_name}_window"]

            # losses
            window_metrics["train_loss_start"] = train_losses[start_epoch]
            window_metrics["train_loss_end"] = train_losses[end_epoch]
            window_metrics["test_loss_start"] = test_losses[start_epoch]
            window_metrics["test_loss_end"] = test_losses[end_epoch]
            # start/end metrics to capture:
            #   neuron, attention, embedding frequencies
            learned_frequencies_start = self._get_learned_frequencies(start_epoch_index)
            learned_frequencies_end = self._get_learned_frequencies(end_epoch_index)
            window_metrics["learned_frequencies_start"] = learned_frequencies_start
            window_metrics["learned_frequencies_end"] = learned_frequencies_end
            #   neuron, attention, embedding frequency bands
            window_metrics["learned_frequency_bands_start"] = self._get_frequency_bands(learned_frequencies_start, prime)
            window_metrics["learned_frequency_bands_end"] = self._get_frequency_bands(learned_frequencies_end, prime)
            #   effective dimensionality/SV Participation Ratio
            window_metrics["resid_post_pr_w_in_start"] = self.analysis_data.effective_dimensionality_pr_w_in[start_epoch_index]
            window_metrics["resid_post_pr_w_in_end"] = self.analysis_data.effective_dimensionality_pr_w_in[end_epoch_index]
            window_metrics["resid_post_pr_w_out_start"] = self.analysis_data.effective_dimensionality_pr_w_out[start_epoch_index]
            window_metrics["resid_post_pr_w_out_end"] = self.analysis_data.effective_dimensionality_pr_w_out[end_epoch_index]

            #   circularity
            window_metrics["resid_post_circularity_start"] = self.analysis_data.geometry_resid_post_circularity[start_epoch_index]
            window_metrics["resid_post_circularity_end"] = self.analysis_data.geometry_resid_post_circularity[end_epoch_index]
            #   fischer discriminant
            window_metrics["resid_post_fisher_mean_start"] = self.analysis_data.geometry_resid_post_fisher_mean[start_epoch_index]
            window_metrics["resid_post_fisher_mean_end"] = self.analysis_data.geometry_resid_post_fisher_mean[end_epoch_index]
            #   competition

            self.summary_data[f"{window_name}_window"] = window_metrics

        pass

    def _compute_commitment_epochs_at_threshold(self) -> np.ndarray:
        """Recompute commitment epochs using _NEURON_FRAC_EXPLAINED_BY_FREQUENCY.

        The artifact's stored commitment_epochs use a low threshold (~0.054),
        which causes many neurons to appear committed from epoch 0 due to random
        initialization. This recomputes using the same 0.7 threshold used
        throughout the rest of the summary.
        """
        epochs = np.array(self.analysis_data.neurons_checkpoints)
        dominant_freq = self.analysis_data.neurons_dominant_frequencies
        max_frac = self.analysis_data.neurons_frequency_specialization
        if dominant_freq is None or max_frac is None:
            return np.full(0, np.nan)
        n_epochs, d_mlp = dominant_freq.shape
        commitment_epochs = np.full(d_mlp, np.nan)
        final_freq = dominant_freq[-1]

        for n in range(d_mlp):
            if max_frac[-1, n] < _NEURON_FRAC_EXPLAINED_BY_FREQUENCY:
                continue
            stable_from = n_epochs - 1
            for t in range(n_epochs - 2, -1, -1):
                if max_frac[t, n] >= _NEURON_FRAC_EXPLAINED_BY_FREQUENCY and dominant_freq[t, n] == final_freq[n]:
                    stable_from = t
                else:
                    break
            commitment_epochs[n] = epochs[stable_from]

        return commitment_epochs

    def _load_competition_and_geometry_summary_metrics(self) -> None:

        if not self.analysis_data.neurons_loaded:
            self.analysis_data.load_neuron_data()

        commitment_epochs = self._compute_commitment_epochs_at_threshold()
        competition_window_start = None
        competition_window_end = None
        competition_window_duration = None

        committed_values = commitment_epochs[~np.isnan(commitment_epochs)]
        if len(committed_values) > 0:
            competition_window_start = int(committed_values.min())
            competition_window_end = int(committed_values.max())
            competition_window_duration = competition_window_end - competition_window_start

        self.summary_data["competition_window_start"] = competition_window_start
        self.summary_data["competition_window_end"] = competition_window_end
        self.summary_data["competition_window_duration"] = competition_window_duration

        if not self.analysis_data.geometry_loaded:
            self.analysis_data.load_geometry()

        circularity = self.analysis_data.geometry_resid_post_circularity
        self.summary_data["max_resid_post_circularity"] = float(max(circularity)) if circularity else None

    def _load_metrics(self) -> None:

        self.summary_data["prime"] = self.variant.params["prime"]
        self.summary_data["model_seed"] = self.variant.params["seed"]
        self.summary_data["data_seed"] = self.variant.params["data_seed"]
        self.summary_data["family"] = self.variant.family.name
        self._load_train_test_loss_metrics()
        self._load_neuron_threshold_key_epochs()
        self._load_effective_dimensionality_key_epochs()
        self._load_window_ranges()
        self._load_window_metrics("first_descent")
        self._load_window_metrics("second_descent")
        self._load_window_metrics("plateau")
        self._load_window_metrics("cascade")
        self._load_window_metrics("final")
        self._load_competition_and_geometry_summary_metrics()
        self.summary_data["performance_classification"] = self._get_variant_preformance_classification()

    def _write_summary(self) -> Path:
        output_path = self.variant.variant_dir / "variant_summary.json"
        output_path.write_text(json.dumps(self.summary_data, default=int, indent=2))
        return output_path
