from typing import Optional

from .enums import DataMode, PipelineState
from .reconstruction_result import ReconstructionResult
from .pipeline_steps import (
    build_g_and_H_real,
    build_g_and_H_synthetic,
    evaluate_synthetic_metrics,
    compute_synthetic_difference,
    load_measurement_inputs,
)


class MaTirfPipeline:
    def __init__(self, config, callbacks=None):
        self.config = config
        self.callbacks = callbacks or {}

        # Mode parsé en enum typé (au lieu de string compares 'synthetic-data')
        self.mode = DataMode.from_config(config['input-paths']['mode'])

        # Tous les outputs du pipeline sont encapsulés dans un seul objet typé.
        # Au lieu d'attributs dispersés (self.f, self.f_true, self.metrics, ...),
        # on a un seul self.result qui contient tout.
        self.result: Optional[ReconstructionResult] = ReconstructionResult()

        # État explicite du pipeline. Le DisplayWindow s'abonne aux changements
        # d'état via le callback 'state_changed' pour activer/désactiver des
        # boutons (Save, Stop, etc.) en fonction de la phase courante.
        self.state = PipelineState.IDLE

        self.algorithm = None
        self._running = False

    # =========================
    # STATE MANAGEMENT
    # =========================
    def _set_state(self, new_state: PipelineState):
        """Transition d'état + émission du callback 'state_changed'."""
        old_state = self.state
        if new_state == old_state:
            return
        self.state = new_state
        self._emit("state_changed", old_state, new_state)

    # =========================
    # CALLBACK SYSTEM
    # =========================
    def _emit(self, name, *args):
        if name in self.callbacks:
            self.callbacks[name](*args)

    # =========================
    # MODE
    # =========================
    def is_synthetic_data(self):
        """Compat publique : True si mode synthétique. À long terme, préférer self.mode."""
        return self.mode == DataMode.SYNTHETIC

    # =========================
    # START
    # =========================
    def start(self):
        self._emit("message", "Initializing run with the current configuration.")

        try:
            errors = self.validate_config()
            if errors:
                self._emit("message", f"⚠ Configuration incomplete ({len(errors)} issue(s)):")
                for err in errors:
                    self._emit("message", f"  - {err}")
                self._emit("error", "Cannot start: please fix the configuration above.")
                return

            self._set_state(PipelineState.LOADING)
            self.setup()

            self._set_state(PipelineState.COMPUTING)
            self._running = True
            self.run()

        except Exception as e:
            # Si setup() ou run() lèvent une exception (ex: load_tif fail, assert
            # nz==nz_true, etc.), on garantit que le DisplayWindow soit notifié
            # via le callback 'error' au lieu de rester figé indéfiniment.
            import traceback
            traceback.print_exc()
            self._running = False
            self._set_state(PipelineState.FAILED)
            self._emit("error", f"{type(e).__name__}: {e}")

    # =========================
    # SETUP
    # =========================
    def setup(self):
        """
        Charge les données + construit l'opérateur H + instancie l'algo.

        Délègue le calcul à des fonctions pures de pipeline_steps.py.
        Cette méthode ne fait que de l'orchestration : décider QUELLE étape
        appeler selon le mode, et stocker les résultats dans self.result.
        """
        # Import lazy pour éviter le cycle core <-> algorithms.
        from algorithms import ALGORITHMS

        # 1) Construction de g et H selon le mode
        if self.mode == DataMode.REAL:
            g, H, measurement_params = build_g_and_H_real(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = None
        else:  # DataMode.SYNTHETIC
            g, H, f_true = build_g_and_H_synthetic(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = f_true
            measurement_params, _, _ = load_measurement_inputs(self.config)

        # 2) Instanciation de l'algorithme avec les paramètres adéquats
        oper_params = self.config['oper-params']
        self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"](
            measurement_params, oper_params
        )

    # =========================
    # RUN
    # =========================
    def run(self):
        """
        Branche les callbacks de l'algorithme et démarre son thread.

        Cette méthode est NON-BLOQUANTE : elle retourne immédiatement.
        Le résultat arrive via le callback 'finished' une fois le thread terminé.
        """
        # branch callbacks to algorithm
        self.algorithm.callbacks = {
            "message": self._on_algo_message,
            "finished": self._on_algo_finished,
            "error": self._on_algo_error,
        }

        self.algorithm._run(self.result.g, self.result.H, self.config['algo-params'])

    def _on_algo_message(self, msg):
        self.result.messages += msg + "\n"
        self._emit("message", msg)

    def _on_algo_finished(self, f):
        self.result.f = f

        if self.mode == DataMode.SYNTHETIC:
            # Métriques de qualité (SSIM, FSC, etc.)
            metrics, delta = evaluate_synthetic_metrics(
                self.result.f, self.result.f_true, self.config
            )
            self.result.metrics = metrics
            self.result.delta = delta

            # Différence visualisable (utilisée par DifferenceViewer côté GUI).
            # On la calcule ici dans le pipeline plutôt que de laisser le
            # viewer GUI faire les calculs lui-même.
            diff, _alpha = compute_synthetic_difference(
                self.result.f, self.result.f_true, self.config
            )
            self.result.diff = diff

        self._running = False
        self._set_state(PipelineState.COMPLETED)
        self._emit("finished", self.result)

    def _on_algo_error(self, err):
        self._running = False
        self._set_state(PipelineState.FAILED)
        self._emit("error", err)

    # =========================
    # SAVE / LOAD
    # =========================
    # NOTE : la logique I/O fichier est dans core/result_io.py.
    # Ces méthodes restent ici comme façade pour ne pas casser les appels
    # existants depuis le GUI (DisplayWindow.save_reconstruction, etc.).
    def save_results(self, save_dir):
        from .result_io import save_reconstruction
        save_reconstruction(self.result, save_dir, self.config, self.mode)
        self._emit("message", f"Saved reconstruction in {save_dir}")

    def load_results(self, directory):
        from .result_io import load_reconstruction
        self.result = load_reconstruction(directory, self.mode)

        # Si mode synth, on recalcule diff + delta (ils ne sont pas sauvegardés
        # car recalculables depuis f et f_true). Ainsi DifferenceViewer pourra
        # consommer result.diff comme sur une run fraîche.
        if self.mode == DataMode.SYNTHETIC and self.result.f is not None and self.result.f_true is not None:
            diff, _alpha = compute_synthetic_difference(
                self.result.f, self.result.f_true, self.config
            )
            self.result.diff = diff
            if self.result.delta is None:
                _, delta = evaluate_synthetic_metrics(
                    self.result.f, self.result.f_true, self.config
                )
                self.result.delta = delta

        # Charger une reconstruction depuis disque = équivalent à un run réussi
        self._set_state(PipelineState.COMPLETED)

    # =========================
    # STOP
    # =========================
    def stop(self):
        if self.algorithm:
            self.algorithm.stop_running()
        self._running = False
        # Transition vers INTERRUPTED uniquement si on était dans un état actif
        # (évite de revenir d'un état COMPLETED/FAILED vers INTERRUPTED).
        if self.state in (PipelineState.LOADING, PipelineState.COMPUTING):
            self._set_state(PipelineState.INTERRUPTED)
            self._emit("message", "Algorithm interrupted by user.")

    # =========================
    # VALIDATE CONFIG
    # =========================
    def validate_config(self) -> list[str]:
        """
        Retourne la liste des erreurs de configuration.

        Liste vide => config valide.
        Liste non vide => l'utilisateur saura EXACTEMENT ce qui manque (au lieu
        d'un simple "Running failed" cryptique sur stdout).
        """
        errors = []
        cfg = self.config

        if cfg['algorithm'] == 'None':
            errors.append("Algorithm: not selected")
        if cfg['input-paths']['tif'] == 'None':
            errors.append("Input file (TIF): not provided")
        if cfg['input-paths']['json'] == 'None':
            errors.append("Measurement parameters file (JSON): not provided")
        if cfg['oper-params']['nz'] == 'None':
            errors.append("Operator parameter 'nz': not set")
        if cfg['oper-params']['z0'] == 'None':
            errors.append("Operator parameter 'z0': not set")
        if cfg['oper-params']['zN'] == 'None':
            errors.append("Operator parameter 'zN': not set")
        if cfg['add-noise']['add_noise'] and cfg['add-noise']['sigma'] == 'None':
            errors.append("Noise sigma: required when 'add noise' is enabled")

        return errors