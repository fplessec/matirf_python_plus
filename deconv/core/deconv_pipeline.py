"""
Orchestrateur principal d'une déconvolution 2D.

Même esprit que core/matirf_pipeline.py mais en simplifié :
    - pas de delta (anisotropie spécifique à MA-TIRF)
    - pas d'oper_params séparés (PSF entièrement décrite par le JSON)
    - I/O = PNG plutôt que TIF

Cette classe n'effectue elle-même AUCUN calcul lourd : elle se contente
d'orchestrer les fonctions pures de pipeline_steps.py et de gérer la machine
à états + les callbacks vers le GUI.
"""

from typing import Optional

from .enums import DataMode, PipelineState
from .deconv_result import DeconvResult
from .pipeline_steps import (
    build_g_and_H_real,
    build_g_and_H_synthetic,
    evaluate_synthetic_metrics,
    compute_synthetic_difference,
)


class DeconvPipeline:
    def __init__(self, config, callbacks=None):
        self.config = config
        self.callbacks = callbacks or {}

        # Mode parsé en enum typé
        self.mode = DataMode.from_config(config['input-paths']['mode'])

        # Tous les outputs sont encapsulés dans un seul objet typé.
        self.result: Optional[DeconvResult] = DeconvResult()

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
        """Compat publique : True si mode synthétique."""
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
            # On garantit que le DisplayWindow soit notifié via le callback
            # 'error' au lieu de rester figé indéfiniment si setup() ou run()
            # lèvent une exception.
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
        Charge les données + construit la PSF H + instancie l'algo.

        Délègue le calcul à des fonctions pures de pipeline_steps.py.
        Cette méthode ne fait que de l'orchestration : décider QUELLE étape
        appeler selon le mode, et stocker les résultats dans self.result.
        """
        # Import lazy pour éviter le cycle core <-> algorithms.
        from deconv.algorithms import ALGORITHMS

        # 1) Construction de g et H selon le mode
        if self.mode == DataMode.REAL:
            g, H = build_g_and_H_real(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = None
        else:  # DataMode.SYNTHETIC
            g, H, f_true = build_g_and_H_synthetic(self.config)
            self.result.g = g
            self.result.H = H
            self.result.f_true = f_true

        # 2) Instanciation de l'algorithme.
        # Note : en deconv 2D les algos n'ont pas besoin de measurement_params
        # ni d'oper_params puisque toute l'info (PSF) est déjà dans H.
        self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"]()

    # =========================
    # RUN
    # =========================
    def run(self):
        """
        Branche les callbacks de l'algorithme et démarre son thread.

        Cette méthode est NON-BLOQUANTE : elle retourne immédiatement.
        Le résultat arrive via le callback 'finished' une fois le thread terminé.
        """
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
            # Métriques de qualité (PSNR, SSIM, MSE)
            self.result.metrics = evaluate_synthetic_metrics(
                self.result.f, self.result.f_true
            )

            # Différence visualisable (utilisée par le DifferenceViewer côté GUI).
            self.result.diff = compute_synthetic_difference(
                self.result.f, self.result.f_true
            )

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
    # Ces méthodes restent ici comme façade pour le GUI (DisplayWindow.save_deconv,
    # etc.).
    def save_results(self, save_dir):
        from .result_io import save_deconv
        save_deconv(self.result, save_dir, self.config, self.mode)
        self._emit("message", f"Saved deconvolution in {save_dir}")

    def load_results(self, directory):
        from .result_io import load_deconv
        self.result = load_deconv(directory, self.mode)

        # Si mode synth, on recalcule diff (non sauvegardée car recalculable).
        if self.mode == DataMode.SYNTHETIC and self.result.f is not None and self.result.f_true is not None:
            self.result.diff = compute_synthetic_difference(
                self.result.f, self.result.f_true
            )

        # Charger un résultat depuis disque = équivalent à un run réussi
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
        Liste non vide => l'utilisateur saura EXACTEMENT ce qui manque.
        """
        errors = []
        cfg = self.config

        if cfg['algorithm'] == 'None':
            errors.append("Algorithm: not selected")
        if cfg['input-paths']['png'] == 'None':
            errors.append("Input file (PNG): not provided")
        if cfg['input-paths']['json'] == 'None':
            errors.append("PSF parameters file (JSON): not provided")
        if cfg['add-noise'].get('add_noise', False) and cfg['add-noise'].get('sigma', 'None') == 'None':
            errors.append("Noise sigma: required when 'add noise' is enabled")

        return errors
