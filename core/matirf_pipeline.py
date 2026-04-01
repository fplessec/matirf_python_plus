# import os
# from os import makedirs
# from os.path import join
#
# from algorithms import ALGORITHMS
# from core.reconstruction_metrics import compute_all_metrics
# from in_out import save_tif, save_toml, save_txt, save_csv, load_tif, load_json, load_txt, load_csv
# from operations import compute_matirf_operator_from_params, apply_matirf_operator
# from preprocess_measurement import preprocess_measurement_stack
#
#
# class MaTirfPipeline:
#     def __init__(self, config, callbacks=None):
#         self.config = config
#         self.callbacks = callbacks or {}
#         self.f = None
#         self.f_true = None
#         self.metrics = None
#         self.messages = ""
#         self.algorithm = None
#         self._running = False
#
#
#     def is_synthetic_data(self):
#         return self.config['input-paths']['mode'] == 'synthetic-data'
#
#     def _emit(self, name, *args):
#         if name in self.callbacks:
#             self.callbacks[name](*args)
#
#     def start(self):
#         print("Initializing Run with the current configuration.")
#
#         if not self.check_missing_parameters_in_config():
#             self._emit("message", "Running failed.\n")
#             return
#
#         self.setup()
#         self._running = True
#         self.run()
#
#     def setup(self):
#         tif_path = self.config['input-paths']['tif']
#         json_path = self.config['input-paths']['json']
#         measurement_params = load_json(json_path)
#         oper_params = self.config['oper-params']
#         add_noise_params = self.config['add-noise']
#         self.algo_params = self.config['algo-params']
#         if not self.is_synthetic_data():  # working with real MA-TIRF measurement
#             self.f_true = None
#             g = load_tif(tif_path)
#             self.g, measurement_params = preprocess_measurement_stack(g, measurement_params, add_noise_params)
#             self.H = compute_matirf_operator_from_params(measurement_params, oper_params)
#             self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"](measurement_params, oper_params)
#         else:  # working with synthetic data
#             self.f_true = load_tif(tif_path)
#             nz_true = self.f_true.shape[0]
#             nz = oper_params['nz']
#             assert nz == nz_true, (
#                 f"\nWhile computing the synthetic MA-TIRF measurement:\nThe parameter 'nz' in 'oper-params' (nz = "
#                 f"{nz}) must be equal to the number of plans in the synthetic truth ({nz_true})."
#             )
#             self.H = compute_matirf_operator_from_params(measurement_params, oper_params)
#             g = apply_matirf_operator(self.H, self.f_true)
#             self.g, _ = preprocess_measurement_stack(g, measurement_params, add_noise_params)
#             self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"](measurement_params, oper_params)
#
#     def run(self):
#         def on_message(msg):
#             self.messages += msg + "\n"
#             self._emit("message", msg)
#
#         def on_finished(result):
#             print("ON FINISHED IS CALLED")
#             self.f = result
#             if self.is_synthetic_data():
#                 self.metrics = compute_all_metrics(self.f, self.f_true)
#             self._running = False
#             self._emit("finished", result)
#
#         def on_error(err):
#             print("ON ERROR IS CALLED")
#             self._running = False
#             self._emit("error", err)
#
#         # branche les signaux
#         self.algorithm.signals.message.connect(on_message)
#         self.algorithm.signals.finished.connect(on_finished)
#         self.algorithm.signals.error.connect(on_error)
#
#         self.algorithm._run(self.g, self.H, self.algo_params)
#
#
#     def save_results(self, save_dir,):
#         makedirs(save_dir, exist_ok=True)
#
#         if self.f is None:  # minimal security
#             raise ValueError("Cannot save results: 'f' is None")
#
#         save_tif(self.f, join(save_dir, 'f.TIF'))
#         save_toml(self.config, join(save_dir, 'config.toml'))
#         messages = getattr(self, "messages", "")
#         save_txt(messages, join(save_dir, 'messages.txt'))
#
#         if self.is_synthetic_data():
#             if self.f_true is None:
#                 raise ValueError("Synthetic mode but 'f_true' is None")
#             save_tif(self.f_true, join(save_dir, 'f_true.TIF'))
#             # compute metrics only if not already available
#             if getattr(self, "metrics", None) is None:
#                 self.metrics = compute_all_metrics(self.f, self.f_true)
#             save_csv(self.metrics, join(save_dir, "recons_metrics.csv"))
#
#         self._emit("message", f"Saved reconstruction in {save_dir}")
#
#
#     def load_results(self, directory):
#         required_files = [
#             "config.toml",
#             "f.TIF",
#             "messages.txt"
#         ]
#         for filename in required_files:
#             path = join(directory, filename)
#             if not os.path.exists(path):
#                 raise FileNotFoundError(f"Missing required file: {filename}")
#
#         self.f = load_tif(join(directory, "f.TIF"))
#         self.messages = load_txt(join(directory, "messages.txt"))
#
#         if self.is_synthetic_data():
#             synthetic_files = [
#                 "f_true.TIF",
#                 "recons_metrics.csv"
#             ]
#             for filename in synthetic_files:
#                 path = join(directory, filename)
#                 if not os.path.exists(path):
#                     raise FileNotFoundError(f"Missing synthetic file: {filename}")
#             self.f_true = load_tif(join(directory, "f_true.TIF"))
#             self.metrics = load_csv(join(directory, "recons_metrics.csv"))
#         else:
#             self.f_true = None
#             self.metrics = None
#
#     def stop(self):
#         if self.algorithm:
#             self.algorithm.stop_running()
#         self._running = False
#
#
#     def check_missing_parameters_in_config(self):
#         is_any_parameter_missing = False
#         if self.config['algorithm'] == 'None':
#             print("Missing parameter to Run: no 'algorithm' selected.")
#             is_any_parameter_missing = True
#         if self.config['input-paths']['tif'] == 'None':
#             print("Missing parameter to Run: no '.tif file' selected in 'input-paths' section.")
#             is_any_parameter_missing = True
#         if self.config['input-paths']['json'] == 'None':
#             print("Missing parameter to Run: no '.json file' selected in 'input-paths' section.")
#             is_any_parameter_missing = True
#         if self.config['oper-params']['nz'] == 'None':
#             print("Missing parameter to Run: no 'nz' selected in 'oper-params' section.")
#             is_any_parameter_missing = True
#         if self.config['oper-params']['z0'] == 'None':
#             print("Missing parameter to Run: no 'z0' selected in 'oper-params' section.")
#             is_any_parameter_missing = True
#         if self.config['oper-params']['zN'] == 'None':
#             print("Missing parameter to Run: no 'zN' selected in 'oper-params' section.")
#             is_any_parameter_missing = True
#         if self.config['add-noise']['add_noise']:
#             if self.config['add-noise']['sigma'] == 'None':
#                 print("Missing parameter to Run: no 'sigma' selected in 'add-noise' section.")
#                 is_any_parameter_missing = True
#         return not is_any_parameter_missing  # if not is_any_parameter_missing == True, then you can run






import os
from os import makedirs
from os.path import join

from algorithms import ALGORITHMS
from core.reconstruction_metrics import compute_all_metrics
from in_out import (
    save_tif, save_toml, save_txt, save_csv,
    load_tif, load_json, load_txt, load_csv
)
from core.operations import compute_matirf_operator_from_params, apply_matirf_operator
from core.preprocess_measurement import preprocess_measurement_stack


class MaTirfPipeline:
    def __init__(self, config, callbacks=None):
        self.config = config
        self.callbacks = callbacks or {}

        self.f = None
        self.f_true = None
        self.metrics = None
        self.messages = ""

        self.algorithm = None
        self._running = False

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
        return self.config['input-paths']['mode'] == 'synthetic-data'

    # =========================
    # START
    # =========================
    def start(self):
        print("Initializing Run with the current configuration.")

        if not self.check_missing_parameters_in_config():
            self._emit("message", "Running failed.\n")
            return

        self.setup()
        self._running = True
        self.run()

    # =========================
    # SETUP
    # =========================
    def setup(self):
        tif_path = self.config['input-paths']['tif']
        json_path = self.config['input-paths']['json']

        measurement_params = load_json(json_path)
        oper_params = self.config['oper-params']
        add_noise_params = self.config['add-noise']
        self.algo_params = self.config['algo-params']

        if not self.is_synthetic_data():
            self.f_true = None

            g = load_tif(tif_path)
            self.g, measurement_params = preprocess_measurement_stack(
                g, measurement_params, add_noise_params
            )

            self.H = compute_matirf_operator_from_params(measurement_params, oper_params)

        else:
            self.f_true = load_tif(tif_path)

            nz_true = self.f_true.shape[0]
            nz = oper_params['nz']
            assert nz == nz_true

            self.H = compute_matirf_operator_from_params(measurement_params, oper_params)

            g = apply_matirf_operator(self.H, self.f_true)
            self.g, _ = preprocess_measurement_stack(
                g, measurement_params, add_noise_params
            )

        # 🔥 create algorithm
        self.algorithm = ALGORITHMS[self.config["algorithm"]]["object"](
            measurement_params, oper_params
        )

    # =========================
    # RUN
    # =========================
    def run(self):

        def on_message(msg):
            self.messages += msg + "\n"
            self._emit("message", msg)

        def on_finished(result):
            self.f = result

            if self.is_synthetic_data():
                self.metrics = compute_all_metrics(self.f, self.f_true)

            self._running = False
            self._emit("finished", result)

        def on_error(err):
            self._running = False
            self._emit("error", err)

        # 🔥 branch callbacks to algorithm
        self.algorithm.callbacks = {
            "message": on_message,
            "finished": on_finished,
            "error": on_error
        }

        self.algorithm._run(self.g, self.H, self.algo_params)

    # =========================
    # SAVE
    # =========================
    def save_results(self, save_dir):
        makedirs(save_dir, exist_ok=True)

        if self.f is None:
            raise ValueError("Cannot save results: 'f' is None")

        save_tif(self.f, join(save_dir, 'f.TIF'))
        save_toml(self.config, join(save_dir, 'config.toml'))
        save_txt(self.messages, join(save_dir, 'messages.txt'))

        if self.is_synthetic_data():
            if self.f_true is None:
                raise ValueError("Synthetic mode but 'f_true' is None")

            save_tif(self.f_true, join(save_dir, 'f_true.TIF'))

            if self.metrics is None:
                self.metrics = compute_all_metrics(self.f, self.f_true)

            save_csv(self.metrics, join(save_dir, "recons_metrics.csv"))

        self._emit("message", f"Saved reconstruction in {save_dir}")

    # =========================
    # LOAD
    # =========================
    def load_results(self, directory):
        required = ["config.toml", "f.TIF", "messages.txt"]

        for f in required:
            if not os.path.exists(join(directory, f)):
                raise FileNotFoundError(f)

        self.f = load_tif(join(directory, "f.TIF"))
        self.messages = load_txt(join(directory, "messages.txt"))

        if self.is_synthetic_data():
            self.f_true = load_tif(join(directory, "f_true.TIF"))
            self.metrics = load_csv(join(directory, "recons_metrics.csv"))
        else:
            self.f_true = None
            self.metrics = None

    # =========================
    # STOP
    # =========================
    def stop(self):
        if self.algorithm:
            self.algorithm.stop_running()
        self._running = False

    # =========================
    # CHECK CONFIG
    # =========================
    def check_missing_parameters_in_config(self):
        is_missing = False

        if self.config['algorithm'] == 'None':
            print("Missing algorithm")
            is_missing = True

        if self.config['input-paths']['tif'] == 'None':
            print("Missing tif")
            is_missing = True

        if self.config['input-paths']['json'] == 'None':
            print("Missing json")
            is_missing = True

        if self.config['oper-params']['nz'] == 'None':
            is_missing = True

        if self.config['oper-params']['z0'] == 'None':
            is_missing = True

        if self.config['oper-params']['zN'] == 'None':
            is_missing = True

        if self.config['add-noise']['add_noise']:
            if self.config['add-noise']['sigma'] == 'None':
                is_missing = True

        return not is_missing