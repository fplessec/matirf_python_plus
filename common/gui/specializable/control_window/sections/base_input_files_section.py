"""
Base input-files section for any inverse problem — fully declarative.

A concrete section is defined by class attributes only: the config/cache handles, the mode
labels, and TWO `FileSlot` specs (IMAGE_SLOT + JSON_SLOT) describing each file. From those
specs the base builds the two FileSelectors, wires their secondary buttons (a preprocessing
preview for the image slot, a parameters editor for the json slot), the optional JSON
validator, and the real/synthetic mode toggle. No factory functions or callbacks needed.

    FileSlot(toml_key, noun, dialog_filter, title_real, title_synthetic,
             require_keys=None, preview=None, editor=None)
    Preview(viewer_class, compute, titles, size=(800,900), error_check=None)
        -> the "See preprocessed file" button (image slot)
    Editor(ui=None, title_noun="Parameters", width=600, editor_class=None)
        -> the "Create/Modify .json" button (json slot); editor_class overrides the
           generic JsonParametersEditor for problems that need extra fields (e.g. angles)

Declarative configuration (override as class attributes):
    CONFIG_PATH, DEFAULT_CONFIG, UPDATE_CACHE_FN, MEASUREMENTS_DIR
    REAL_MODE_TEXT, SYNTHETIC_MODE_TEXT, DEFAULT_MODE_REAL
    IMAGE_SLOT, JSON_SLOT
"""

from dataclasses import dataclass
from typing import Callable, Iterable

from PyQt5.QtGui import QPalette
from PyQt5.QtWidgets import QGroupBox, QVBoxLayout, QHBoxLayout, QLabel, QMessageBox

from common.gui.widgets import QSwitchButton, QSeparator
from common.gui.reusable import FileSelector, SelectorButton
from common import DataMode
from common.in_out import load_or_create_toml, load_json
from common.settings import FontSize


@dataclass
class Preview:
    """
    The image slot's "See preprocessed file" popup spec.

    >> viewer_class : type        image viewer for each side (ImageAndHisto2D/3DViewer)
    >> compute      : callable    (config, mode) -> (left_image, right_image)
                                  usually <Problem>Operations.compute_preprocessing_preview
    >> titles       : dict        {'real': (left, right), 'synthetic': (left, right)}
    >> size         : (w, h)      popup size (default 800x900)
    >> error_check  : callable    optional (config) -> list[str]: messages listing the
                                  missing parameters when the preview cannot be computed
    """
    viewer_class: type
    compute: Callable
    titles: dict
    size: tuple = (800, 900)
    error_check: Callable = None


@dataclass
class Editor:
    """
    The json slot's "Create/Modify .json" editor spec.

    >> ui           : dict     params UI dict for the generic JsonParametersEditor
                               (SimpleParameterWidget format); ignored if editor_class is set
    >> title_noun   : str      window / dialog noun, e.g. "PSF Parameters"
    >> width        : int      editor window width
    >> editor_class : type     optional custom editor (called as editor_class(parent=selector))
                               for problems whose editor has extra fields (e.g. matirf angles)
    """
    ui: dict = None
    title_noun: str = "Parameters"
    width: int = 600
    editor_class: type = None


@dataclass
class FileSlot:
    """
    Declarative description of ONE file input (the image slot or the json slot).

    >> toml_key        : str       key under '[input-paths]' where the path is stored ('tif'/'png'/'json')
    >> noun            : str       short file kind for the UI ('tif' / 'png' / 'json')
    >> dialog_filter   : str       Qt open-dialog filter, e.g. "Image Files (*.png)"
    >> title_real      : str       title shown in real-data mode
    >> title_synthetic : str       title shown in synthetic-data mode
    >> require_keys    : iterable  optional: validate the chosen .json contains these keys
    >> preview         : Preview   optional: adds the "See preprocessed file" button (image slot)
    >> editor          : Editor    optional: adds the "Create/Modify .json" button (json slot)
    """
    toml_key: str
    noun: str
    dialog_filter: str
    title_real: str
    title_synthetic: str
    require_keys: Iterable = None
    preview: Preview = None
    editor: Editor = None


class BaseInputFilesSection(QGroupBox):
    """
    Input-files section of a control window — declared entirely by class attributes.

    --------
    > Layout :
    --------
        [ real  <switch>  synthetic ]                    (mode toggle)
        [ image FileSelector | json FileSelector ]

    ----------
    > Parameters (override as class attributes) :
    ----------

    >> CONFIG_PATH / DEFAULT_CONFIG : Path / dict
        The cached config.toml and its fallback content.

    >> UPDATE_CACHE_FN : callable
        Function(key_path, value) writing into the cached config. Wrap a module-level
        function with staticmethod() so it is NOT bound as a method:
            UPDATE_CACHE_FN = staticmethod(update_cache)

    >> MEASUREMENTS_DIR : Path
        Directory the file dialogs open in.

    >> REAL_MODE_TEXT / SYNTHETIC_MODE_TEXT : str
        Labels of the two modes shown next to the toggle.

    >> DEFAULT_MODE_REAL : bool
        Mode used when the config has none yet (True = real).

    >> IMAGE_SLOT / JSON_SLOT : FileSlot
        The two file inputs (see FileSlot / Preview / Editor above). The base builds a
        FileSelector from each, wires the preview / editor secondary buttons, the JSON
        validator (require_keys), and the mode toggle. No factory or callback needed.

    ----------
    > Example :
    ----------

        class InputFilesSection(BaseInputFilesSection):
            CONFIG_PATH = MATIRF_CONFIG_PATH
            DEFAULT_CONFIG = DEFAULT_MATIRF_CONFIG
            UPDATE_CACHE_FN = staticmethod(update_cache)
            MEASUREMENTS_DIR = MATIRF_MEASUREMENTS_DIR
            REAL_MODE_TEXT = "Work with real MA-TIRF measurement"
            SYNTHETIC_MODE_TEXT = "Simulate measurement with synthetic truth"
            DEFAULT_MODE_REAL = True

            IMAGE_SLOT = FileSlot(
                toml_key="tif", noun="tif", dialog_filter="Image Files (*.tif *.tiff)",
                title_real="Path of the MA-TIRF image stack",
                title_synthetic="Path of the 3D object (synthetic truth)",
                preview=Preview(ImageAndHisto3DViewer, compute_preview_fn,
                                titles=PREVIEW_TITLES, size=(700, 900), error_check=errors_fn),
            )
            JSON_SLOT = FileSlot(
                toml_key="json", noun="json", dialog_filter="Parameters Files (*.json)",
                title_real="Path of the measurement parameters",
                title_synthetic="Path of the simulated parameters",
                require_keys=MEASUREMENT_PARAMETERS_UI,               # auto .json validator
                editor=Editor(editor_class=MeasurementParametersEditor),  # or Editor(ui=..., title_noun=...)
            )
    """

    # ── declarative configuration (override these class attributes) ──────────
    CONFIG_PATH = None
    DEFAULT_CONFIG = None
    UPDATE_CACHE_FN = None
    MEASUREMENTS_DIR = None
    REAL_MODE_TEXT = ""
    SYNTHETIC_MODE_TEXT = ""
    DEFAULT_MODE_REAL = True
    IMAGE_SLOT = None     # FileSlot
    JSON_SLOT = None      # FileSlot

    def __init__(self, parent=None):
        super().__init__("Input Files")
        self.on_color = self.palette().color(QPalette.WindowText).name()
        self.off_color = self.palette().color(QPalette.PlaceholderText).name()
        self.parent = parent
        self.is_mode_real = self._get_cached_mode()
        self._setup_ui()

    # ── selectors built from the slot specs ──────────────────────────────────

    def _create_image_selector(self):
        return self._build_selector(self.IMAGE_SLOT)

    def _create_json_selector(self):
        return self._build_selector(self.JSON_SLOT)

    def _build_selector(self, slot):
        extra = None
        if slot.preview is not None:
            extra = SelectorButton(
                text="See preprocessed file",
                on_click=lambda selector, p=slot.preview: self._open_preview(selector, p),
                visible_when=self.are_both_file_selected)   # shown only when both files selected
        elif slot.editor is not None:
            extra = SelectorButton(
                text=lambda selected: "Modify .json file" if selected else "Create .json file",
                on_click=lambda selector, e=slot.editor: self._open_editor(selector, e))
        validate = self._make_validator(slot.require_keys) if slot.require_keys else None
        return FileSelector(
            self,
            noun=slot.noun, dialog_filter=slot.dialog_filter, toml_key=slot.toml_key,
            measurements_dir=self.MEASUREMENTS_DIR, update_cache_fn=self.UPDATE_CACHE_FN,
            title_real=slot.title_real, title_synthetic=slot.title_synthetic,
            validate_fn=validate, extra_button=extra,
        )

    @staticmethod
    def _make_validator(require_keys):
        def validate(filepath):
            data = load_json(filepath)
            missing = [k for k in require_keys if k not in data]
            if missing:
                print(f"Failed to select '{filepath}': invalid file, missing required key(s): "
                      f"{', '.join(missing)}")
            return not missing
        return validate

    # ── secondary-button actions (preview / editor), driven by the specs ─────

    def _open_preview(self, selector, preview):
        from common.gui.reusable import PreprocessViewer
        try:
            viewer = PreprocessViewer(
                selector, config_path=self.CONFIG_PATH, default_config=self.DEFAULT_CONFIG,
                compute_preview=preview.compute, viewer_class=preview.viewer_class,
                titles=preview.titles, size=preview.size)
        except AssertionError as e:
            QMessageBox.warning(selector, "Cannot preview the preprocessing",
                                f"The measurement files are inconsistent:\n{e}")
            return
        except Exception as e:
            config = load_or_create_toml(self.CONFIG_PATH, default_config=self.DEFAULT_CONFIG)
            msgs = preview.error_check(config) if preview.error_check else []
            detail = ("\n- " + "\n- ".join(msgs)) if msgs else f"{type(e).__name__}: {e}"
            QMessageBox.warning(selector, "Cannot preview the preprocessing",
                                f"Cannot compute the preview:\n{detail}")
            return
        selector.set_sub_window(viewer)

    def _open_editor(self, selector, editor):
        if editor.editor_class is not None:
            selector.set_sub_window(editor.editor_class(parent=selector))
        else:
            from common.gui.reusable import JsonParametersEditor
            selector.set_sub_window(JsonParametersEditor(
                selector, params_ui_dict=editor.ui, measurements_dir=self.MEASUREMENTS_DIR,
                title_noun=editor.title_noun, width=editor.width))

    # ── mode / config plumbing (built from the declarative attributes) ───────

    def _get_cached_mode(self) -> bool:
        try:
            mode = load_or_create_toml(self.CONFIG_PATH, default_config=self.DEFAULT_CONFIG)['input-paths']['mode']
            return mode == DataMode.REAL.value
        except (KeyError, FileNotFoundError):
            return self.DEFAULT_MODE_REAL

    def _real_mode_text(self):
        return self.REAL_MODE_TEXT

    def _synthetic_mode_text(self):
        return self.SYNTHETIC_MODE_TEXT

    def _on_switch_mode(self, is_mode_real):
        self.UPDATE_CACHE_FN(['input-paths', 'mode'],
                             DataMode.REAL.value if is_mode_real else DataMode.SYNTHETIC.value)

    def _load_config_for_update(self, toml_path):
        return load_or_create_toml(toml_path, default_config=self.DEFAULT_CONFIG)

    def _get_file_paths_from_config(self, config):
        return (config['input-paths'][self.IMAGE_SLOT.toml_key],
                config['input-paths'][self.JSON_SLOT.toml_key])

    # ── shared UI construction ──────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(1, 1, 1, 1)
        layout.addLayout(self._create_top_layout())
        layout.addLayout(self._create_bottom_layout())
        self.setLayout(layout)

    def _create_top_layout(self):
        top = QHBoxLayout()
        top.setContentsMargins(4, 4, 4, 4)
        self.switch_button = QSwitchButton()
        if not self.is_mode_real:
            self.switch_button.switch_to_right()
        self.switch_button.toggled.connect(self._switch_mode)
        self.real_label = QLabel(self._real_mode_text())
        self.synth_label = QLabel(self._synthetic_mode_text())
        self._update_labels()
        top.addStretch()
        top.addWidget(self.real_label)
        top.addWidget(self.switch_button)
        top.addWidget(self.synth_label)
        top.addStretch()
        return top

    def _create_bottom_layout(self):
        bot = QHBoxLayout()
        bot.setContentsMargins(1, 1, 1, 1)
        self.image_selector = self._create_image_selector()
        self.json_selector = self._create_json_selector()
        bot.addWidget(self.image_selector)
        bot.addWidget(QSeparator('V'))
        bot.addWidget(self.json_selector)
        # both selectors now exist: set the initial state-dependent button states
        self.notify_selection_changed()
        return bot

    # ── shared logic ────────────────────────────────────────────────────

    def _update_labels(self):
        on = f"color: {self.on_color}; font-style: italic; font-size: {FontSize.SMALL}pt;"
        off = f"color: {self.off_color}; font-style: italic; font-size: {FontSize.SMALL}pt;"
        self.real_label.setStyleSheet(on if self.is_mode_real else off)
        self.synth_label.setStyleSheet(off if self.is_mode_real else on)

    def _switch_mode(self):
        self.is_mode_real = not self.is_mode_real
        self._update_labels()
        self._on_switch_mode(self.is_mode_real)
        self.image_selector.update_mode()
        self.json_selector.update_mode()
        self.notify_selection_changed()

    def are_both_file_selected(self):
        return self.image_selector.is_file_selected and self.json_selector.is_file_selected

    ## re-evaluates every selector's state-dependent secondary button (text + visibility),
    ## so cross-selector dependencies (e.g. a preview button shown only when BOTH files are
    ## selected) stay consistent without selectors reaching into each other's widgets.
    def notify_selection_changed(self):
        for selector in (self.image_selector, self.json_selector):
            if hasattr(selector, 'refresh_extra_button'):
                selector.refresh_extra_button()

    def update_ui_from_toml(self, toml_path):
        config = self._load_config_for_update(toml_path)
        mode = config['input-paths']['mode']
        self.is_mode_real = (mode == DataMode.REAL.value)
        self._update_labels()
        self.image_selector.update_mode()
        self.json_selector.update_mode()
        if self.is_mode_real:
            self.switch_button.switch_to_left(no_signal=True)
        else:
            self.switch_button.switch_to_right(no_signal=True)
        image_path, json_path = self._get_file_paths_from_config(config)
        self.image_selector.update_selected_file(image_path)
        self.json_selector.update_selected_file(json_path)
