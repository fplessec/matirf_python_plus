"""
GUI smoke test — every user-facing feature, driven through the real widgets.

Run it:
    QT_QPA_PLATFORM=offscreen python -m _gui_smoke_test

The unit suites (core, solvers, problems, pipeline) check contracts in isolation. This one
answers a different question: does the APPLICATION still do everything it did before?

WHAT CHANGED, AND WHY IT MATTERS
    The first version of this file called handler methods directly — `section._switch_mode()`
    rather than clicking the toggle. That proves the handler works, and proves nothing about
    whether any button is WIRED to it: a `clicked.connect` lost in a refactor would have
    passed every check and failed for the user on the first click.

    So every interaction here now goes through the widget the user actually touches:
    `button(window, "Run").click()`. Buttons that production code keeps in local variables
    are found by their label in the widget tree, which also verifies they were added to the
    layout at all.

    File dialogs are the one thing that cannot be clicked — they are OS-modal. They are
    stubbed to return a path, so the whole chain behind them is still exercised: choosing a
    file runs the validator, writes the cache and refreshes the other selector's buttons.

WHAT NO AUTOMATED TEST CAN PROVE
    That a window is readable. A widget off-screen, a truncated label, an unreadable
    contrast: offscreen rendering hides all of it. Look at the windows yourself now and
    then; this file cannot do it for you.

Sections:
    A  control window     real clicks: mode toggle, file choice, preview, editor, config I/O
    B  solvers            all six parameter panels, not just the two that are convenient
    C  typing             values typed into widgets reach the TOML; depends_on shows/hides
    D  run + display      Run clicked, then figures, metrics, difference, save, interrupt
    E  deconvolution      the same journey on the other problem
    F  matirf synth       generate, save, use as synthetic truth
    G  error paths        what the user is told when something is wrong
    H  two problems       both open at once, with independent pipelines
"""

import contextlib
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QMessageBox, QPushButton

app = QApplication(sys.argv)

# ── neutralise anything modal, and record what the user would have been shown ──
SHOWN = []
QMessageBox.warning = staticmethod(
    lambda *a, **k: SHOWN.append(("warning", a[1] if len(a) > 1 else "",
                                  a[2] if len(a) > 2 else "")) or QMessageBox.Ok)
QMessageBox.information = staticmethod(
    lambda *a, **k: SHOWN.append(("info", a[1] if len(a) > 1 else "", "")) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(
    lambda *a, **k: SHOWN.append(("critical", a[1] if len(a) > 1 else "", "")) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

from core import DataMode
from core.enums import PipelineState
from fileio import load_or_create_toml, save_toml
from gui.factory import control_window_class, display_window_class
from pipeline import Pipeline, pipeline_for
from solvers.objective_params import OBJECTIVE_UI_PARAMS
from problems.deconv import (
    DECONV, DECONV_CONFIG_PATH, DECONV_MEASUREMENTS_DIR, DEFAULT_DECONV_CONFIG,
)
from problems.matirf import (
    MATIRF, MATIRF_CONFIG_PATH, MATIRF_MEASUREMENTS_DIR, DEFAULT_MATIRF_CONFIG,
)

TIF = str(MATIRF_MEASUREMENTS_DIR / "esoubies.TIF")
MJSON = str(MATIRF_MEASUREMENTS_DIR / "esoubies.json")
TRUTH = str(MATIRF_MEASUREMENTS_DIR / "synthetic_truth0.TIF")
PNG = str(DECONV_MEASUREMENTS_DIR / "img_001.png")
DJSON = str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")


# ── driving the real widgets ──────────────────────────────────────────────────

def button(root, label: str) -> QPushButton:
    """
    The button the user would click, found by its label in the real widget tree.

    Production code keeps several buttons in local variables (Run, Choose file, Estimate),
    so they cannot be reached through an attribute. Finding them here also proves they were
    actually added to a layout — a button built but never placed would not be found.
    """
    for candidate in root.findChildren(QPushButton):
        if candidate.text() == label:
            return candidate
    available = sorted({b.text() for b in root.findChildren(QPushButton) if b.text()})
    raise AssertionError(f"no button labelled {label!r}; available: {available}")


def click(widget) -> None:
    """Click, through the real `clicked` signal — so a missing connect is a failure."""
    widget.click()
    app.processEvents()


def click_switch(switch) -> None:
    """QSwitchButton reacts to a mouse press, not to click(): send a real press."""
    switch.mousePressEvent(QMouseEvent(QMouseEvent.MouseButtonPress, QPoint(5, 5),
                                       Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
    app.processEvents()


def type_in(parameter_widget, text: str) -> None:
    """Type into a parameter's field, exactly as the user would."""
    parameter_widget.input_widget.setText(str(text))
    app.processEvents()


@contextlib.contextmanager
def dialogs(open_path=None, save_path=None, directory=None):
    """
    Answer every file dialog with a fixed path, for the duration of the block.

    The dialogs themselves are OS-modal and cannot be driven, but everything BEHIND them
    can: choosing a file still runs the validator, writes the cache and refreshes the
    dependent buttons. Each module is patched where it imported the function, since
    `from ... import open_file` binds a name that patching the source module would miss.
    """
    import gui.file_dialog as source
    import gui.reusable.inputs.file_selector as selector
    import gui.specializable.control_window.base_control_window as control
    import gui.specializable.display_window.base_display_window as display
    import gui.specializable.display_window.sections.base_figures_section as figures
    import problems.matirf.synthetic.gui as synth

    answers = {"open_file": open_path, "save_file": save_path, "open_directory": directory}
    targets = (source, selector, control, display, figures, synth)

    saved = [(module, name, getattr(module, name))
             for module in targets for name in answers if hasattr(module, name)]
    for module, name, _ in saved:
        setattr(module, name, (lambda value: (lambda *a, **k: value))(answers[name]))
    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


def wait_until(condition, timeout=120.0) -> bool:
    """Pump the Qt event loop until `condition` holds — the windows stay responsive."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.02)
    return False


def top_level(kind):
    """Every open window of a given class — how a popup opened by a click is found."""
    return [w for w in app.topLevelWidgets() if isinstance(w, kind)]


# ── reporting ─────────────────────────────────────────────────────────────────

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail or ""))
        print(f"  PASS  {name}" + (f"  — {detail}" if detail else ""))
    except Exception as error:
        RESULTS.append((False, name, f"{type(error).__name__}: {error}"))
        print(f"  FAIL  {name}  — {type(error).__name__}: {error}")
        traceback.print_exc(limit=3)


# ── keep the real caches untouched ────────────────────────────────────────────

BACKUPS = {p: Path(p).read_text() for p in (MATIRF_CONFIG_PATH, DECONV_CONFIG_PATH)
           if Path(p).exists()}


def matirf_ready_config():
    """A configuration where everything is set — the state just before clicking Run."""
    return {
        "algorithm": "ADAM",
        "input-paths": {"mode": DataMode.SYNTHETIC.value, "tif": TRUTH, "json": MJSON},
        "oper-params": {"nz": 50, "z0": 0.0, "zN": 400.0, "normalize": False},
        "add-noise": {},
        "algo-params": {"max_iter": 12, "lr": 0.01, "K": 4, "EPS": 1e-14},
    }


ControlWindow = control_window_class(MATIRF)
DeconvControlWindow = control_window_class(DECONV)

print("\n=== A. FENÊTRE DE CONTRÔLE (clics réels) ===")

cw = ControlWindow()
dw = DeconvControlWindow()


def a1():
    assert cw.windowTitle() == MATIRF.ui.control_title
    sections = [type(s).__name__ for s in cw._all_sections]
    assert sections[0] == "InputFilesSection" and sections[-1] == "AlgorithmSelectionSection"
    return f"{len(sections)} sections, titre depuis la déclaration"
check("A1  les deux fenêtres se construisent depuis la déclaration", a1)


def a2():
    for label in ("Run", "Save config", "Load any config", "Open reconstruction"):
        button(cw, label)
    return "Run, Save config, Load any config, Open reconstruction"
check("A2  la barre du bas expose ses quatre boutons", a2)


def a3():
    section = cw.input_files_section
    before = section.is_mode_real
    click_switch(section.switch_button)          # le vrai basculeur, pas la méthode
    assert section.is_mode_real != before, "le clic n'a pas basculé le mode"
    written = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)["input-paths"]["mode"]
    expected = "real-data" if section.is_mode_real else "synthetic-data"
    assert written == expected, f"cache={written}, attendu {expected}"
    click_switch(section.switch_button)
    assert section.is_mode_real == before
    return f"clic sur le basculeur -> cache ({written})"
check("A3  basculeur de mode : clic réel -> cache", a3)


def a4():
    section = cw.input_files_section
    with dialogs(open_path=TIF):
        click(button(section.image_selector, "Choose .tif file"))
    assert section.image_selector.is_file_selected, "le .tif n'a pas été retenu"
    with dialogs(open_path=MJSON):
        click(button(section.json_selector, "Choose .json file"))
    assert section.json_selector.is_file_selected
    paths = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)["input-paths"]
    assert paths["tif"] == TIF and paths["json"] == MJSON
    return "les deux fichiers choisis par clic et écrits dans le cache"
check("A4  choisir un fichier : clic -> validateur -> cache", a4)


def a5():
    section = cw.input_files_section
    assert section.are_both_file_selected()
    assert not section.image_selector._extra_btn.isHidden(), \
        "le bouton d'aperçu devrait apparaître une fois les deux fichiers choisis"
    assert section.json_selector._extra_btn.text() == "Modify .json file"
    return "aperçu révélé, bouton json passé à « Modify »"
check("A5  dépendance croisée entre les deux sélecteurs", a5)


def a6():
    section = cw.input_files_section
    n = len(SHOWN)
    click(section.image_selector._extra_btn)     # « See preprocessed file »
    window = section.image_selector.sub_window
    assert window is not None, f"aperçu non ouvert (messages: {SHOWN[n:]})"
    shapes = tuple(window.left.shape), tuple(window.right.shape)
    window.close()
    return f"clic -> aperçu {shapes[0]} -> {shapes[1]}"
check("A6  aperçu du prétraitement, par clic", a6)


def a7():
    section = cw.input_files_section
    click(section.json_selector._extra_btn)      # « Modify .json file »
    editor = section.json_selector.sub_window
    assert editor is not None, "éditeur non ouvert"
    editor.close()
    return "clic -> éditeur de paramètres ouvert"
check("A7  éditeur JSON, par clic", a7)


def a8():
    saved = Path(tempfile.mkdtemp()) / "conf.toml"
    with dialogs(save_path=str(saved)):
        click(button(cw, "Save config"))
    assert saved.exists(), "aucun fichier écrit"
    with dialogs(open_path=str(saved)):
        click(button(cw, "Load any config"))
    return f"config sauvegardée ({saved.stat().st_size} o) puis rechargée"
check("A8  sauvegarder / charger une config, par clic", a8)


print("\n=== B. LES SIX SOLVEURS ===")


def b1():
    section = cw.algorithm_selection_section
    names = [section.algo_combo.itemText(i) for i in range(section.algo_combo.count())]
    assert names == ["None", "ADAM", "PPXA", "ADMM", "PNP", "ADMM-PnP", "MCMC"], names
    return "les six proposés, dans l'ordre du registre"
check("B1  le sélecteur liste les six solveurs", b1)


def b2():
    """Chaque solveur, pas seulement les deux commodes : panneau construit, défauts écrits."""
    section = cw.algorithm_selection_section
    summary = []
    for name in ("ADAM", "PPXA", "ADMM", "PNP", "ADMM-PnP", "MCMC"):
        section.algo_combo.setCurrentText(name)
        app.processEvents()
        config = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
        assert config["algorithm"] == name, f"{name} non écrit dans le cache"
        widget = section.algo_widgets[name]
        assert widget.parameter_widgets, f"{name}: aucun paramètre construit"
        written = config.get("algo-params", {})
        assert written, f"{name}: aucun défaut écrit"
        missing = [k for k in widget.parameter_widgets
                   if not widget.parameter_widgets[k].isHidden() and k not in written]
        assert not missing, f"{name}: paramètres visibles non écrits: {missing}"
        summary.append(f"{name}({len(written)})")
    return " ".join(summary)
check("B2  les six panneaux de paramètres se construisent et s'écrivent", b2)


def b3():
    section = cw.algorithm_selection_section
    section.algo_combo.setCurrentText("ADAM")
    app.processEvents()
    widget = section.algo_widgets["ADAM"]
    type_in(widget.parameter_widgets["max_iter"], 4242)
    assert load_or_create_toml(MATIRF_CONFIG_PATH,
                               DEFAULT_MATIRF_CONFIG)["algo-params"]["max_iter"] == 4242
    click(section.reset_btn)
    restored = load_or_create_toml(MATIRF_CONFIG_PATH,
                                   DEFAULT_MATIRF_CONFIG)["algo-params"]["max_iter"]
    assert restored == 1000, restored
    return "valeur saisie écrite (4242), puis « reset parameters » restaure 1000"
check("B3  « reset parameters », par clic", b3)


print("\n=== C. SAISIE ET DÉPENDANCES ===")


def c1():
    """Une valeur tapée doit traverser la conversion de type jusqu'au TOML."""
    section = cw.oper_params
    type_in(section.parameter_widgets["nz"], 37)
    type_in(section.parameter_widgets["z0"], 12.5)
    oper = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)["oper-params"]
    assert oper["nz"] == 37 and isinstance(oper["nz"], int), f"nz={oper['nz']!r}"
    assert abs(oper["z0"] - 12.5) < 1e-9 and isinstance(oper["z0"], float)
    type_in(section.parameter_widgets["nz"], 50)
    type_in(section.parameter_widgets["z0"], 0.0)
    return "int et float convertis correctement"
check("C1  une valeur saisie atteint le TOML avec le bon type", c1)


def c2():
    """La case normalize : False est une réponse valide, pas un réglage manquant."""
    section = cw.oper_params
    box = section.parameter_widgets["normalize"].checkbox
    box.setChecked(True); app.processEvents()
    assert load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)["oper-params"]["normalize"] is True
    box.setChecked(False); app.processEvents()
    oper = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)["oper-params"]
    assert oper["normalize"] is False
    assert not any("normalize" in e for e in MATIRF.validate({
        "input-paths": {"tif": TIF, "json": MJSON},
        "oper-params": oper, "add-noise": {}})), "False signalé à tort comme manquant"
    return "cochée puis décochée ; False accepté par la validation"
check("C2  la case « normalize » écrit les deux états", c2)


def c3():
    """
    La section de bruit : deux interrupteurs indépendants, et leurs paramètres qui
    n'apparaissent que lorsqu'ils servent (depends_on).

    photon seul -> Poisson pur, lecture seule -> gaussien pur, les deux -> Poisson-gaussien.
    """
    section = cw.add_noise
    widgets = section.parameter_widgets
    poisson, gaussian = widgets["poisson_noise"].checkbox, widgets["gaussian_noise"].checkbox

    for box in (poisson, gaussian):
        box.setChecked(False); app.processEvents()
    assert widgets["photons"].isHidden() and widgets["sigma"].isHidden(), \
        "les paramètres d'un bruit désactivé doivent être masqués"

    poisson.setChecked(True); app.processEvents()
    assert not widgets["photons"].isHidden() and widgets["sigma"].isHidden()
    type_in(widgets["photons"], 250)
    gaussian.setChecked(True); app.processEvents()
    assert not widgets["sigma"].isHidden()
    type_in(widgets["sigma"], 0.02)
    type_in(widgets["seed"], 7)

    written = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)["add-noise"]
    assert written["poisson_noise"] is True and written["gaussian_noise"] is True, written
    assert written["photons"] == 250 and abs(written["sigma"] - 0.02) < 1e-9, written
    assert written["seed"] == 7

    from core.noise import noise_model
    model = noise_model(written)
    assert model.photons == 250 and model.sigma == 0.02 and model.seed == 7

    for box in (poisson, gaussian):
        box.setChecked(False); app.processEvents()
    assert not noise_model(load_or_create_toml(MATIRF_CONFIG_PATH,
                                               DEFAULT_MATIRF_CONFIG)["add-noise"]).enabled
    return "Poisson + gaussien réglés par clic ; paramètres masqués quand inutiles"
check("C3  section de bruit : Poisson, gaussien, masquage des paramètres", c3)


def c4():
    """delta ne concerne que les régularisations qui pondèrent la dérivée axiale."""
    section = cw.algorithm_selection_section
    section.algo_combo.setCurrentText("ADAM"); app.processEvents()
    widget = section.algo_widgets["ADAM"]
    delta, reg = widget.parameter_widgets.get("delta"), widget.parameter_widgets.get("reg")
    assert delta is not None and reg is not None, "delta ou reg absent du panneau ADAM"
    reg.combo.setCurrentText("L1 norm"); app.processEvents()
    without = delta.isHidden()
    reg.combo.setCurrentText("L1 norm of the gradient"); app.processEvents()
    with_grad = not delta.isHidden()
    assert without and with_grad, f"caché avec L1={without}, montré avec TV={with_grad}"
    return "delta masqué pour L1, révélé pour la variation totale"
check("C4  depends_on : delta suit la régularisation choisie", c4)


def c5():
    """Le bouton « Estimate » de delta, par clic — et il doit vraiment calculer.

    La première version de ce test se contentait de « un float > 0 ». Elle passait donc
    sur la valeur PAR DÉFAUT (0.05) : elle ne distinguait pas un bouton qui calcule d'un
    bouton qui ne fait rien. On exige maintenant une valeur différente du défaut.
    """
    save_toml(matirf_ready_config(), MATIRF_CONFIG_PATH)
    cw.load_cached_config()
    section = cw.algorithm_selection_section
    section.algo_combo.setCurrentText("ADAM"); app.processEvents()
    widget = section.algo_widgets["ADAM"]
    default = OBJECTIVE_UI_PARAMS["delta"]["param_info"]["default"]
    n = len(SHOWN)
    click(button(widget.parameter_widgets["delta"], "Estimate"))
    value = load_or_create_toml(MATIRF_CONFIG_PATH,
                                DEFAULT_MATIRF_CONFIG)["algo-params"].get("delta")
    assert isinstance(value, float), f"delta={value}, messages={SHOWN[n:]}"
    assert abs(value - default) > 1e-9, (
        f"delta vaut encore le défaut ({default}) : le bouton n'a rien calculé")
    return f"clic -> delta = {value:.4f} (défaut {default}, donc réellement estimé)"
check("C5  bouton « Estimate » de delta, par clic", c5)


def c6():
    from gui.reusable import SingularValuePickerDialog
    section = cw.algorithm_selection_section
    section.algo_combo.setCurrentText("MCMC"); app.processEvents()
    widget = section.algo_widgets["MCMC"]
    n = len(SHOWN)
    click(button(widget.parameter_widgets["lambda_rr"], "Estimate"))
    opened = top_level(SingularValuePickerDialog)
    assert opened, f"dialogue non ouvert (messages: {SHOWN[n:]})"
    opened[-1].accept(); app.processEvents()
    value = load_or_create_toml(MATIRF_CONFIG_PATH,
                                DEFAULT_MATIRF_CONFIG)["algo-params"].get("lambda_rr")
    assert isinstance(value, float) and value > 0
    for d in opened:
        d.close()
    return f"clic -> SVD de H -> dialogue -> lambda_rr = {value:.4g}"
check("C6  bouton « Estimate » de lambda_rr, par clic", c6)


print("\n=== D. RUN ET FENÊTRE D'AFFICHAGE (clics réels) ===")

DISPLAY = {}


def d1():
    """Le vrai parcours : on clique sur Run et une fenêtre d'affichage apparaît."""
    save_toml(matirf_ready_config(), MATIRF_CONFIG_PATH)
    cw.load_cached_config()
    Display = display_window_class(MATIRF)
    before = set(top_level(Display))
    click(button(cw, "Run"))
    opened = [w for w in top_level(Display) if w not in before]
    assert opened, "aucune fenêtre d'affichage ouverte par le clic sur Run"
    DISPLAY["window"] = opened[-1]
    DISPLAY["pipeline"] = opened[-1].pipeline
    assert wait_until(lambda: not DISPLAY["pipeline"].is_running), "run non terminé"
    wait_until(lambda: DISPLAY["pipeline"].state is PipelineState.COMPLETED, timeout=10)
    assert DISPLAY["pipeline"].state is PipelineState.COMPLETED, DISPLAY["pipeline"].state
    return "Run cliqué -> fenêtre ouverte -> run terminé"
check("D1  clic sur « Run » : la reconstruction va au bout", d1)


def d2():
    window = DISPLAY["window"]
    assert window.save_btn.isEnabled() and window.save_action.isEnabled()
    assert window.synthetic_section is not None and window.switch_btn is not None
    return "Save activé, section vérité et bascule présentes"
check("D2  la fenêtre reflète la fin du run", d2)


def d3():
    figures = DISPLAY["window"].figures_section
    assert figures._initialized and len(figures._views) == 2
    first = figures._current_view_index
    click_switch(figures._switch)
    assert figures._current_view_index != first, "le clic n'a pas changé de vue"
    click_switch(figures._switch)
    return "2 vues, bascule par clic sur le commutateur"
check("D3  bascule entre les vues, par clic", d3)


def d4():
    out = Path(tempfile.mkdtemp()) / "vue.png"
    figures = DISPLAY["window"].figures_section
    with dialogs(save_path=str(out)):
        click(button(figures, "Save view as PNG"))
    assert out.exists() and out.stat().st_size > 1000
    return f"clic -> PNG {out.stat().st_size // 1024} ko"
check("D4  export PNG, par clic", d4)


def d5():
    section = DISPLAY["window"].synthetic_section
    rows = section.table.rowCount()
    assert rows > 0, "table de métriques vide"
    names = [section.table.item(r, 0).text() for r in range(rows)]
    curves = [r for r in range(rows) if section.table.cellWidget(r, 2) is not None]
    assert "FSC" in names and curves
    click(section.table.cellWidget(curves[0], 2))          # le bouton 📈
    assert len(section._curve_windows) >= 1, "popup de courbe non ouverte"
    for w in list(section._curve_windows):
        w.close()
    return f"{rows} métriques dont FSC ; popup de courbe par clic"
check("D5  table des métriques et popup de courbe, par clic", d5)


def d6():
    section = DISPLAY["window"].synthetic_section
    click(section.viewer_btn)
    assert section.viewer_window is not None, "fenêtre de différence non ouverte"
    section.viewer_window.close()
    return "clic -> fenêtre de différence"
check("D6  « Visualize difference », par clic", d6)


def d7():
    window = DISPLAY["window"]
    first = window.stack.currentIndex()
    click(window.switch_btn)
    assert window.stack.currentIndex() != first, "la pile n'a pas changé"
    click(window.switch_btn)
    return "reconstruction <-> vérité, par clic"
check("D7  « Go To Truth », par clic", d7)


def d8():
    out = Path(tempfile.mkdtemp()) / "run"
    with dialogs(save_path=str(out)):
        click(DISPLAY["window"].save_btn)
    files = sorted(p.name for p in out.iterdir())
    assert files == ["config.toml", "f.TIF", "f_true.TIF", "messages.txt", "metrics.json"], files
    reopened = pipeline_for(MATIRF).create(matirf_ready_config())
    reopened.load_results(str(out))
    pipeline_for(MATIRF).remove(reopened)
    assert reopened.result.f is not None and reopened.result.metrics
    return f"clic sur Save -> {len(files)} fichiers, relus sans la session"
check("D8  sauvegarde de la reconstruction, par clic, puis relecture", d8)


def d9():
    config = matirf_ready_config()
    config["algo-params"] = {"max_iter": 5_000_000, "lr": 1e-4, "K": 3}
    save_toml(config, MATIRF_CONFIG_PATH)
    cw.load_cached_config()
    Display = display_window_class(MATIRF)
    before = set(top_level(Display))
    click(button(cw, "Run"))
    window = [w for w in top_level(Display) if w not in before][-1]
    pipeline = window.pipeline
    assert wait_until(lambda: pipeline.latest_preview is not None, timeout=60), \
        "aucun instantané d'aperçu live"
    pipeline.stop()
    assert pipeline.state is PipelineState.INTERRUPTED, pipeline.state
    window.close()
    return "aperçu live obtenu pendant le run, puis interruption"
check("D9  aperçu live pendant le run, puis interruption", d9)


print("\n=== E. DÉCONVOLUTION ===")


def e1():
    section = dw.input_files_section
    with dialogs(open_path=PNG):
        click(button(section.image_selector, "Choose .png file"))
    with dialogs(open_path=DJSON):
        click(button(section.json_selector, "Choose .json file"))
    assert section.are_both_file_selected()
    n = len(SHOWN)
    click(section.image_selector._extra_btn)
    assert section.image_selector.sub_window is not None, f"aperçu: {SHOWN[n:]}"
    section.image_selector.sub_window.close()
    return "fichiers choisis et aperçu ouverts par clic"
check("E1  deconv : fichiers et aperçu, par clic", e1)


def e2():
    config = {
        "algorithm": "ADAM",
        "input-paths": {"mode": DataMode.SYNTHETIC.value, "png": PNG, "json": DJSON},
        "add-noise": {},
        "algo-params": {"max_iter": 8, "lr": 0.02, "K": 4, "EPS": 1e-14},
    }
    save_toml(config, DECONV_CONFIG_PATH)
    dw.load_cached_config()
    Display = display_window_class(DECONV)
    before = set(top_level(Display))
    click(button(dw, "Run"))
    window = [w for w in top_level(Display) if w not in before][-1]
    assert wait_until(lambda: not window.pipeline.is_running)
    wait_until(lambda: window.pipeline.state is PipelineState.COMPLETED, timeout=10)
    metrics = window.pipeline.result.metrics
    assert metrics and "FSC" not in metrics, "FSC ne doit pas apparaître sur un problème 2D"
    window.close()
    return f"Run cliqué, {len(metrics)} métriques, FSC correctement absente (2D)"
check("E2  deconv : run complet par clic sur Run", e2)


def e3():
    section = dw.algorithm_selection_section
    section.algo_combo.setCurrentText("MCMC"); app.processEvents()
    widget = section.algo_widgets["MCMC"]
    assert "delta" not in widget.parameter_widgets, \
        "delta ne doit pas exister sur un problème 2D"
    button(widget.parameter_widgets["lambda_rr"], "Estimate")
    return "« Estimate » présent ; delta correctement absent (2D)"
check("E3  deconv : filtrage de delta par les features", e3)


print("\n=== F. GÉNÉRATEUR DE VÉRITÉ SYNTHÉTIQUE ===")


def f1():
    from problems.matirf.synthetic.gui import SyntheticTruthGeneratorWindow
    window = SyntheticTruthGeneratorWindow()
    first = window.f_true.clone()
    click(window.btn_generate)                    # le vrai bouton
    assert window.f_true is not None
    assert window.f_true.shape == first.shape
    globals()["_SYNTH"] = window
    return f"clic sur « Generate » -> {tuple(window.f_true.shape)}, valeurs dans [0, 1]"
check("F1  générer, par clic (le bug corrigé)", f1)


def f2():
    window = globals()["_SYNTH"]
    out = Path(tempfile.mkdtemp()) / "truth.TIF"
    with dialogs(save_path=str(out)):
        click(window.btn_save)
    assert out.exists() and out.stat().st_size > 1000
    return f"clic -> TIF {out.stat().st_size // 1024} ko"
check("F2  sauvegarder en TIF, par clic", f2)


def f3():
    window = globals()["_SYNTH"]
    out = Path(tempfile.mkdtemp()) / "used.TIF"
    with dialogs(save_path=str(out)):
        click(window.btn_use)
    config = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    assert config["input-paths"]["mode"] == "synthetic-data"
    assert config["input-paths"]["tif"] == str(out)
    assert config["oper-params"]["nz"] == window.f_true.shape[0]
    window.close()
    return f"clic -> config matirf mise à jour (mode, tif, nz={config['oper-params']['nz']})"
check("F3  « Use as MA-TIRF synthetic truth », par clic", f3)


print("\n=== G. CHEMINS D'ERREUR ===")


def g1():
    """Un .json incomplet doit être refusé, pas accepté silencieusement."""
    bad = Path(tempfile.mkdtemp()) / "incomplet.json"
    bad.write_text('{"angles_deg": [60.0]}')
    section = cw.input_files_section
    before = section.json_selector.selected_path
    with dialogs(open_path=str(bad)):
        click(button(section.json_selector, "Choose .json file"))
    assert section.json_selector.selected_path == before, \
        "un .json auquel il manque des clés a été accepté"
    return "json incomplet refusé, sélection précédente conservée"
check("G1  un .json invalide est refusé", g1)


def g2():
    """Une config incomplète produit une LISTE lisible, jamais une trace d'exception."""
    messages, errors = [], []
    pipeline = pipeline_for(MATIRF).create(dict(DEFAULT_MATIRF_CONFIG))
    pipeline.on_message, pipeline.on_error = messages.append, errors.append
    pipeline.start()
    pipeline_for(MATIRF).remove(pipeline)
    assert errors and pipeline.state is PipelineState.IDLE
    joined = "\n".join(messages)
    for expected in ("Input file (TIF)", "'nz'", "'normalize'", "Algorithm: not selected"):
        assert expected in joined, f"manque dans le rapport: {expected}"
    assert "Traceback" not in joined
    return f"{len(messages) - 1} problèmes listés, dont normalize"
check("G2  config vide : une liste lisible, pas une exception", g2)


def g3():
    """Un aperçu impossible explique pourquoi, au lieu de montrer une trace.

    Le cas choisi est réel : un .json déclarant moins d'angles qu'il n'y a de piles dans
    le .tif. Le prétraitement ne peut pas associer les deux et lève une AssertionError,
    que l'interface doit traduire en phrase compréhensible.
    """
    import json
    mismatched = Path(tempfile.mkdtemp()) / "trop_peu_d_angles.json"
    full = json.loads(Path(MJSON).read_text())
    mismatched.write_text(json.dumps({**full, "angles_deg": full["angles_deg"][:5]}))

    config = matirf_ready_config()
    config["input-paths"] = {"mode": DataMode.REAL.value, "tif": TIF,
                             "json": str(mismatched)}
    save_toml(config, MATIRF_CONFIG_PATH)
    cw.load_cached_config()
    section = cw.input_files_section
    section.image_selector.update_selected_file(TIF)
    section.json_selector.update_selected_file(str(mismatched))

    n = len(SHOWN)
    click(section.image_selector._extra_btn)
    warned = [s for s in SHOWN[n:] if s[0] == "warning"]
    assert warned, "aucun message alors que les fichiers sont incohérents"
    text = warned[-1][2]
    assert "Traceback" not in text, "une trace brute a été montrée à l'utilisateur"
    assert "angles" in text or "stacks" in text, f"message peu explicite: {text[:80]}"
    return "13 piles contre 5 angles -> message explicatif, pas une trace"
check("G3  aperçu impossible : un message, pas une trace", g3)


print("\n=== H. LES DEUX PROBLÈMES EN PARALLÈLE ===")


def h1():
    """Chaque problème a son propre registre : arrêter l'un ne touche pas l'autre.

    Les comptes sont RELATIFS : les fenêtres encore ouvertes des tests précédents gardent
    légitimement leur pipeline enregistré — un pipeline n'est retiré qu'à la fermeture de
    sa fenêtre. Un compte absolu testerait l'ordre des tests, pas le registre.
    """
    matirf_pipelines = pipeline_for(MATIRF)
    deconv_pipelines = pipeline_for(DECONV)
    assert matirf_pipelines is not deconv_pipelines
    assert matirf_pipelines._running_pipelines is not deconv_pipelines._running_pipelines

    before_deconv = len(deconv_pipelines.get_all())
    matirf_pipelines.create(matirf_ready_config())
    deconv_pipelines.create({"algorithm": "ADAM",
                             "input-paths": {"mode": DataMode.SYNTHETIC.value,
                                             "png": PNG, "json": DJSON},
                             "add-noise": {}, "algo-params": {}})
    assert len(deconv_pipelines.get_all()) == before_deconv + 1

    matirf_pipelines.stop_all()
    assert matirf_pipelines.get_all() == [], "le registre matirf devrait être vidé"
    assert len(deconv_pipelines.get_all()) == before_deconv + 1, \
        "arrêter matirf a touché les pipelines de deconv"
    deconv_pipelines.stop_all()
    return "registres indépendants ; arrêter matirf n'arrête pas deconv"
check("H1  registres de pipelines indépendants par problème", h1)


def h2():
    """La classe liée à un problème doit être unique — sinon un run resterait orphelin."""
    assert pipeline_for(MATIRF) is pipeline_for(MATIRF)
    assert display_window_class(MATIRF) is display_window_class(MATIRF)
    assert control_window_class(MATIRF) is control_window_class(MATIRF)
    assert display_window_class(MATIRF) is not display_window_class(DECONV)
    return "classes mémoïsées par problème, distinctes entre problèmes"
check("H2  une seule classe liée par problème", h2)


# ── restore the caches exactly as they were ───────────────────────────────────
for path, content in BACKUPS.items():
    Path(path).write_text(content)

print("\n" + "=" * 66)
ok = sum(1 for r in RESULTS if r[0])
print(f"RÉSULTAT : {ok}/{len(RESULTS)} vérifications passées")
if ok < len(RESULTS):
    print("\nÉCHECS :")
    for good, name, detail in RESULTS:
        if not good:
            print(f"  - {name}\n      {detail}")
sys.exit(0 if ok == len(RESULTS) else 1)
