"""
GUI smoke test — every user-facing feature, exercised end to end.

Run it:
    python -m _gui_smoke_test

The unit suites (core, solvers, problems, pipeline) check contracts in isolation. This one
answers a different question: does the APPLICATION still do everything it did before? It
drives the real windows — the real selectors, the real Estimate buttons, a real
reconstruction, the real save/load — and prints PASS/FAIL per feature.

Run it after any change to the GUI, to a problem, or to the pipeline. It is the fastest way
to know that a refactor did not quietly remove a capability, which is exactly how the
synthetic generator's preview broke once without anyone noticing.

It runs headless (QT_QPA_PLATFORM=offscreen). Modal dialogs are neutralised so nothing
blocks, and the two cached config.toml are backed up and restored, so running it never
disturbs the configuration you were working on.

Sections:
    B  control window      inputs, preview, editor, parameters, solver choice, Estimate
    C  run + display       full run, figures, metrics, difference, save/load, interrupt
    D  deconvolution       the same, on the other problem
    E  matirf synth        generate, preview, save, use as synthetic truth
"""
import os, sys, json, shutil, tempfile, time, traceback
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QMessageBox
import PyQt5.QtWidgets as QtW

app = QApplication(sys.argv)

# ── neutralise anything modal, record what would have been shown ──────────────
SHOWN = []
QMessageBox.warning = staticmethod(lambda *a, **k: SHOWN.append(("warning", a[1] if len(a) > 1 else "")) or QMessageBox.Ok)
QMessageBox.information = staticmethod(lambda *a, **k: SHOWN.append(("info", a[1] if len(a) > 1 else "")) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: SHOWN.append(("critical", a[1] if len(a) > 1 else "")) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)

RESULTS = []
def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail or ""))
        print(f"  PASS  {name}" + (f"  — {detail}" if detail else ""))
    except Exception as e:
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"  FAIL  {name}  — {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)

from problems.matirf import MATIRF_MEASUREMENTS_DIR, MATIRF_CONFIG_PATH
from problems.deconv import DECONV_MEASUREMENTS_DIR, DECONV_CONFIG_PATH

TIF  = str(MATIRF_MEASUREMENTS_DIR / "esoubies.TIF")
MJSON = str(MATIRF_MEASUREMENTS_DIR / "esoubies.json")
TRUTH = str(MATIRF_MEASUREMENTS_DIR / "synthetic_truth0.TIF")
PNG  = str(DECONV_MEASUREMENTS_DIR / "img_001.png")
DJSON = str(DECONV_MEASUREMENTS_DIR / "psf_params_example.json")

# keep the user's real caches untouched
BACKUPS = {}
for p in (MATIRF_CONFIG_PATH, DECONV_CONFIG_PATH):
    if Path(p).exists():
        BACKUPS[p] = Path(p).read_text()

print("\n=== B. FENÊTRE DE CONTRÔLE ===")

from gui.factory import control_window_class, display_window_class
from problems.matirf import MATIRF
from problems.deconv import DECONV
ControlWindow = control_window_class(MATIRF)
DeconvControlWindow = control_window_class(DECONV)
from fileio import load_or_create_toml, save_toml
from problems.matirf import DEFAULT_MATIRF_CONFIG

cw = ControlWindow()
dw = DeconvControlWindow()

def b1():
    return f"matirf + deconv construites"
check("B1  les deux fenêtres de contrôle se construisent", b1)

def b2():
    s = cw.input_files_section
    assert s.image_selector and s.json_selector
    assert hasattr(s, "switch_button")
    return "sélecteurs image + json + bascule de mode"
check("B2  section fichiers d'entrée", b2)

def b3():
    s = cw.input_files_section
    before = s.is_mode_real
    s._switch_mode()
    cfg = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    written = cfg["input-paths"]["mode"]
    s._switch_mode()  # back
    assert s.is_mode_real == before
    return f"bascule réel/synthétique écrite dans le cache ({written})"
check("B3  bascule de mode -> cache", b3)

def b4():
    s = cw.input_files_section
    s.image_selector.update_selected_file(TIF)
    s.json_selector.update_selected_file(MJSON)
    assert s.are_both_file_selected()
    assert not s.image_selector._extra_btn.isHidden(), "bouton preview devrait apparaître"
    assert s.json_selector._extra_btn.text() == "Modify .json file"
    return "bouton preview apparaît quand les 2 fichiers sont choisis"
check("B4  dépendance croisée des sélecteurs", b4)

def b5():
    s = cw.input_files_section
    v = s._make_validator(("angles_deg", "n_glass"))
    bad = Path(tempfile.mkdtemp()) / "bad.json"; bad.write_text('{"angles_deg": [1]}')
    assert v(MJSON) is True and v(str(bad)) is False
    return "json valide accepté, json incomplet rejeté"
check("B5  validateur du .json", b5)

def b6():
    s = cw.input_files_section
    n = len(SHOWN)
    s._open_preview(s.image_selector, s.IMAGE_SLOT.preview)
    win = s.image_selector.sub_window
    assert win is not None, f"aperçu non ouvert (messages: {SHOWN[n:]})"
    shape = tuple(win.left.shape), tuple(win.right.shape)
    win.close()
    return f"fenêtre d'aperçu ouverte {shape[0]} -> {shape[1]}"
check("B6  aperçu du prétraitement (matirf, mode réel)", b6)

def b7():
    s = cw.input_files_section
    s._open_editor(s.json_selector, s.JSON_SLOT.editor)
    ed = s.json_selector.sub_window
    assert ed is not None
    n = len(ed._widgets) if hasattr(ed, "_widgets") else "?"
    ed.close()
    return f"éditeur de paramètres ouvert"
check("B7  éditeur JSON (Create/Modify)", b7)

def b8():
    sec = cw.oper_params
    assert hasattr(sec, "update_ui_from_toml")
    cw.update_cache_fn(["oper-params", "nz"], 10)
    cw.update_cache_fn(["oper-params", "z0"], 0.0)
    cw.update_cache_fn(["oper-params", "zN"], 400.0)
    cw.update_cache_fn(["oper-params", "normalize"], False)
    sec.update_ui_from_toml(MATIRF_CONFIG_PATH)
    cfg = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    assert cfg["oper-params"]["nz"] == 10
    return "nz/z0/zN écrits et relus"
check("B8  section paramètres opérateur (matirf)", b8)

def b9():
    sec = cw.add_noise
    cw.update_cache_fn(["add-noise", "add_noise"], True)
    cw.update_cache_fn(["add-noise", "sigma"], 0.01)
    sec.update_ui_from_toml(MATIRF_CONFIG_PATH)
    cfg = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    assert cfg["add-noise"]["add_noise"] is True and cfg["add-noise"]["sigma"] == 0.01
    cw.update_cache_fn(["add-noise", "add_noise"], False)
    return "add_noise + sigma écrits et relus"
check("B9  section ajout de bruit", b9)

def b10():
    sec = cw.algorithm_selection_section
    names = [sec.algo_combo.itemText(i) for i in range(sec.algo_combo.count())]
    assert names == ["None", "ADAM", "PPXA", "ADMM", "PNP", "ADMM-PnP", "MCMC"], names
    return f"{len(names)-1} solveurs proposés"
check("B10 sélection d'algorithme : les 6 solveurs", b10)

def b11():
    sec = cw.algorithm_selection_section
    sec.algo_combo.setCurrentText("ADAM")
    cfg = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    assert cfg["algorithm"] == "ADAM", cfg["algorithm"]
    keys = set(cfg["algo-params"])
    assert {"max_iter", "lr"} <= keys, keys
    return f"ADAM sélectionné, {len(keys)} paramètres écrits"
check("B11 changement d'algorithme -> cache", b11)

def b12():
    sec = cw.algorithm_selection_section
    w = sec.algo_widgets["ADAM"]
    w.reset_default_values()
    cfg = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    assert cfg["algo-params"]["max_iter"] == 1000
    return "valeurs par défaut restaurées (max_iter=1000)"
check("B12 bouton 'reset parameters'", b12)

def b13():
    from gui.estimators import attach
    from solvers import SOLVERS as _S
    MATIRF_SOLVERS = attach(_S, MATIRF.ui.estimators, MATIRF)
    p = MATIRF_SOLVERS["ADAM"].get_ui_params(MATIRF.features)
    assert "extra_button" in p["delta"], "bouton Estimate absent de delta"
    p2 = MATIRF_SOLVERS["MCMC"].get_ui_params(MATIRF.features)
    assert "extra_button" in p2["lambda_rr"], "bouton Estimate absent de lambda_rr"
    from solvers import SOLVERS
    assert "extra_button" not in SOLVERS["ADAM"].get_ui_params(MATIRF.features)["delta"]
    return "delta + lambda_rr ont un bouton; le registre partagé reste intact"
check("B13 boutons 'Estimate' (matirf)", b13)

def b14():
    # on passe par le VRAI chemin câblé : le bouton tel que la fenêtre l'a monté
    from gui.estimators import attach
    from solvers import SOLVERS as _S
    view = attach(_S, MATIRF.ui.estimators, MATIRF)
    button = view["ADAM"].get_ui_params(MATIRF.features)["delta"]["extra_button"]
    w = cw.algorithm_selection_section.algo_widgets["ADAM"]
    n = len(SHOWN)
    button["callback"](w, cw.update_cache_fn, cw.load_config, MATIRF_CONFIG_PATH)
    cfg = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    d = cfg["algo-params"].get("delta")
    assert isinstance(d, float) and d > 0, f"delta={d}, messages={SHOWN[n:]}"
    return f"delta estimé = {d}"
check("B14 callback Estimate delta (calcul réel)", b14)

def b15():
    # le dialogue réel est modeless (show + signal accepted) : on le retrouve parmi les
    # fenêtres de premier niveau et on l'accepte, sans rien remplacer
    from gui.estimators import attach
    from gui.reusable import SingularValuePickerDialog
    from solvers import SOLVERS as _S
    view = attach(_S, MATIRF.ui.estimators, MATIRF)
    button = view["MCMC"].get_ui_params(MATIRF.features)["lambda_rr"]["extra_button"]
    w = cw.algorithm_selection_section.algo_widgets["MCMC"]
    n = len(SHOWN)
    button["callback"](w, cw.update_cache_fn, cw.load_config, MATIRF_CONFIG_PATH)
    dialogs = [x for x in app.topLevelWidgets() if isinstance(x, SingularValuePickerDialog)]
    assert dialogs, f"dialogue non ouvert (messages: {SHOWN[n:]})"
    dialogs[-1].accept(); app.processEvents()
    cfg = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    v = cfg["algo-params"].get("lambda_rr")
    assert isinstance(v, float) and v > 0, f"lambda_rr={v}"
    for d in dialogs: d.close()
    return f"SVD de H, dialogue ouvert, lambda_rr écrit = {v:.4g}"
check("B15 callback Estimate lambda_rr (SVD de H)", b15)

def b16():
    path = Path(tempfile.mkdtemp()) / "saved.toml"
    cfg = cw.load_config(MATIRF_CONFIG_PATH)
    cw.save_config(cfg, path)
    reread = cw.load_config(path)
    assert reread["algorithm"] == cfg["algorithm"]
    cw.save_config(reread, MATIRF_CONFIG_PATH)
    cw.load_cached_config()
    return "config sauvegardée puis rechargée dans toutes les sections"
check("B16 sauvegarder / charger une config", b16)

print("\n=== C. EXÉCUTION + FENÊTRE D'AFFICHAGE ===")

from pipeline import pipeline_for
from problems.matirf import MATIRF
from problems.deconv import DECONV
from core import DataMode
from core.enums import PipelineState

def synth_cfg():
    return {"algorithm": "ADAM",
            "input-paths": {"mode": DataMode.SYNTHETIC.value, "tif": TRUTH, "json": MJSON},
            "oper-params": {"nz": 50, "z0": 0.0, "zN": 400.0, "normalize": False},
            "add-noise": {},
            "algo-params": {"max_iter": 15, "lr": 0.01, "K": 5, "EPS": 1e-14}}

MP = pipeline_for(MATIRF)
pipeline = MP.create(synth_cfg())

DisplayWindow = display_window_class(MATIRF)
disp = DisplayWindow(pipeline)

def c1():
    assert disp.figures_section is not None
    assert disp.synthetic_section is not None, "section vérité synthétique manquante"
    assert disp.switch_btn is not None, "bouton Go To Truth manquant"
    return "figures + vérité synthétique + bouton de bascule"
check("C1  fenêtre d'affichage construite (mode synthétique)", c1)

def c2():
    msgs = []
    pipeline.on_message = msgs.append
    pipeline.start()
    t0 = time.time()
    while pipeline.is_running and time.time() - t0 < 90:
        app.processEvents(); time.sleep(0.02)
    for _ in range(50):
        app.processEvents(); time.sleep(0.01)
    assert pipeline.state is PipelineState.COMPLETED, pipeline.state
    assert pipeline.result.f is not None
    return f"run terminé, {len(msgs)} messages"
check("C2  Run complet depuis la fenêtre", c2)

def c3():
    assert disp.save_btn.isEnabled(), "bouton Save non activé après le run"
    assert disp.save_action.isEnabled()
    return "bouton + menu Save activés"
check("C3  Save activé à la fin du run", c3)

def c4():
    fs = disp.figures_section
    assert fs._initialized, "vues non créées"
    assert len(fs._views) == 2, len(fs._views)
    assert fs.supports_view1_export()
    i0 = fs._current_view_index
    fs._on_switch_toggled()
    assert fs._current_view_index != i0
    fs._on_switch_toggled()
    return "2 vues, bascule OK, export PNG supporté"
check("C4  section figures : 2 vues + bascule", c4)

def c5():
    out = Path(tempfile.mkdtemp()) / "view1.png"
    disp.figures_section.export_view1(str(out))
    assert out.exists() and out.stat().st_size > 1000, out.stat().st_size if out.exists() else "absent"
    return f"PNG écrit ({out.stat().st_size // 1024} ko)"
check("C5  export PNG de la vue 1", c5)

def c6():
    st = disp.synthetic_section
    st.update_plot()
    rows = st.table.rowCount()
    assert rows > 0, "table de métriques vide"
    names = [st.table.item(r, 0).text() for r in range(rows)]
    curve = [r for r in range(rows) if st.table.cellWidget(r, 2) is not None]
    assert "FSC" in names, names
    assert curve, "aucune métrique courbe avec bouton"
    return f"{rows} métriques dont FSC, {len(curve)} courbe(s)"
check("C6  table des métriques (vérité synthétique)", c6)

def c7():
    st = disp.synthetic_section
    st._open_viewer()
    assert st.viewer_window is not None
    st.viewer_window.close()
    return "fenêtre de différence ouverte"
check("C7  visualiser la différence", c7)

def c8():
    st = disp.synthetic_section
    r = st.table.rowCount()
    row = next(i for i in range(r) if st.table.cellWidget(i, 2) is not None)
    name = st.table.item(row, 0).text()
    st.table.cellWidget(row, 2).click()
    app.processEvents()
    assert len(st._curve_windows) >= 1
    for w in list(st._curve_windows): w.close()
    return f"popup de courbe ouverte ({name})"
check("C8  popup de courbe (métrique FSC)", c8)

def c9():
    disp._change_right_column()
    idx = disp.stack.currentIndex()
    disp._change_right_column()
    assert idx == 1, idx
    return "bascule reconstruction <-> vérité"
check("C9  bouton 'Go To Truth'", c9)

def c10():
    d = Path(tempfile.mkdtemp()) / "run"
    pipeline.save_results(str(d))
    files = sorted(p.name for p in d.iterdir())
    assert files == ["config.toml", "f.TIF", "f_true.TIF", "messages.txt", "metrics.json"], files
    p2 = MP.create(synth_cfg()); p2.load_results(str(d)); MP.remove(p2)
    assert p2.result.f is not None and p2.result.metrics
    return f"{len(files)} fichiers écrits puis relus"
check("C10 sauvegarde + réouverture d'une reconstruction", c10)

def c11():
    cfg = synth_cfg(); cfg["algo-params"] = {"max_iter": 5_000_000, "lr": 1e-4, "K": 2}
    p = MP.create(cfg)
    p.start()
    t0 = time.time()
    while p.latest_preview is None and time.time() - t0 < 30:
        app.processEvents(); time.sleep(0.02)
    got = p.latest_preview is not None
    p.stop(); MP.remove(p)
    assert got, "aucun instantané d'aperçu live"
    assert p.state is PipelineState.INTERRUPTED, p.state
    return "aperçu live obtenu, puis interruption"
check("C11 aperçu live + interruption", c11)

print("\n=== D. DECONV ===")

def d1():
    cfg = {"algorithm": "ADAM",
           "input-paths": {"mode": DataMode.SYNTHETIC.value, "png": PNG, "json": DJSON},
           "add-noise": {}, "algo-params": {"max_iter": 10, "lr": 0.02, "K": 5, "EPS": 1e-14}}
    DP = pipeline_for(DECONV)
    p = DP.create(cfg)
    DDW = display_window_class(DECONV)
    w = DDW(p)
    p.start()
    t0 = time.time()
    while p.is_running and time.time() - t0 < 60:
        app.processEvents(); time.sleep(0.02)
    for _ in range(50): app.processEvents(); time.sleep(0.01)
    assert p.state is PipelineState.COMPLETED, p.state
    assert p.result.metrics and "FSC" not in p.result.metrics
    DP.remove(p)
    return f"run deconv complet, {len(p.result.metrics)} métriques (pas de FSC : 2D)"
check("D1  deconv : fenêtre + run complet", d1)

def d2():
    s = dw.input_files_section
    s.image_selector.update_selected_file(PNG)
    s.json_selector.update_selected_file(DJSON)
    n = len(SHOWN)
    s._open_preview(s.image_selector, s.IMAGE_SLOT.preview)
    win = s.image_selector.sub_window
    assert win is not None, f"aperçu non ouvert: {SHOWN[n:]}"
    win.close()
    return "aperçu deconv ouvert"
check("D2  deconv : aperçu du prétraitement", d2)

def d3():
    from gui.estimators import attach
    from solvers import SOLVERS as _S
    DECONV_SOLVERS = attach(_S, DECONV.ui.estimators, DECONV)
    D = DECONV
    p = DECONV_SOLVERS["MCMC"].get_ui_params(D.features)
    assert "extra_button" in p["lambda_rr"]
    assert "delta" not in p, "delta ne doit pas apparaître sur un problème 2D"
    return "Estimate lambda_rr présent; delta correctement absent (2D)"
check("D3  deconv : bouton Estimate + filtrage de delta", d3)

print("\n=== E. GÉNÉRATEUR DE VÉRITÉ SYNTHÉTIQUE (matirf synth) ===")

def e1():
    from problems.matirf.synthetic.gui import SyntheticTruthGeneratorWindow
    w = SyntheticTruthGeneratorWindow()
    w._generate()
    assert w.f_true is not None, "génération échouée"
    shp = tuple(w.f_true.shape)
    rng = (float(w.f_true.min()), float(w.f_true.max()))
    globals()["_SYNTH_WIN"] = w
    return f"vérité générée {shp}, valeurs dans [{rng[0]:.2f}, {rng[1]:.2f}]"
check("E1  générer / prévisualiser", e1)

def e2():
    w = globals()["_SYNTH_WIN"]
    assert w.figures is not None, "aperçu non affiché"
    return "aperçu branché sur la section figures de matirf"
check("E2  aperçu du générateur", e2)

def e3():
    import problems.matirf.synthetic.gui as G
    w = globals()["_SYNTH_WIN"]
    out = Path(tempfile.mkdtemp()) / "truth.TIF"
    G.save_file = lambda *a, **k: str(out)
    p = w._save_tif_to(str(out), "t")
    assert Path(p).exists()
    return f"TIF écrit ({Path(p).stat().st_size // 1024} ko)"
check("E3  sauvegarde en TIF", e3)

def e4():
    import problems.matirf.synthetic.gui as G
    w = globals()["_SYNTH_WIN"]
    out = Path(tempfile.mkdtemp()) / "used.TIF"
    G.save_file = lambda *a, **k: str(out)
    n = len(SHOWN)
    w._use_as_truth()
    cfg = load_or_create_toml(MATIRF_CONFIG_PATH, DEFAULT_MATIRF_CONFIG)
    assert cfg["input-paths"]["mode"] == "synthetic-data", cfg["input-paths"]["mode"]
    assert cfg["input-paths"]["tif"] == str(out)
    assert cfg["oper-params"]["nz"] == w.f_true.shape[0]
    return f"config matirf mise à jour (mode, tif, nz={cfg['oper-params']['nz']}, z0, zN)"
check("E4  'Use as MA-TIRF synthetic truth'", e4)

# ── restore the user's caches ────────────────────────────────────────────────
for p, content in BACKUPS.items():
    Path(p).write_text(content)

print("\n" + "=" * 62)
ok = sum(1 for r in RESULTS if r[0]); tot = len(RESULTS)
print(f"RÉSULTAT : {ok}/{tot} vérifications passées")
if ok < tot:
    print("\nÉCHECS :")
    for good, name, detail in RESULTS:
        if not good:
            print(f"  - {name}\n      {detail}")
sys.exit(0 if ok == tot else 1)
