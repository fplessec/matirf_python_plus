import torch

from core.operations import compute_matirf_operator_from_params, apply_matirf_operator
from core.preprocess_measurement import preprocess_measurement_stack
from in_out import save_tif, load_json, MEASUREMENTS_DIR

nb_peaks = 5
z_centers = torch.linspace(50, 450, nb_peaks)  # nm
peak_widths = torch.linspace(10, 50, nb_peaks)  # nm

# volume shape:
Nz, Ny, Nx = 500, 100, 100  # z, y, x
# volume axis
z = torch.arange(Nz).view(Nz, 1, 1)   # (z,1,1)
y = torch.arange(Ny).view(1, Ny, 1)   # (1,y,1)
x = torch.arange(Nx).view(1, 1, Nx)   # (1,1,x)
# z axis is from 1 to 500 nm
x_norm = x / (Nx - 1)
y_norm = y / (Ny - 1)
# x and y axis are from 0 to 1 (no unit) in their normalized versions
# xy plan is divided into nb_peaks x nb_peaks patchs
delta_x = 0.5 / (nb_peaks + 1)
delta_y = 0.5 / (nb_peaks + 1)

# final volume
f_true = torch.zeros((Nz, Ny, Nx))
f_true = torch.zeros((Nz, Ny, Nx), dtype=torch.bool)

# construction:
for i in range(nb_peaks):
    for j in range(nb_peaks):
        z0 = z_centers[i]
        w = peak_widths[j]
        # position in the xy plan
        xi = (i + 1) / (nb_peaks + 1)
        yj = (j + 1) / (nb_peaks + 1)
        # masque z (rectangular fct)
        mask_z = (torch.abs(z - z0) <= w / 2)
        # localisation in x
        mask_x = (torch.abs(x_norm - xi) <= delta_x)
        # localisation in y
        mask_y = (torch.abs(y_norm - yj) <= delta_y)
        # logical product to form 3D rectangular box:
        mask = mask_z & mask_y & mask_x
        # logical addition:
        f_true |= mask

# convert to float:
f_true = f_true.float()

# save as:
save_tif(f_true, "f_true_peaks_mesh.TIF")

# constructing the synthetic MA-TIRF stacks measurements:
measurement_params = load_json(MEASUREMENTS_DIR / "esoubies.json")
oper_params = {
    'nz': Nz,
    'z0': 0,
    'zN': 500,
    'normalize': False
}
add_noise_params = {
    'add_noise': False,
    'is_gaussian': True,
    'sigma': 0
}
H = compute_matirf_operator_from_params(measurement_params, oper_params)

g_synth = apply_matirf_operator(H, f_true)
g_synth, _ = preprocess_measurement_stack(
    g_synth, measurement_params, add_noise_params
)
# save as:
save_tif(g_synth, "g_synth_peaks_mesh.TIF")


if __name__=="__main__":  # visualization
    import sys
    from pathlib import Path

    from PyQt5.QtWidgets import QApplication, QStyleFactory, QHBoxLayout, QGroupBox, QWidget

    from gui.more_widgets import ImageAndHisto3DViewer, DepthMapViewer, ProfilesViewer
    from in_out import load_tif
    import settings


    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create(settings.app_style))
    palette = settings.dark_palette if settings.dark_style else settings.light_palette

    package_path = Path(__file__).parent
    image3d = load_tif(package_path / "f_true_peaks_mesh.TIF")

    qgroup_slices_histogram = ImageAndHisto3DViewer(image=image3d, title='slice by slice and histogram')
    qgroup_depth_profiles = QGroupBox(title='depthmap and profiles')
    z0, zN = 0, 500
    widget1 = DepthMapViewer(image3d, z0, zN)
    widget2 = ProfilesViewer(image3d, z0, zN)
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget1, 2)  # 2/3 of the width
    layout.addWidget(widget2, 1)  # 1/3 of the width
    qgroup_depth_profiles.setLayout(layout)

    window = QWidget()
    main_layout = QHBoxLayout()
    main_layout.addWidget(qgroup_slices_histogram)
    main_layout.addWidget(qgroup_depth_profiles)
    window.setLayout(main_layout)
    window.resize(2100, 700)
    window.show()

    sys.exit(app.exec_())