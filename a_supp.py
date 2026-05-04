import torch
from math import pi
import matplotlib.pyplot as plt

from core.operations import compute_tirf_intensity_at_interface, compute_min_max_angles_from_params
from in_out import load_or_create_toml, CONFIG_PATH, load_json


config = load_or_create_toml(CONFIG_PATH)
measurement_params = load_json(config['input-paths']['json'])
print(measurement_params)

wavelength_nm = measurement_params['wavelength_nm']
ni, nt = measurement_params['n_glass'], measurement_params['n_medium']
theta_crit, theta_max = compute_min_max_angles_from_params(measurement_params)
theta_deg = torch.tensor(measurement_params['angles_deg'])
M = len(theta_deg)

i0 = compute_tirf_intensity_at_interface(theta_deg, ni, nt)
kappa = 4 * pi / wavelength_nm * torch.sqrt((ni * torch.sin(theta_deg / 180 * pi)) ** 2 - nt ** 2)


z0, zN = 0, 300  #nm
N = 13
depths = torch.linspace(z0, zN, N+1)

z_left = depths[:-1].repeat(M, 1)
z_right = depths[1:].repeat(M, 1)
K = kappa.repeat(N, 1).permute(1, 0)

a = torch.exp(-K * z_left) - torch.exp(-K * z_right)
h = (i0/kappa).repeat(N, 1).permute(1, 0) * a
h *= 100

# add noise to h to change the inverse pb:
mean, std = -1e-2, 1e-4
#h = mean+std*torch.randn_like(h)

h = torch.eye(min(M, N))
to_cat = torch.zeros(M, max(M, N) - min(M, N)) if M<N else torch.zeros(max(M, N) - min(M, N), N)
h = torch.cat((h, to_cat), dim=1 if M<N else 0)

mean, std = -1e-2, 1e-4
mask = torch.where(torch.randn_like(h) >=0.5, 1, 0)
h += 0.1*torch.randn_like(h) * mask

print(h)

print('r of h:', torch.linalg.matrix_rank(h).item())


z_center, width = 200, 40
f_true = torch.where(width/2 >= abs(depths - z_center), 1., 0.)[:-1]
f_true *= 100

g_fake = h @ f_true
print(g_fake.shape)
print(g_fake)


z = depths[:-1]


plt.figure()
plt.subplot(2,1,1)
plt.plot(z, f_true, label='f_true')
plt.legend()
plt.subplot(2,1,2)




print('PSEUDO INV')
h_pseudo_inv = torch.inverse(h.t() @ h) @ h.t()
f = h_pseudo_inv @ g_fake
for j in range(len(f_true)):
    print(depths[j].item(), 'nm')
    print(f[j].item())
print('\n\n')
#plt.plot(z, f, label='pseudo inv')


print('PSEUDO INV TIKHO')
gamma = 1000.
gamma = 1.
h_pseudo_inv_tikho = torch.inverse(h.t() @ h + gamma * torch.eye(N, N)) @ h.t()
f = h_pseudo_inv_tikho @ g_fake
for j in range(len(f_true)):
    print(depths[j].item(), 'nm')
    print(f[j].item())
print('\n\n')
plt.plot(z, f, label='ridge regression')


u, s, v = torch.linalg.svd(h)
print('s', s.shape)
s_inv = 1/s
to_cat = torch.zeros(M, max(M, N) - min(M, N)) if M<N else torch.zeros(max(M, N) - min(M, N), N)
d = torch.cat((torch.diag(s), to_cat), 1 if M<N else 0)
d_inv = torch.cat((torch.diag(s_inv), to_cat), 1 if M<N else 0).t()
print('u', u.shape)
print('v', v.shape)
print('d', d.shape)



print('INV SVD')
h_inv_svd = v @ d_inv @ u.t()
f = h_inv_svd @ g_fake
for j in range(len(f_true)):
    print(depths[j].item(), 'nm')
    print(f[j].item())
print('\n\n')
plt.plot(z, f, label='inv svd 1')


g_svd = u.t() @ g_fake
f_svd = d_inv @ g_svd
f = v @ f_svd
print('INV SVD 2')
for j in range(len(f_true)):
    print(depths[j].item(), 'nm')
    print(f[j].item())
print('\n\n')
plt.plot(z, f, label='inv svd 2')


rank = torch.linalg.matrix_rank(h)
s_order, _ = s.clone().sort(descending=True)
min_s = s_order[rank-1]
s_truncated = torch.where(s >= min_s, s, 0.)
s_inv = 1/s
d_inv = torch.cat((torch.diag(s_inv), to_cat), 1 if M<N else 0).t()
g_svd = u.t() @ g_fake
f_svd = d_inv @ g_svd
f = v @ f_svd
print('INV SVD truncated')
for j in range(len(f_true)):
    print(depths[j].item(), 'nm')
    print(f[j].item())
print('\n\n')
plt.plot(z, f, label='inv svd truncated')



gamma = 1000.
gamma = 1.
d_inv = torch.cat((torch.diag(s / (s**2 + gamma)), to_cat), 1 if M<N else 0).t()
g_svd = u.t() @ g_fake
f_svd = d_inv @ g_svd
f = v @ f_svd
print('INV SVD tihkonov')
for j in range(len(f_true)):
    print(depths[j].item(), 'nm')
    print(f[j].item())
print('\n\n')
plt.plot(z, f, label='inv svd tihkonov')













ax = plt.gca()
lines = ax.get_lines()
n_lines = len(lines)
line_width = 3
for i, line in enumerate(lines):
    line.set_linewidth(n_lines*line_width - i*line_width)






plt.yscale('linear')
plt.legend()
plt.tight_layout()
plt.show()
