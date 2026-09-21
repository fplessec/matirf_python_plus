# MA-TIRF-specific settings

# The integrals in the computation of H are approximated by a sum of finite elements.
# 100 is way enough to compute precisely the integrals.
precision = 100

# The normalization of the measurement is no longer a constant here: it is chosen per run in
# the '[input-paths]' section (see core/normalization.py).
