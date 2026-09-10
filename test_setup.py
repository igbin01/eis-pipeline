import numpy as np
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit

# Simulate a simple coated-metal spectrum:
# R0 = solution resistance, R1 = coating pore resistance,
# CPE1 = coating capacitance (non-ideal)
f = np.logspace(5, -2, 60)
true_params = [50, 8000, 2e-7, 0.90]
truth = CustomCircuit('R0-p(R1,CPE1)', initial_guess=true_params)
Z = truth.predict(f, use_initial=True)

# Add a little noise so it resembles real measured data
rng = np.random.default_rng(42)
Z_noisy = Z * (1 + 0.01 * rng.normal(size=Z.shape))

# Now fit it back
fitted = CustomCircuit('R0-p(R1,CPE1)', initial_guess=[100, 5000, 1e-6, 0.8])
fitted.fit(f, Z_noisy)
print(fitted)

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(Z_noisy.real, -Z_noisy.imag, 'o', label='data')
Z_fit = fitted.predict(f)
ax.plot(Z_fit.real, -Z_fit.imag, '-', label='fit')
ax.set_xlabel("Z' (ohm)")
ax.set_ylabel("-Z'' (ohm)")
ax.legend()
ax.axis('equal')
plt.tight_layout()
plt.savefig('nyquist_test.png', dpi=150)
print('Saved nyquist_test.png')
