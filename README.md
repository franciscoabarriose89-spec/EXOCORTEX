# Exocortex: Kählerian Tensor Thermodynamics & Non-Invasive Cognitive Augmentation Library

Created for the public library `Exocortex` by Francisco Amador Barrios Espinoza.

## Manifesto

### Humanitarian vision

Exocortex is a public, open research library that frames computation as a projective, thermodynamically stable substrate for human cognition and machine intelligence. The project honors the vision of a future where physical drudgery is reduced by automation, science and artistry flourish, and human presence is amplified through non-invasive external cognition.

### In English

> Exocortex is a public computational substrate designed to harmonize human intelligence, AGI, and quantum-scale symbolic systems. It models knowledge as a geometrically stable signal in complex projective space, preserving information without dissipating it into uncontrolled entropy.

### En español

> Exocortex es un sustrato computacional público diseñado para armonizar la inteligencia humana, la AGI y los sistemas simbólicos cuánticos. Modela el conocimiento como una señal geométricamente estable en el espacio proyectivo complejo, preservando la información sin disiparla en entropía no controlada.

## Mathematical foundation

The core geometry is the complex projective space $\mathrm{CP}^N$ with the Fubini-Study form:

$$
\omega_{FS} = \frac{i}{2\pi} \partial \bar{\partial} \log\left(1 + \sum_{j=1}^{N}|z_j|^2\right).
$$

The Amador transform is the harmonic filter used to preserve invariant structure:

$$
\mathcal{T}_A(S_t, V_t, \alpha, \tau) = \alpha S_t + (1 - \alpha)V_t - C_A(F, \tau).
$$

The fluid-time dimension is defined as:

$$
W = t + i\psi.
$$

## Quick start

```python
from exocortex.core.substrate import KahlerianSubstrate
from exocortex.core.entropy import ZeroEntropyVerifier
from exocortex.transforms.amador import AmadorTransform

substrate = KahlerianSubstrate(dimension=3)
state = substrate.make_state([1.0, 2.0, 3.0])
transform = AmadorTransform(alpha=0.5, tau=0.25)
filtered = transform.apply([1.0, 2.0, 3.0], [1.0, 2.1, 2.9])

assert ZeroEntropyVerifier(tolerance=1e-15).check([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
print(filtered)
```

## Geodesic robotics example

```python
from exocortex.robotics.geodesics import GeodesicController

controller = GeodesicController(max_speed=4.0)
path = controller.step([0.0, 0.0], [10.0, 0.0], dt=0.5)
print(path)
print(controller.deterministic_latency())
```

## License

This project is distributed under the MIT License with a humanitarian clause favoring open, ethical and beneficial scientific progress.
