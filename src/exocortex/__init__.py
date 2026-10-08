"""Exocortex: Kählerian Tensor Thermodynamics & Non-Invasive Cognitive Augmentation Library.

The package exposes the canonical mathematical objects described in the Exocortex
manifesto: complex projective substrates, zero-entropy verification, Amador
transforms, Hodge decomposition, fluid-time manifolds and deterministic
robotic geodesics.
"""

from .core.entropy import ZeroEntropyVerifier, assert_zero_entropy, zero_entropy_delta
from .core.substrate import (
    ComplexProjectiveSubstrate,
    FubiniStudyMetric,
    KahlerianSubstrate,
    SustratoKahlariano,
)
from .engine import ExocortexEngine
from .fluid_time.holomorphic import HolomorphicPlate, PhaseModulator
from .fluid_time.manifold_w import ComplexTime, FluidTimeManifold, ManifoldW
from .interfaces.agi_bridge import AGIBridge, ExocortexBridge
from .interfaces.quantum_ent import QuantumEntityMapper, QuantumEntangler
from .robotics.geodesics import GeodesicController, GeodesicNavigator
from .robotics.kinematics import (
    IndustrialKinematicsController,
    RobotKinematicsController,
)
from .transforms.amador import AmadorTransform, TransformadaAmador
from .transforms.hodge import HarmonicProjection, HodgeDecomposition

__version__ = "0.1.0"
__author__ = "Francisco Amador Barrios Espinoza"

__all__ = [
    "AGIBridge",
    "AmadorTransform",
    "ComplexProjectiveSubstrate",
    "ComplexTime",
    "ExocortexBridge",
    "ExocortexEngine",
    "FluidTimeManifold",
    "FubiniStudyMetric",
    "GeodesicController",
    "GeodesicNavigator",
    "HarmonicProjection",
    "HodgeDecomposition",
    "HolomorphicPlate",
    "IndustrialKinematicsController",
    "KahlerianSubstrate",
    "ManifoldW",
    "PhaseModulator",
    "QuantumEntangler",
    "QuantumEntityMapper",
    "RobotKinematicsController",
    "SustratoKahlariano",
    "TransformadaAmador",
    "ZeroEntropyVerifier",
    "assert_zero_entropy",
    "zero_entropy_delta",
    "__author__",
    "__version__",
]
