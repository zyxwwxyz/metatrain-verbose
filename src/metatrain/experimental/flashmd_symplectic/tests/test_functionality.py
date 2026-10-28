import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import ModelOutput, System

from metatrain.experimental.flashmd_symplectic.model import FlashMDSymplectic
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import TargetInfo
from metatrain.utils.neighbor_lists import (
    get_requested_neighbor_lists,
    get_system_with_neighbor_lists,
)


@pytest.mark.filterwarnings("ignore:custom data:UserWarning")
def test_forward():
    "Run a forward pass of FlashMD on two small systems and verify the output shapes."

    # load default hyper parameters for FlashMD
    full_hypers = get_default_hypers("experimental.flashmd_symplectic")
    model_hypers = full_hypers["model"]

    # define dataset (especially the targets)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            name: TargetInfo(
                layout=TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=torch.empty((0, 3, 1), dtype=torch.float64),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.empty((0, 2), dtype=int),
                            ),
                            components=[
                                Labels.range("xyz", 3),
                            ],
                            properties=Labels.range("length", 1),
                        )
                    ],
                ),
                quantity="length",
                unit="angstrom",
            )
            for name in ["position", "momentum"]
        },
    )

    # create a FlashMD model and attach a (random) raw PET model
    model = FlashMDSymplectic(model_hypers, dataset_info)

    # define example systems
    dtype = torch.float32
    systems = []
    systems.append(
        # system 0: 3 atoms
        System(
            types=torch.tensor([1, 6, 1]),
            positions=torch.randn(3, 3, dtype=dtype),
            cell=torch.eye(3, dtype=dtype),
            pbc=torch.tensor([True] * 3),
        )
    )
    systems.append(
        # system 1: 2 atoms
        System(
            types=torch.tensor([6, 1]),
            positions=torch.randn(2, 3, dtype=dtype),
            cell=torch.eye(3, dtype=dtype),
            pbc=torch.tensor([True] * 3),
        )
    )

    # attach neighbor lists to the systems
    nbr_lists = get_requested_neighbor_lists(model)
    systems = [get_system_with_neighbor_lists(system, nbr_lists) for system in systems]

    # add random momenta to the systems
    for system in systems:
        num_atoms = len(system)
        tmap = TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    # TODO: get the momenta from the system if available!
                    values=torch.randn(num_atoms, 3, dtype=dtype),
                    samples=Labels(
                        names=["system"],
                        values=torch.arange(num_atoms, dtype=int).unsqueeze(-1),
                    ),
                    components=[],
                    properties=Labels.range("length", 3),
                ),
            ],
        )
        system.add_data("momentum", tmap)

    outputs = {
        "position": ModelOutput(unit="angstrom", sample_kind="atom"),
        "momentum": ModelOutput(unit="angstrom", sample_kind="atom"),
    }
    result_dict = model(systems, outputs)

    assert set(result_dict.keys()) == set(outputs.keys())

    # 2+3=5 atoms in total, both outputs are 3D vectors per atom
    assert result_dict["position"][0].values.shape == (5, 3, 1)
    assert result_dict["momentum"][0].values.shape == (5, 3, 1)


@pytest.mark.filterwarnings("ignore:custom data:UserWarning")
def test_last_layer_features():
    """Position and momentum are derivatives of mtt::S3, so only mtt::S3 has
    last-layer features."""

    model_hypers = get_default_hypers("experimental.flashmd_symplectic")["model"]
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 6],
        targets={
            name: TargetInfo(
                layout=TensorMap(
                    keys=Labels.single(),
                    blocks=[
                        TensorBlock(
                            values=torch.empty((0, 3, 1), dtype=torch.float64),
                            samples=Labels(
                                names=["system", "atom"],
                                values=torch.empty((0, 2), dtype=int),
                            ),
                            components=[Labels.range("xyz", 3)],
                            properties=Labels.range(name, 1),
                        )
                    ],
                ),
                quantity=quantity,
                unit=unit,
            )
            for name, quantity, unit in [
                ("position", "length", "angstrom"),
                ("momentum", "momentum", "(eV*u)^(1/2)"),
            ]
        },
    )
    model = FlashMDSymplectic(model_hypers, dataset_info)
    model.set_masses({1: 1.008, 6: 12.011})

    supported_outputs = model.supported_outputs()
    assert "mtt::aux::S3_last_layer_features" in supported_outputs
    assert "mtt::aux::position_last_layer_features" not in supported_outputs
    assert "mtt::aux::momentum_last_layer_features" not in supported_outputs

    capabilities = model.export().capabilities()
    assert "mtt::aux::S3_last_layer_features" in capabilities.outputs
    assert "mtt::aux::position_last_layer_features" not in capabilities.outputs
    assert "mtt::aux::momentum_last_layer_features" not in capabilities.outputs

    dtype = torch.float32
    system = System(
        types=torch.tensor([1, 6, 1]),
        positions=torch.randn(3, 3, dtype=dtype),
        cell=torch.eye(3, dtype=dtype),
        pbc=torch.tensor([True] * 3),
    )
    system = get_system_with_neighbor_lists(system, get_requested_neighbor_lists(model))
    system.add_data(
        "momentum",
        TensorMap(
            keys=Labels.single(),
            blocks=[
                TensorBlock(
                    values=torch.randn(3, 3, 1, dtype=dtype),
                    samples=Labels(
                        names=["system", "atom"],
                        values=torch.tensor([[0, 0], [0, 1], [0, 2]]),
                    ),
                    components=[Labels.range("xyz", 3)],
                    properties=Labels.range("momentum", 1),
                )
            ],
        ),
    )

    outputs = model(
        [system], {"mtt::aux::S3_last_layer_features": ModelOutput(sample_kind="atom")}
    )
    features = outputs["mtt::aux::S3_last_layer_features"].block()
    assert features.values.shape == (3, model.last_layer_feature_size)

    for name in ["position", "momentum"]:
        with pytest.raises(ValueError, match="has no last-layer features"):
            model(
                [system],
                {
                    f"mtt::aux::{name}_last_layer_features": ModelOutput(
                        sample_kind="atom"
                    )
                },
            )
