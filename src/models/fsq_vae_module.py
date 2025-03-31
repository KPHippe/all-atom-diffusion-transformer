"""Copyright (c) Meta Platforms, Inc. and affiliates."""

import copy
from typing import Dict

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_scatter import scatter
from torchmetrics import MeanMetric

from src.eval.crystal_reconstruction import CrystalReconstructionEvaluator
from src.eval.mof_reconstruction import MOFReconstructionEvaluator
from src.eval.molecule_reconstruction import MoleculeReconstructionEvaluator
from src.models.quantizers.fsq import FSQ
from src.models.vae_module import VariationalAutoencoderLitModule
from src.utils import pylogger

log = pylogger.RankedLogger(__name__)


IDX_TO_DATASET = {
    0: "mp20",
    1: "qm9",
    2: "qmof150",
}
DATASET_TO_IDX = {
    "mp20": 0,  # periodic
    "qm9": 1,  # non-periodic
    "qmof150": 0,  # periodic
}


class FSQVAELitModule(VariationalAutoencoderLitModule):
    """LightningModule for autoencoding 3D atomic systems using FSQ instead of VAE.

    This is a modification of the VariationalAutoencoderLitModule to use FSQ instead of
    the diagonal Gaussian distribution.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_dim: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        scheduler_frequency: int,
        loss_weights: Dict,
        augmentations: DictConfig,
        visualization: DictConfig,
        compile: bool,
        fsq_levels: list = [8, 5, 5, 5],
        fsq_codebooks: int = 1,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # encoder and decoder models
        self.encoder = encoder
        self.decoder = decoder

        # FSQ quantizer instead of diagonal Gaussian
        self.fsq = FSQ(
            levels=fsq_levels,
            dim=latent_dim,
            num_codebooks=fsq_codebooks,
            return_indices=True,
        )

        # Linear layers for encoding/decoding (replacing quant_conv and post_quant_conv)
        self.pre_quant_conv = torch.nn.Linear(
            self.encoder.d_model, latent_dim, bias=False
        )
        self.post_quant_conv = torch.nn.Linear(
            latent_dim, self.decoder.d_model, bias=False
        )

        # weights for scaling loss functions per dataset type
        self.loss_weights = loss_weights
        self.loss_weights_atom_types = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_atom_types"].values()]),
            requires_grad=False,
        )
        self.loss_weights_lengths = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_lengths"].values()]),
            requires_grad=False,
        )
        self.loss_weights_angles = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_angles"].values()]),
            requires_grad=False,
        )
        self.loss_weights_frac_coords = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_frac_coords"].values()]),
            requires_grad=False,
        )
        self.loss_weights_pos = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_pos"].values()]),
            requires_grad=False,
        )
        # No KL divergence or commitment loss for FSQ, but keep param for compatibility
        self.loss_weights_kl = torch.nn.Parameter(
            torch.tensor([*self.loss_weights["loss_kl"].values()]),
            requires_grad=False,
        )

        # evaluator objects for computing metrics
        self.val_reconstruction_evaluators = {
            "mp20": CrystalReconstructionEvaluator(),
            "qm9": MoleculeReconstructionEvaluator(),
            "qmof150": MOFReconstructionEvaluator(),
        }
        self.test_reconstruction_evaluators = {
            "mp20": CrystalReconstructionEvaluator(),
            "qm9": MoleculeReconstructionEvaluator(),
            "qmof150": MOFReconstructionEvaluator(),
        }

        # metric objects for calculating and averaging across batches
        self.train_metrics = ModuleDict(
            {
                "loss": MeanMetric(),
                "loss_atom_types": MeanMetric(),
                "loss_lengths": MeanMetric(),
                "loss_angles": MeanMetric(),
                "loss_frac_coords": MeanMetric(),
                "loss_pos": MeanMetric(),
                "unscaled/loss_atom_types": MeanMetric(),
                "unscaled/loss_lengths": MeanMetric(),
                "unscaled/loss_angles": MeanMetric(),
                "unscaled/loss_frac_coords": MeanMetric(),
                "unscaled/loss_pos": MeanMetric(),
                "dataset_idx": MeanMetric(),
            }
        )
        self.val_metrics = ModuleDict(
            {
                "mp20": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "loss_atom_types": MeanMetric(),
                        "loss_lengths": MeanMetric(),
                        "loss_angles": MeanMetric(),
                        "loss_frac_coords": MeanMetric(),
                        "loss_pos": MeanMetric(),
                        "unscaled/loss_atom_types": MeanMetric(),
                        "unscaled/loss_lengths": MeanMetric(),
                        "unscaled/loss_angles": MeanMetric(),
                        "unscaled/loss_frac_coords": MeanMetric(),
                        "unscaled/loss_pos": MeanMetric(),
                        "match_rate": MeanMetric(),
                        "rms_dist": MeanMetric(),
                    }
                ),
                "qm9": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "loss_atom_types": MeanMetric(),
                        "loss_lengths": MeanMetric(),
                        "loss_angles": MeanMetric(),
                        "loss_frac_coords": MeanMetric(),
                        "loss_pos": MeanMetric(),
                        "unscaled/loss_atom_types": MeanMetric(),
                        "unscaled/loss_lengths": MeanMetric(),
                        "unscaled/loss_angles": MeanMetric(),
                        "unscaled/loss_frac_coords": MeanMetric(),
                        "unscaled/loss_pos": MeanMetric(),
                        "match_rate": MeanMetric(),
                        "rms_dist": MeanMetric(),
                    }
                ),
                "qmof150": ModuleDict(
                    {
                        "loss": MeanMetric(),
                        "loss_atom_types": MeanMetric(),
                        "loss_lengths": MeanMetric(),
                        "loss_angles": MeanMetric(),
                        "loss_frac_coords": MeanMetric(),
                        "loss_pos": MeanMetric(),
                        "unscaled/loss_atom_types": MeanMetric(),
                        "unscaled/loss_lengths": MeanMetric(),
                        "unscaled/loss_angles": MeanMetric(),
                        "unscaled/loss_frac_coords": MeanMetric(),
                        "unscaled/loss_pos": MeanMetric(),
                        "match_rate": MeanMetric(),
                        "rms_dist": MeanMetric(),
                    }
                ),
            }
        )
        self.test_metrics = copy.deepcopy(self.val_metrics)

    def encode(self, batch):
        encoded_batch = self.encoder(batch)

        # Apply pre-quantization layer
        latent_features = self.pre_quant_conv(encoded_batch["x"])

        # Store the original features for commitment loss
        encoded_batch["pre_quant_features"] = latent_features

        # Apply FSQ quantization
        quantized_features, indices = self.fsq(latent_features)

        # Store quantized features and indices
        encoded_batch["x"] = quantized_features
        encoded_batch["indices"] = indices

        return encoded_batch

    def decode(self, encoded_batch):
        # Apply post-quantization layer (no change from original)
        encoded_batch["x"] = self.post_quant_conv(encoded_batch["x"])
        out = self.decoder(encoded_batch)
        return out

    def forward(self, batch: Data):
        encoded_batch = self.encode(batch)
        out = self.decode(encoded_batch)
        return out, encoded_batch

    def reconstruction_criterion(
        self, batch: Data, out: Dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        # Atom types loss
        loss_atom_types = F.cross_entropy(
            out["atom_types"], batch.atom_types, reduction="none"
        )

        # Lattice lengths loss, after scaling by num_atoms**(1/3)
        loss_lengths = F.mse_loss(
            out["lengths"], batch.lengths_scaled, reduction="none"
        ).mean(dim=1)

        # Lattice angles loss, in radians
        loss_angles = F.mse_loss(
            out["angles"], batch.angles_radians, reduction="none"
        ).mean(dim=1)

        # Fractional coordinates loss
        loss_frac_coords = F.mse_loss(
            out["frac_coords"], batch.frac_coords, reduction="none"
        ).mean(dim=1)

        # Coordinates loss after zero-centering, use nm as unit (not A)
        pos_pred = out["pos"]
        pos_true = batch.pos / 10.0  # nm to A
        pos_mean_pred = scatter(pos_pred, batch.batch, dim=0, reduce="mean")[
            batch.batch
        ]
        pos_mean_true = scatter(pos_true, batch.batch, dim=0, reduce="mean")[
            batch.batch
        ]
        loss_pos = F.mse_loss(
            pos_pred - pos_mean_pred, pos_true - pos_mean_true, reduction="none"
        ).mean(dim=1)

        return {
            "loss_atom_types": loss_atom_types,
            "loss_lengths": loss_lengths,
            "loss_angles": loss_angles,
            "loss_frac_coords": loss_frac_coords,
            "loss_pos": loss_pos,
        }

    def criterion(
        self,
        batch: Data,
        encoded_batch: Dict[str, torch.Tensor],
        out: Dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Reconstruction loss only (no KL divergence or commitment loss for FSQ)
        loss_reconst = self.reconstruction_criterion(batch, out)

        # Assign loss_weights tensors based on dataset_idx attribute in batch
        weights_atom_types = self.loss_weights_atom_types[
            batch.dataset_idx[batch.batch]
        ]
        weights_lengths = self.loss_weights_lengths[batch.dataset_idx]
        weights_angles = self.loss_weights_angles[batch.dataset_idx]
        weights_frac_coords = self.loss_weights_frac_coords[
            batch.dataset_idx[batch.batch]
        ]
        weights_pos = self.loss_weights_pos[batch.dataset_idx[batch.batch]]

        loss = (
            (weights_atom_types * loss_reconst["loss_atom_types"]).mean()
            + (weights_lengths * loss_reconst["loss_lengths"]).mean()
            + (weights_angles * loss_reconst["loss_angles"]).mean()
            + (weights_frac_coords * loss_reconst["loss_frac_coords"]).mean()
            + (weights_pos * loss_reconst["loss_pos"]).mean()
        )

        return {
            "loss": loss,
            "loss_atom_types": weights_atom_types * loss_reconst["loss_atom_types"],
            "loss_lengths": weights_lengths * loss_reconst["loss_lengths"],
            "loss_angles": weights_angles * loss_reconst["loss_angles"],
            "loss_frac_coords": weights_frac_coords * loss_reconst["loss_frac_coords"],
            "loss_pos": weights_pos * loss_reconst["loss_pos"],
            "unscaled/loss_atom_types": loss_reconst["loss_atom_types"],
            "unscaled/loss_lengths": loss_reconst["loss_lengths"],
            "unscaled/loss_angles": loss_reconst["loss_angles"],
            "unscaled/loss_frac_coords": loss_reconst["loss_frac_coords"],
            "unscaled/loss_pos": loss_reconst["loss_pos"],
        }
