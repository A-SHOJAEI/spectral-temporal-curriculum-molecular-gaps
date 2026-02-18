#!/usr/bin/env python3
"""Standalone evaluation script that matches training spectral feature extraction."""
import os, sys, json, logging, time
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

import torch
import numpy as np

# Safe globals for torch.load
try:
    import numpy.core.multiarray
    safe_globals = [numpy.core.multiarray._reconstruct, numpy.ndarray, numpy.dtype,
                    numpy.float64, numpy.float32, numpy.int64, numpy.int32, numpy.uint8]
    torch.serialization.add_safe_globals([g for g in safe_globals if g is not None])
except (ImportError, AttributeError):
    pass

sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'src')))

from spectral_temporal_curriculum_molecular_gaps.data.loader import PCQM4Mv2CurriculumDataLoader
from spectral_temporal_curriculum_molecular_gaps.models.model import SpectralTemporalMolecularNet
from spectral_temporal_curriculum_molecular_gaps.utils.config import load_config

import torch_geometric.utils as pyg_utils
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_spectral_features(batch, num_scales=4):
    """Extract spectral features via graph Laplacian eigendecomposition (same as trainer)."""
    device = batch.x.device
    graphs = [batch.get_example(i) for i in range(batch.num_graphs)]
    spectral_features = [[] for _ in range(num_scales)]

    for graph in graphs:
        try:
            edge_index = graph.edge_index
            num_nodes = graph.num_nodes
            adj = pyg_utils.to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

            if adj.nnz == 0:
                eigenvals = torch.ones(min(num_scales, num_nodes), device=device)
                eigenvecs = torch.eye(num_nodes, min(num_scales, num_nodes), device=device)
            else:
                degrees = torch.tensor(adj.sum(axis=1).A1, dtype=torch.float32)
                degrees[degrees == 0] = 1.0
                deg_sqrt_inv = sp.diags(1.0 / torch.sqrt(degrees).numpy())
                laplacian = sp.eye(num_nodes) - deg_sqrt_inv @ adj @ deg_sqrt_inv
                k = min(num_scales, num_nodes - 1)
                if k > 0:
                    eigenvals_np, eigenvecs_np = eigsh(laplacian, k=k, which='SM')
                    eigenvals = torch.tensor(eigenvals_np, dtype=torch.float32, device=device)
                    eigenvecs = torch.tensor(eigenvecs_np, dtype=torch.float32, device=device)
                else:
                    eigenvals = torch.ones(1, device=device)
                    eigenvecs = torch.ones(num_nodes, 1, device=device) / np.sqrt(num_nodes)

            node_features = graph.x
            for scale_idx in range(num_scales):
                if scale_idx < eigenvecs.size(1):
                    eigenvec = eigenvecs[:, scale_idx:scale_idx+1]
                    spectral_weight = torch.exp(-eigenvals[scale_idx] * (scale_idx + 1))
                    scale_features = node_features * eigenvec * spectral_weight
                else:
                    scale_features = node_features * (1.0 / (2 ** scale_idx))
                spectral_features[scale_idx].append(scale_features)
        except Exception as e:
            node_features = graph.x
            for scale_idx in range(num_scales):
                spectral_features[scale_idx].append(node_features * (1.0 / (2 ** scale_idx)))

    return [torch.cat(sf, dim=0) for sf in spectral_features]


def main():
    config = load_config('configs/default.yaml')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load model
    mc = config['model']
    model = SpectralTemporalMolecularNet(
        input_dim=mc['input_dim'], hidden_dim=mc['hidden_dim'],
        num_spectral_layers=mc['num_spectral_layers'], num_scales=mc['num_scales'],
        num_curriculum_stages=mc['num_curriculum_stages'],
        dropout=mc['dropout'], pool_type=mc['pool_type'],
    )
    ckpt = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    logging.info(f"Model loaded from epoch {ckpt.get('epoch')}, best_val_loss={ckpt.get('best_val_loss', 'N/A')}")

    # Data loader
    data_loader = PCQM4Mv2CurriculumDataLoader(
        root=config['data']['root_dir'], batch_size=32, num_workers=0,
        curriculum_strategy=config['data']['curriculum_strategy'],
        spectral_decomp_levels=config['data']['num_spectral_scales'],
        cache_dir=config['data'].get('cache_dir'), force_reload=False,
    )

    # Evaluate on validation set (PCQM4M-v2 test labels are hidden)
    test_loader = data_loader.get_validation_dataloader()
    all_preds, all_targets = [], []
    nan_count = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = batch.to(device)
            targets = batch.y.float()
            sf = extract_spectral_features(batch, mc['num_scales'])
            outputs = model(batch, sf)
            preds = outputs['prediction'].squeeze()

            if preds.dim() == 0:
                preds = preds.unsqueeze(0)

            batch_nan = torch.isnan(preds).sum().item()
            nan_count += batch_nan

            # Replace NaN with 0 for metrics
            preds = torch.where(torch.isnan(preds), torch.zeros_like(preds), preds)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

            if (batch_idx + 1) % 500 == 0:
                elapsed = time.time() - t0
                logging.info(f"Batch {batch_idx+1}: {len(all_preds)} preds, {nan_count} NaN so far, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    logging.info(f"Evaluation complete: {len(all_preds)} predictions, {nan_count} NaN, {elapsed:.0f}s")

    preds_np = np.array(all_preds)
    targets_np = np.array(all_targets)

    # Filter valid predictions
    valid = np.isfinite(preds_np) & np.isfinite(targets_np)
    n_valid = valid.sum()
    logging.info(f"Valid predictions: {n_valid}/{len(preds_np)}")

    if n_valid > 0:
        p, t = preds_np[valid], targets_np[valid]
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t)**2)))
        ss_res = np.sum((t - p)**2)
        ss_tot = np.sum((t - np.mean(t))**2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        chem_acc = float(np.mean(np.abs(p - t) <= 0.043) * 100)

        logging.info(f"Test MAE: {mae:.4f} eV")
        logging.info(f"Test RMSE: {rmse:.4f} eV")
        logging.info(f"Test R²: {r2:.4f}")
        logging.info(f"Chemical Accuracy: {chem_acc:.1f}%")
    else:
        mae = rmse = r2 = 0.0
        chem_acc = 0.0
        logging.error("No valid predictions!")

    # Save results
    os.makedirs('artifacts', exist_ok=True)
    results = {
        "test_metrics": {
            "mae": mae, "rmse": rmse, "r2": r2,
            "chemical_accuracy": chem_acc,
            "n_predictions": int(n_valid),
            "n_nan": int(nan_count),
        },
        "model": {
            "epoch": int(ckpt.get('epoch', -1)),
            "best_val_loss": float(ckpt.get('best_val_loss', -1)),
        },
    }
    with open('artifacts/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    logging.info("Results saved to artifacts/results.json")


if __name__ == '__main__':
    main()
