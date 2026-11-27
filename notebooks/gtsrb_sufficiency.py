"""
Sufficiency / comprehensiveness experiment for FAME attribution maps (GTSRB).

Consumes the binary attribution maps saved by `gtsrb_fc.py` (one `<index>.npy`
per test sample, 1 = important pixel / explanation E, 0 = freed pixel) and
measures, for the same Keras classifier used by FAME:

- sufficiency: mean softmax when only the explanation E is kept and the
  background B is perturbed with Gaussian noise;
- comprehensiveness: mean softmax when the explanation E is perturbed and the
  background B is kept;
- a random baseline using a random pixel mask with the same cardinality as E;
- optionally (--pgd_rho), adversarial sufficiency: softmax after an untargeted
  l2 PGD attack with budget rho restricted to the background B.

The noise scheme replicates robust-XAI `sufficiency.py`: each perturbed
coordinate receives i.i.d. Gaussian noise with standard deviation
`noise_level / sqrt(n_perturbed_pixels)`, and the softmax is averaged over
`n_softmax_samples` Monte Carlo draws.

The output folder (metadata.json + samples/*.json) is compatible with
robust-XAI `aggregate_sufficiency.py`.
"""

import os
from argparse import ArgumentParser, Namespace

os.environ["KERAS_BACKEND"] = "torch"
from shared_gpu import configure_shared_gpu_from_argv
configure_shared_gpu_from_argv()

import json
import pickle
import shutil
import time
from pathlib import Path
from typing import List

import keras
import numpy as np
import torch
from keras.models import load_model
from tqdm import tqdm

IMG_SIDE = 32
CHANNEL = 3
N_PIXELS = IMG_SIDE * IMG_SIDE  # 1024 feature groups (pixels)
N_INPUT = N_PIXELS * CHANNEL    # 3072 flat input coordinates (channels_last)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_softmax(model: keras.models.Model, batch: np.ndarray) -> np.ndarray:
    # The GTSRB CNN has a linear output layer, so apply the softmax here.
    return softmax(model.predict(batch, verbose=0))


def mean_softmax_under_noise(
    model: keras.models.Model,
    sample: np.ndarray,
    noise_level: float,
    pixels_to_perturb: np.ndarray,
    n_monte_carlo: int,
    rng: np.random.Generator,
) -> List[float]:
    """
    Mean softmax of the classifier over Monte Carlo Gaussian perturbations
    restricted to the pixels marked with 1 in `pixels_to_perturb` (shape
    (N_PIXELS,)). All 3 channels of a perturbed pixel are perturbed
    (channels_last layout: pixel i -> flat coordinates 3i, 3i+1, 3i+2).
    """
    n_perturbed_pixels = float(pixels_to_perturb.sum())
    if n_perturbed_pixels == 0:
        # Nothing to perturb: return the clean softmax of the classifier.
        return predict_softmax(model, sample[None])[0].tolist()

    assert noise_level > 0, "Noise level must be positive."
    std = noise_level / np.sqrt(n_perturbed_pixels)
    coord_mask = np.repeat(pixels_to_perturb.astype(np.float32), CHANNEL)
    noise = rng.standard_normal((n_monte_carlo, N_INPUT)).astype(np.float32)
    perturbed = sample[None] + noise * (coord_mask * std)
    probs = predict_softmax(model, perturbed)
    return probs.mean(axis=0).tolist()


def _model_torch_device(model: keras.models.Model) -> torch.device:
    """Device on which the Keras (torch backend) model weights live."""
    try:
        return model.weights[0].value.device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pgd_l2_attack_softmax(
    model: keras.models.Model,
    sample: np.ndarray,
    label: int,
    pixels_to_perturb: np.ndarray,
    rho: float,
    n_steps: int,
) -> List[float]:
    """
    Untargeted l2 PGD attack restricted to the pixels marked with 1 in
    `pixels_to_perturb` (shape (N_PIXELS,), all 3 channels of a perturbed pixel
    are attacked), with l2 budget `rho` over all perturbed coordinates.
    Maximizes the cross-entropy w.r.t. `label` (the original prediction) and
    returns the softmax of the attacked sample. Gradients flow through the
    Keras model thanks to the torch backend.
    """
    if float(pixels_to_perturb.sum()) == 0 or rho <= 0:
        # Nothing to perturb: return the clean softmax of the classifier.
        return predict_softmax(model, sample[None])[0].tolist()

    device = _model_torch_device(model)
    coord_mask = torch.as_tensor(
        np.repeat(pixels_to_perturb.astype(np.float32), CHANNEL), device=device
    )
    x = torch.as_tensor(sample[None], device=device)
    label_tensor = torch.tensor([label], device=device)
    step_size = 2.5 * rho / n_steps
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(n_steps):
        logits = model(x + delta, training=False)
        loss = torch.nn.functional.cross_entropy(logits, label_tensor)
        grad = torch.autograd.grad(loss, delta)[0] * coord_mask
        grad_norm = grad.flatten().norm().clamp_min(1e-12)
        delta = delta + step_size * grad / grad_norm
        # Project onto the l2 ball of radius rho (restricted to the mask).
        delta_norm = delta.flatten().norm()
        if delta_norm.item() > rho:
            delta = delta * (rho / delta_norm)
        delta = (delta * coord_mask).detach().requires_grad_(True)
    with torch.no_grad():
        probs = torch.softmax(model(x + delta, training=False), dim=1)
    return probs[0].cpu().tolist()


def generate_random_mask(explanation_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random binary pixel mask with the same cardinality as `explanation_mask`."""
    n_important = int(explanation_mask.sum())
    flat_mask = np.zeros(explanation_mask.size, dtype=np.float32)
    random_indices = rng.permutation(explanation_mask.size)[:n_important]
    flat_mask[random_indices] = 1.0
    return flat_mask.reshape(explanation_mask.shape)


def to_class_indices(y: np.ndarray) -> np.ndarray:
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1).astype("int64")
    return y.astype("int64")


def process_sample(
    model: keras.models.Model,
    sample: np.ndarray,
    numerical_label: int,
    explanation_mask: np.ndarray,
    out_filepath: Path,
    noise_level: float,
    n_softmax_samples: int,
    rng: np.random.Generator,
    pgd_rho: float,
    pgd_steps: int,
) -> dict:
    json_object = {}
    json_object["numerical_label"] = int(numerical_label)
    json_object["sample_height"] = IMG_SIDE
    json_object["sample_width"] = IMG_SIDE
    clean_probs = predict_softmax(model, sample[None])[0]
    json_object["original_classifier_prediction"] = int(np.argmax(clean_probs))
    json_object["clean_softmax"] = clean_probs.tolist()
    json_object["explanation_size"] = int(explanation_mask.sum())
    json_object["n_total_dimensions"] = int(explanation_mask.size)

    # Experiment 1 (sufficiency): perturb everything except the explanation
    # and measure the mean confidence of the classifier per class.
    json_object["sufficiency_softmax"] = mean_softmax_under_noise(
        model=model,
        sample=sample,
        noise_level=noise_level,
        pixels_to_perturb=1.0 - explanation_mask,
        n_monte_carlo=n_softmax_samples,
        rng=rng,
    )

    # Experiment 2 (comprehensiveness): perturb only the important features
    # (the explanation) and measure the mean confidence per class.
    json_object["perturb_explanation_softmax"] = mean_softmax_under_noise(
        model=model,
        sample=sample,
        noise_level=noise_level,
        pixels_to_perturb=explanation_mask,
        n_monte_carlo=n_softmax_samples,
        rng=rng,
    )

    # Random baseline: a random binary mask with exactly the same number of
    # important pixels as the explanation, evaluated the same way.
    random_explanation_mask = generate_random_mask(explanation_mask, rng)
    json_object["random_sufficiency_softmax"] = mean_softmax_under_noise(
        model=model,
        sample=sample,
        noise_level=noise_level,
        pixels_to_perturb=1.0 - random_explanation_mask,
        n_monte_carlo=n_softmax_samples,
        rng=rng,
    )
    json_object["random_perturb_explanation_softmax"] = mean_softmax_under_noise(
        model=model,
        sample=sample,
        noise_level=noise_level,
        pixels_to_perturb=random_explanation_mask,
        n_monte_carlo=n_softmax_samples,
        rng=rng,
    )

    # Adversarial sufficiency: l2 PGD attack with budget pgd_rho restricted to
    # the non-important pixels (and the random-mask baseline counterpart).
    if pgd_rho is not None and pgd_rho > 0:
        attacked_label = json_object["original_classifier_prediction"]
        json_object["pgd_sufficiency_softmax"] = pgd_l2_attack_softmax(
            model=model,
            sample=sample,
            label=attacked_label,
            pixels_to_perturb=1.0 - explanation_mask,
            rho=pgd_rho,
            n_steps=pgd_steps,
        )
        json_object["random_pgd_sufficiency_softmax"] = pgd_l2_attack_softmax(
            model=model,
            sample=sample,
            label=attacked_label,
            pixels_to_perturb=1.0 - random_explanation_mask,
            rho=pgd_rho,
            n_steps=pgd_steps,
        )

    with open(out_filepath, "w") as f:
        json.dump(json_object, f, indent=4)
    return json_object


def main(args: Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    maps_dir = Path(args.maps_dir).resolve()
    map_files = sorted(maps_dir.glob("*.npy"), key=lambda p: int(p.stem))
    if not map_files:
        raise FileNotFoundError(f"No attribution maps (*.npy) found in {maps_dir}")
    print(f"Found {len(map_files)} attribution maps in {maps_dir}")

    with open(os.path.join("models", "gtsrb.pickle"), "rb") as handle:
        data = pickle.load(handle)
    x_test = data["x_test"].astype("float32") / 255
    x_test = np.reshape(x_test, (-1, N_INPUT))
    y_test = to_class_indices(data["y_test"])

    model = load_model("./models/xairobas_gtsrb-cnn.keras")

    out_folder = Path(args.out_folder) if args.out_folder is not None \
        else Path("./results/sufficiency") / f"{maps_dir.name}_{int(time.time())}"
    print(f"Saving results to {out_folder}")
    if out_folder.exists():
        shutil.rmtree(out_folder)
    out_folder_samples = out_folder / "samples"
    out_folder_samples.mkdir(parents=True, exist_ok=True)

    metadata = vars(args).copy()
    metadata["mask"] = "fame"
    metadata["dataset"] = "gtsrb"
    metadata["radius_threshold"] = None
    metadata["maps_dir"] = str(maps_dir)
    metadata["out_folder"] = str(out_folder)
    metadata["n_maps"] = len(map_files)
    with open(out_folder / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    for map_file in tqdm(map_files, desc="Processing samples"):
        sample_index = int(map_file.stem)
        # 1 = important pixel (explanation), 0 = freed / robustness certified.
        explanation_mask = np.load(map_file).astype(np.float32)
        if explanation_mask.size != N_PIXELS:
            raise ValueError(
                f"Unexpected map size {explanation_mask.size} in {map_file}, expected {N_PIXELS}"
            )
        if args.invert_mask:
            explanation_mask = 1.0 - explanation_mask
        process_sample(
            model=model,
            sample=x_test[sample_index],
            numerical_label=int(y_test[sample_index]),
            explanation_mask=explanation_mask,
            out_filepath=out_folder_samples / f"{sample_index:05d}.json",
            noise_level=args.noise_level,
            n_softmax_samples=args.n_softmax_samples,
            rng=rng,
            pgd_rho=args.pgd_rho,
            pgd_steps=args.pgd_steps,
        )
    print(f"Done. Results in {out_folder}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Sufficiency / comprehensiveness test for FAME attribution maps on GTSRB.")
    parser.add_argument("--maps_dir", required=True, type=str, help="Folder with the <index>.npy attribution maps saved by gtsrb_fc.py.")
    parser.add_argument("--noise_level", required=False, type=float, default=15.0, help="Noise level for the sufficiency perturbations (same scheme as robust-XAI, default matches its GTSRB setting).")
    parser.add_argument("--n_softmax_samples", required=False, type=int, default=100, help="Number of Monte Carlo noise draws to average the softmax over.")
    parser.add_argument("--pgd_rho", required=False, type=float, default=None, help="l2 budget of the PGD attack on the non-important pixels. If not specified, the PGD experiment is skipped.")
    parser.add_argument("--pgd_steps", required=False, type=int, default=50, help="Number of PGD steps (step size is 2.5 * pgd_rho / pgd_steps).")
    parser.add_argument("--out_folder", required=False, type=str, default=None, help="Output folder for the results.")
    parser.add_argument("--seed", required=False, type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--invert_mask", action="store_true", help="Invert the attribution mask if set (treat freed pixels as the explanation).")
    parser.add_argument(
        "--shared_gpus",
        action="store_true",
        help="Select the GPU with the most free memory and mask visibility to that GPU only.",
    )
    args = parser.parse_args()
    main(args)
