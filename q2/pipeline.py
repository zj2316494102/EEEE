from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from . import MAX_LENGTH, MODALITIES
from .baselines import ConcatMLP, restrict_to_modality
from .config import Q2Config, dump_yaml, load_config
from .data import (
    DatasetBundle,
    NormalizationStats,
    SplitData,
    fit_normalization,
    load_aligned_dataset,
    save_normalization,
    sha256_file,
)
from .evaluation import (
    aggregate_scenario_metrics,
    evaluate_scenarios,
    evaluate_split,
    majority_and_mean_baseline,
)
from .inference import load_attachment3_samples, write_attachment3_predictions
from .masking import MaskScenario, build_fixed_validation_scenarios, mask_statistics
from .model import RobustGatedTemporalFusion
from .training import TrainingResult, class_weights_from_training, train_one_model


LOGGER = logging.getLogger("q2")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: list[str] | None = None) -> None:
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            clean: dict[str, Any] = {}
            for key in fieldnames:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                if value is None:
                    value = "NA"
                clean[key] = value
            writer.writerow(clean)


def _timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_run_id(value: str) -> str:
    if not value or not all(character.isalnum() or character in "._-" for character in value):
        raise ValueError("run_id must contain only letters, digits, '.', '_' or '-'")
    return value


def _resolve_resume_directory(value: str, run_root: Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        from_cwd = (Path.cwd() / candidate).resolve()
        candidate = from_cwd if from_cwd.exists() else (run_root / candidate).resolve()
    return candidate.resolve()


def _select_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _make_selection_scenarios(valid: SplitData, config: Q2Config, compact: bool) -> list[MaskScenario]:
    if compact:
        return build_fixed_validation_scenarios(
            valid.mask_array(),
            fractions=(0.20, 0.40),
            positions=("middle",),
            groups=((0,), (1,), (2,), (0, 1, 2)),
            relation_modes=("overlap",),
            seeds=(int(config.get("random_seeds", [2026])[0]),),
        )
    return build_fixed_validation_scenarios(
        valid.mask_array(),
        fractions=tuple(float(item) for item in config.get("selection_missing_fractions", [0.20])),
        positions=tuple(config.get("selection_missing_positions", ["middle"])),
        groups=tuple(tuple(int(index) for index in group) for group in config.get("selection_missing_groups", [[0], [1], [2], [0, 1, 2]])),
        relation_modes=tuple(config.get("selection_relation_modes", ["overlap"])),
        seeds=(int(config.get("random_seeds", [2026])[0]),),
    )


def _make_full_scenarios(valid: SplitData, config: Q2Config, compact: bool) -> list[MaskScenario]:
    if compact:
        return _make_selection_scenarios(valid, config, compact=True)
    return build_fixed_validation_scenarios(
        valid.mask_array(),
        fractions=tuple(float(item) for item in config.get("validation_missing_fractions", [0.10, 0.20, 0.40, 0.60])),
        positions=tuple(config.get("validation_missing_positions", ["front", "middle", "back"])),
        groups=tuple(tuple(int(index) for index in group) for group in config.get("validation_missing_groups", [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]])),
        relation_modes=tuple(config.get("validation_relation_modes", ["overlap", "stagger"])),
        seeds=tuple(int(seed) for seed in config.get("validation_mask_seeds", [2026, 42, 3407])),
    )


def _model_config(config: Q2Config) -> dict[str, object]:
    values = config.model_kwargs
    values.update({"input_dims": list(config.input_dims), "max_length": config.max_length})
    return values


def _training_config(config: Q2Config, args: argparse.Namespace) -> dict[str, object]:
    values = dict(config.values)
    if args.max_epochs is not None:
        values["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        values["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        values["learning_rate"] = args.learning_rate
    return values


def _save_checkpoint(
    model: RobustGatedTemporalFusion,
    path: Path,
    seed: int,
    bundle: DatasetBundle,
    config: Q2Config,
) -> None:
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    checkpoint = {
        "model_config": model.config_dict(),
        "state_dict": state,
        "seed": int(seed),
        "feature_version": config.feature_version,
        "label_mapping": bundle.label_mapping,
        "config_hash": config.config_hash,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _load_checkpoint(path: Path, device: torch.device) -> RobustGatedTemporalFusion:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    from .model import build_model

    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def _seed_summary_row(result: TrainingResult, validation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = next((row for row in validation_rows if row.get("scenario") == "complete"), validation_rows[0])
    return {
        "seed": result.seed,
        "best_epoch": result.best_epoch,
        "best_training_selection_score": result.best_score,
        "validation_selection_score": aggregate_scenario_metrics(validation_rows).get("mean_composite_score"),
        "validation_accuracy": complete.get("accuracy"),
        "validation_macro_f1": complete.get("macro_f1"),
        "validation_weighted_f1": complete.get("weighted_f1"),
        "validation_mae": complete.get("mae"),
        "validation_pearson": complete.get("pearson"),
    }


def _environment_snapshot(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }


def _mask_audit_rows(base_masks: np.ndarray, scenarios: list[MaskScenario]) -> list[dict[str, Any]]:
    """Persist compact fixed-mask audit rows without duplicating raw feature arrays."""

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        summary = mask_statistics(base_masks, scenario.masks)
        for modality in MODALITIES:
            index = MODALITIES.index(modality)
            missing = base_masks[:, index] & ~scenario.masks[:, index]
            starts: list[int] = []
            ends: list[int] = []
            for sample in missing:
                positions = np.flatnonzero(sample)
                if len(positions):
                    starts.append(int(positions[0]))
                    ends.append(int(positions[-1]) + 1)
            modality_summary = summary["modalities"][modality]
            rows.append(
                {
                    "scenario": scenario.name,
                    "mask_seed": scenario.seed,
                    "missing_modalities": "+".join(scenario.missing_modalities),
                    "missing_fraction": scenario.fraction,
                    "missing_position": scenario.position,
                    "relation": scenario.relation,
                    "modality": modality,
                    "samples_with_missing": int(np.any(missing, axis=1).sum()),
                    "interval_start_min": min(starts) if starts else "NA",
                    "interval_start_max": max(starts) if starts else "NA",
                    "interval_end_min": min(ends) if ends else "NA",
                    "interval_end_max": max(ends) if ends else "NA",
                    "missing_fraction_mean": modality_summary["missing_fraction_mean"],
                    "missing_fraction_max": modality_summary["missing_fraction_max"],
                    "longest_missing_run_mean_over_50": modality_summary["longest_missing_run_mean_over_50"],
                }
            )
    return rows


def _run_ablations(
    train: SplitData,
    valid: SplitData,
    config: Q2Config,
    training_config: Mapping[str, object],
    class_weights: np.ndarray,
    device: torch.device,
    selection_scenarios: list[MaskScenario],
    seed: int,
    checkpoint_root: Path | None = None,
    resume_requested: bool = False,
    checkpoint_metadata: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    variants = [
        ("no_contiguous_block_augmentation", {}, False, "contiguous"),
        ("random_point_augmentation", {}, True, "random_point"),
        ("fixed_mean_fusion", {"fusion": "mean"}, True, "contiguous"),
        ("no_explicit_mask_embedding", {"use_mask_input": False}, True, "contiguous"),
        ("no_temporal_transformer", {"use_transformer": False}, True, "contiguous"),
    ]
    rows: list[dict[str, Any]] = []

    def checkpoint_args(stage_name: str) -> dict[str, Any]:
        if checkpoint_root is None:
            return {}
        checkpoint_path = checkpoint_root / f"seed_{seed}" / f"{stage_name}.pt"
        metadata = dict(checkpoint_metadata or {})
        metadata["stage"] = stage_name
        return {
            "checkpoint_path": checkpoint_path,
            "resume_path": checkpoint_path if resume_requested else None,
            "checkpoint_metadata": metadata,
        }

    # Single-modality baselines use the same projection/temporal budget while
    # making the other two availability masks permanently false.
    for modality in MODALITIES:
        single_train = restrict_to_modality(train, modality)
        single_valid = restrict_to_modality(valid, modality)
        complete_scenario = MaskScenario(
            name="complete",
            masks=single_valid.mask_array(),
            missing_modalities=(),
            fraction=0.0,
            position="none",
            relation="none",
            seed=None,
        )
        result = train_one_model(
            single_train,
            single_valid,
            _model_config(config),
            training_config,
            seed=seed,
            device=device,
            class_weights=class_weights,
            use_augmentation=False,
            selection_scenarios=[complete_scenario],
            **checkpoint_args(f"single_modality_{modality}"),
        )
        metrics, _ = evaluate_split(
            result.model,
            single_valid,
            device=device,
            batch_size=int(training_config.get("batch_size", 64)),
        )
        rows.append({"variant": f"single_modality_{modality}", "seed": seed, **metrics})

    concat_result = train_one_model(
        train,
        valid,
        _model_config(config),
        training_config,
        seed=seed,
        device=device,
        class_weights=class_weights,
        use_augmentation=False,
        selection_scenarios=selection_scenarios,
        model_factory=lambda: ConcatMLP(
            input_dims=tuple(config.input_dims),
            max_length=config.max_length,
            hidden_dim=int(config.get("concat_mlp_hidden_dim", 256)),
            dropout=float(config.get("dropout", 0.15)),
        ),
        **checkpoint_args("concat_mlp"),
    )
    concat_metrics, _ = evaluate_split(
        concat_result.model,
        valid,
        device=device,
        batch_size=int(training_config.get("batch_size", 64)),
    )
    rows.append({"variant": "concat_mlp", "seed": seed, **concat_metrics})

    for name, overrides, use_aug, aug_mode in variants:
        LOGGER.info("Ablation started: %s", name)
        model_config = _model_config(config)
        model_config.update(overrides)
        result = train_one_model(
            train,
            valid,
            model_config,
            training_config,
            seed=seed,
            device=device,
            class_weights=class_weights,
            use_augmentation=use_aug,
            augmentation_mode=aug_mode,
            selection_scenarios=selection_scenarios,
            **checkpoint_args(name),
        )
        metrics, _ = evaluate_split(result.model, valid, device=device, batch_size=int(training_config.get("batch_size", 64)))
        rows.append({"variant": name, **metrics, "best_epoch": result.best_epoch, "seed": seed})
    return rows


def run_pipeline(args: argparse.Namespace) -> Path:
    config = load_config(args.config)
    run_root = Path(args.run_root).resolve() if args.run_root else config.resolve(config.get("run_root", "runs"))
    resume_requested = bool(args.resume)
    if resume_requested:
        run_dir = _resolve_resume_directory(args.resume, run_root)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {run_dir}")
        run_id = _safe_run_id(run_dir.name)
        if args.run_id and _safe_run_id(args.run_id) != run_id:
            raise ValueError(f"--run-id {args.run_id!r} does not match resume directory {run_id!r}")
        if (run_dir / "DONE").exists():
            raise RuntimeError(f"Run {run_id} is already complete; use a new run_id for a fresh run")
    else:
        run_id = _safe_run_id(args.run_id or _timestamp_run_id())
        run_dir = run_root / run_id
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "run_state.json"
    if resume_requested and state_path.exists():
        saved_state = json.loads(state_path.read_text(encoding="utf-8"))
        if saved_state.get("config_hash") != config.config_hash:
            raise RuntimeError(
                "Resume configuration hash mismatch: "
                f"saved={saved_state.get('config_hash')}, current={config.config_hash}"
            )
        saved_limit = saved_state.get("sample_limit")
        if args.limit is None and saved_limit is not None:
            args.limit = int(saved_limit)
        elif "sample_limit" in saved_state and saved_limit != args.limit:
            raise RuntimeError(
                f"Resume sample limit mismatch: saved={saved_limit}, current={args.limit}"
            )
    elif not resume_requested:
        _write_json(
            state_path,
            {
                "run_id": run_id,
                "config_hash": config.config_hash,
                "feature_version": config.feature_version,
                "sample_limit": args.limit,
                "created_at": datetime.now().isoformat(),
            },
        )
    else:
        LOGGER.warning("Resume directory has no run_state.json; checkpoint metadata will still be checked")
    (run_dir / "FAILED").unlink(missing_ok=True)
    (run_dir / "RUNNING").write_text(datetime.now().isoformat(), encoding="utf-8")
    output_dir = run_dir / "outputs" / "问题2"
    model_dir = output_dir / "model"
    validation_dir = output_dir / "validation"
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "q2.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
        force=True,
    )
    LOGGER.info("Question 2 run started: %s", run_id)
    started = time.time()
    try:
        device = _select_device(args.device)
        LOGGER.info("Using device: %s", device)
        feature_path = config.path_for("aligned_feature_path")
        label_path = config.resolve(config.get("label_path", "")) if config.get("label_path") else None
        bundle = load_aligned_dataset(
            feature_path,
            label_path=label_path,
            max_length=config.max_length,
            limit=args.limit,
        )
        if args.skip_input_hash:
            input_hash = None
        else:
            LOGGER.info("Hashing aligned feature file: %s", feature_path)
            input_hash = sha256_file(feature_path)
        stats = fit_normalization(bundle.train, feature_version=config.feature_version)
        train = stats.transform(bundle.train, name="train_normalized")
        valid = stats.transform(bundle.valid, name="valid_normalized")
        test = stats.transform(bundle.test, name="test_normalized")
        save_normalization(stats, model_dir / "normalization.npz")
        dump_yaml(config.values, output_dir / "config.yaml")

        train_config = _training_config(config, args)
        model_config = _model_config(config)
        seeds = [int(seed) for seed in (args.seeds or config.get("random_seeds", [2026, 42, 3407]))]
        if args.smoke:
            seeds = seeds[:1]
            train_config["max_epochs"] = min(int(train_config.get("max_epochs", 100)), 2)
            train_config["early_stopping_patience"] = min(int(train_config.get("early_stopping_patience", 12)), 2)
        class_weights = class_weights_from_training(
            train, enabled=bool(config.get("use_class_weights", True))
        )
        selection_scenarios = _make_selection_scenarios(valid, config, compact=args.compact_validation or args.smoke)
        use_distillation = (bool(config.get("distillation_enabled", False)) or args.use_distillation) and not args.no_distillation
        if use_distillation:
            train_config["distill_weight"] = float(config.get("distill_weight", 0.20))

        checkpoint_metadata = {
            "config_hash": config.config_hash,
            "feature_version": config.feature_version,
            "sample_limit": args.limit,
            "run_id": run_id,
        }
        seed_results: list[dict[str, Any]] = []
        trained: list[tuple[TrainingResult, list[dict[str, Any]]]] = []
        for seed in seeds:
            LOGGER.info("Training seed %s/%s", seed, seeds)
            seed_checkpoint_dir = checkpoint_root / f"seed_{seed}"
            student_checkpoint = seed_checkpoint_dir / "student.pt"
            teacher = None
            if use_distillation:
                LOGGER.info("Training complete-input teacher for seed %s", seed)
                teacher_checkpoint = seed_checkpoint_dir / "teacher.pt"
                teacher_result = train_one_model(
                    train,
                    valid,
                    model_config,
                    train_config,
                    seed=seed,
                    device=device,
                    class_weights=class_weights,
                    use_augmentation=False,
                    selection_scenarios=[selection_scenarios[0]],
                    checkpoint_path=teacher_checkpoint,
                    resume_path=teacher_checkpoint if resume_requested else None,
                    checkpoint_metadata={
                        **checkpoint_metadata,
                        "stage": f"teacher_seed_{seed}",
                    },
                )
                teacher = teacher_result.model
                if teacher_result.resumed:
                    LOGGER.info(
                        "Resumed teacher seed %s from epoch %s",
                        seed,
                        teacher_result.resume_epoch,
                    )
            result = train_one_model(
                train,
                valid,
                model_config,
                train_config,
                seed=seed,
                device=device,
                class_weights=class_weights,
                use_augmentation=True,
                teacher=teacher,
                selection_scenarios=selection_scenarios,
                checkpoint_path=student_checkpoint,
                resume_path=student_checkpoint if resume_requested else None,
                checkpoint_metadata={
                    **checkpoint_metadata,
                    "stage": f"student_seed_{seed}",
                },
            )
            if result.resumed:
                LOGGER.info(
                    "Resumed student seed %s from epoch %s",
                    seed,
                    result.resume_epoch,
                )
            rows, _ = evaluate_scenarios(
                result.model,
                valid,
                selection_scenarios,
                device=device,
                batch_size=int(train_config.get("batch_size", 64)),
            )
            seed_results.append(_seed_summary_row(result, rows))
            trained.append((result, rows))
        if not trained:
            raise RuntimeError("No model was trained")
        selected_index = int(np.argmax([aggregate_scenario_metrics(rows).get("mean_composite_score") or result.best_score for result, rows in trained]))
        selected_result, selected_selection_rows = trained[selected_index]
        selected_seed = selected_result.seed
        LOGGER.info("Selected seed %s with validation score %.6f", selected_seed, selected_result.best_score)
        full_scenarios = _make_full_scenarios(valid, config, compact=args.compact_validation or args.smoke)
        validation_rows, confusion_rows = evaluate_scenarios(
            selected_result.model,
            valid,
            full_scenarios,
            device=device,
            batch_size=int(train_config.get("batch_size", 64)),
        )
        test_metrics, _ = evaluate_split(
            selected_result.model,
            test,
            device=device,
            batch_size=int(train_config.get("batch_size", 64)),
        )
        valid_complete = next(row for row in validation_rows if row["scenario"] == "complete")
        baseline_valid = majority_and_mean_baseline(valid)
        baseline_test = majority_and_mean_baseline(test)

        model_path = model_dir / "robust_student.pt"
        _save_checkpoint(selected_result.model, model_path, selected_seed, bundle, config)
        _write_csv(
            validation_dir / "metrics_by_scenario.csv",
            validation_rows,
            fieldnames=[
                "scenario", "missing_modalities", "missing_fraction", "missing_position", "relation", "mask_seed",
                "sample_count", "accuracy", "macro_f1", "weighted_f1", "mae", "pearson",
                "f1_negative", "f1_neutral", "f1_positive", "all_unavailable_count", "composite_score",
                "gate_weight_mean", "mask_statistics",
            ],
        )
        _write_csv(
            validation_dir / "confusion_matrix.csv",
            confusion_rows,
            fieldnames=["scenario", "true_class", "predicted_class", "count"],
        )
        _write_csv(validation_dir / "mask_scenarios.csv", _mask_audit_rows(valid.mask_array(), full_scenarios))
        _write_csv(validation_dir / "seed_results.csv", seed_results)
        _write_csv(validation_dir / "training_history.csv", selected_result.history)
        _write_json(validation_dir / "test_metrics.json", test_metrics)
        _write_json(validation_dir / "baseline_metrics.json", {"valid": baseline_valid, "test": baseline_test})

        ablation_rows = [
            {"variant": "majority_and_mean_baseline_valid", **baseline_valid},
            {"variant": "robust_dynamic_gate_complete", **valid_complete},
        ]
        if args.run_ablations and not args.smoke:
            ablation_rows.extend(
                _run_ablations(
                    train,
                    valid,
                    config,
                    train_config,
                    class_weights,
                    device,
                    selection_scenarios,
                    selected_seed,
                    checkpoint_root=checkpoint_root / "ablations",
                    resume_requested=resume_requested,
                    checkpoint_metadata=checkpoint_metadata,
                )
            )
        _write_csv(validation_dir / "ablation_results.csv", ablation_rows)

        inference_manifest = None
        if not args.skip_attachment3:
            attachment3 = load_attachment3_samples(config.path_for("attachment3_aligned_dir"), max_length=config.max_length)
            inference_manifest = write_attachment3_predictions(
                selected_result.model,
                attachment3,
                stats,
                output_dir,
                device=device,
                batch_size=int(train_config.get("batch_size", 64)),
                model_path=model_path,
                normalization_path=model_dir / "normalization.npz",
            )

        checkpoint_files = [
            str(path.relative_to(run_dir))
            for path in sorted(checkpoint_root.rglob("*.pt"))
        ]
        run_manifest = {
            "run_id": run_id,
            "status": "DONE",
            "resumed": resume_requested,
            "feature_version": config.feature_version,
            "feature_file": str(feature_path),
            "feature_file_sha256": input_hash,
            "label_file": None if label_path is None else str(label_path),
            "label_mapping": bundle.label_mapping,
            "config_hash": config.config_hash,
            "device": str(device),
            "seeds": seeds,
            "selected_seed": selected_seed,
            "selected_best_epoch": selected_result.best_epoch,
            "class_weights": class_weights.tolist(),
            "distillation_enabled": use_distillation,
            "dataset_metadata": bundle.metadata,
            "sample_counts": {"train": train.size, "valid": valid.size, "test": test.size},
            "selected_validation_summary": aggregate_scenario_metrics(validation_rows),
            "test_metrics_file": str(validation_dir / "test_metrics.json"),
            "model_file": str(model_path),
            "model_file_sha256": sha256_file(model_path),
            "normalization_file": str(model_dir / "normalization.npz"),
            "normalization_file_sha256": sha256_file(model_dir / "normalization.npz"),
            "checkpoint_dir": str(checkpoint_root),
            "checkpoint_files": checkpoint_files,
            "inference_manifest": inference_manifest,
            "elapsed_seconds": round(time.time() - started, 2),
        }
        _write_json(output_dir / "run_manifest.json", run_manifest)
        _write_json(run_dir / "environment.json", _environment_snapshot(device))
        readme = (
            "# 问题2运行结果\n\n"
            f"- run_id: `{run_id}`\n"
            f"- feature_version: `{config.feature_version}`\n"
            f"- selected_seed: `{selected_seed}`\n"
            f"- input mask rule: `{bundle.metadata['mask_ambiguity_note']}`\n"
            "- validation metrics are label-based; attachment 3 output has no accuracy claim.\n"
            "- `robust_student.pt` is the inference checkpoint; resumable training checkpoints are under `../../checkpoints/`.\n"
        )
        (output_dir / "README.md").write_text(readme, encoding="utf-8")
        _write_json(
            state_path,
            {
                "run_id": run_id,
                "config_hash": config.config_hash,
                "feature_version": config.feature_version,
                "sample_limit": args.limit,
                "status": "DONE",
                "completed_at": datetime.now().isoformat(),
            },
        )
        (run_dir / "DONE").write_text(datetime.now().isoformat(), encoding="utf-8")
        (run_dir / "FAILED").unlink(missing_ok=True)
        (run_dir / "RUNNING").unlink(missing_ok=True)
        LOGGER.info("Question 2 run completed: %s", run_dir)
        return run_dir
    except BaseException as exc:
        failure = {
            "run_id": run_id,
            "status": "FAILED",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": round(time.time() - started, 2),
        }
        _write_json(run_dir / "failure.json", failure)
        (run_dir / "FAILED").write_text(datetime.now().isoformat(), encoding="utf-8")
        (run_dir / "RUNNING").unlink(missing_ok=True)
        LOGGER.exception("Question 2 run failed")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the Question 2 robust emotion model")
    parser.add_argument("--config", default="configs/q2.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--run-root")
    parser.add_argument(
        "--resume",
        help="resume an interrupted run by run id or run directory",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true", help="short deterministic pipeline smoke run")
    parser.add_argument("--compact-validation", action="store_true")
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--use-distillation", action="store_true", help="enable the optional teacher/student candidate")
    parser.add_argument("--no-distillation", action="store_true")
    parser.add_argument("--skip-attachment3", action="store_true")
    parser.add_argument("--skip-input-hash", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
