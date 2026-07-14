#!/usr/bin/env python3
"""Experiment 12R one-touch recovery orchestrator.

The failed confirmation runner remains byte-identical evidence. This runner
loads that source, applies only the three Section 15.10 transformation classes,
persists the exact transformed runtime source and a reversible diff attestation,
then executes it once under distinct recovery markers.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = REPO_ROOT / "models" / "trained" / "experiment_12_w1_cross_market_20260714_104047"
DEVELOPMENT_SCRIPT = REPO_ROOT / "scripts" / "experiment_12_w1_cross_market.py"
FAILED_RUNNER = REPO_ROOT / "scripts" / "experiment_12_w1_cross_market_confirmation.py"
DEVELOPMENT_SCRIPT_SHA256 = "48ad28704e49def8b5bbf8d02f46b2f4fa370e258cd719591b48bdbf87f39f0d"
FAILED_RUNNER_SHA256 = "dd569dafd55f25e59c8d847f42a37c4c10c4b0c42553227dccac0786765006e5"
FROZEN_RECIPE_SHA256 = "3fa2aadc12b231f7260676724abf1cd85dca6e4738ec88ed00cc87a73c4ba1fc"
ORIGINAL_TOUCH_SHA256 = "ec1203581caa14f6caab21fd38428b9a67019be926a9a935281e58b09e7607b8"
ORIGINAL_FAILURE_SHA256 = "0ae27c51e61fcbf643d19ed5912eacd460ad64018deddb21cfd319910057d2ad"

ORIGINAL_TOUCH = RUN_ROOT / "confirmation_touch.json"
ORIGINAL_HISTORY = RUN_ROOT / "confirmation_touch_history.jsonl"
ORIGINAL_FAILURE = RUN_ROOT / "confirmation_touch_failed.json"
ORIGINAL_COMPLETION = RUN_ROOT / "confirmation_touch_completed.json"
ORIGINAL_FAILED_DIR = RUN_ROOT / "confirmation_20260714_105649"

RECOVERY_TOUCH = RUN_ROOT / "recovery_touch.json"
RECOVERY_HISTORY = RUN_ROOT / "recovery_touch_history.jsonl"
RECOVERY_COMPLETION = RUN_ROOT / "recovery_touch_completed.json"
RECOVERY_FAILURE = RUN_ROOT / "recovery_touch_failed.json"
SUPERSEDED_PREFLIGHT = RUN_ROOT / "recovery_preflight.json"
SUPERSEDED_DIFF = RUN_ROOT / "recovery_diff_attestation.json"
SUPERSEDED_RUNTIME_SOURCE = RUN_ROOT / "recovery_runtime_source.py"
RECOVERY_PREFLIGHT = RUN_ROOT / "recovery_preflight_v2.json"
RECOVERY_DIFF = RUN_ROOT / "recovery_diff_attestation_v2.json"
RECOVERY_RUNTIME_SOURCE = RUN_ROOT / "recovery_runtime_source_v2.py"

RESIDUAL_OLD = 'residual_frame = pd.read_csv(FROZEN_RUN / recipe["residual_distribution"]["artifact"])'
RESIDUAL_NEW = (
    'residual_frame = pd.read_csv(FROZEN_RUN / recipe["residual_distribution"]["artifact"], '
    'float_precision="round_trip")'
)

PROTECTED_FUNCTIONS = (
    "build_nhl_games", "event_metadata", "map_events", "primary_bootstrap",
    "predicate_scan", "load_inputs", "load_roster", "build_market_goals",
    "build_translation", "price_quotes", "mean_cluster_bootstrap",
    "scoring_secondary", "build_closing_universe", "selected_clv",
    "overall_verdict",
)
PROTECTED_CONSTANTS = (
    "CORE_SNAPSHOTS", "MARKET_FEATURES", "OUTCOMES", "CLOSING",
    "CONFIRMATION_SEASON", "CONFIRMATION_SEASON_CODE", "CONFIRMATION_START",
    "CONFIRMATION_END", "N_BOOTSTRAP", "BOOTSTRAP_SEED", "THRESHOLDS",
    "DFS_BOOKS", "CORE_SOG_COLUMNS", "CORE_SAVES_COLUMNS", "MARKET_COLUMNS",
    "OUTCOME_COLUMNS", "CLOSING_COLUMNS", "TEAM_NAME_TO_ABBREV",
)


class RecoveryPreflightFailure(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_json(path: Path, payload: Any) -> None:
    write_exclusive(
        path, (json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def transform_failed_source(source: str) -> tuple[str, dict[str, Any]]:
    namespace_count = source.count("confirmation_")
    transformed = source.replace("confirmation_", "recovery_")
    preflight_path_count = transformed.count('"recovery_preflight.json"')
    if preflight_path_count != 1:
        raise RecoveryPreflightFailure(f"expected one recovery preflight path, found {preflight_path_count}")
    transformed = transformed.replace('"recovery_preflight.json"', '"recovery_preflight_v2.json"')
    residual_count = transformed.count(RESIDUAL_OLD)
    if residual_count != 1:
        raise RecoveryPreflightFailure(f"expected one residual parser site, found {residual_count}")
    transformed = transformed.replace(RESIDUAL_OLD, RESIDUAL_NEW)
    stage_count = transformed.count('"stage": "confirmation"')
    transformed = transformed.replace('"stage": "confirmation"', '"stage": "recovery"')
    status_count = transformed.count("CONFIRMATION_COMPLETED")
    transformed = transformed.replace("CONFIRMATION_COMPLETED", "RECOVERY_COMPLETED")
    compile(transformed, str(RECOVERY_RUNTIME_SOURCE), "exec")

    reversed_source = transformed.replace(RESIDUAL_NEW, RESIDUAL_OLD)
    reversed_source = reversed_source.replace('"recovery_preflight_v2.json"', '"recovery_preflight.json"')
    reversed_source = reversed_source.replace('"stage": "recovery"', '"stage": "confirmation"')
    reversed_source = reversed_source.replace("RECOVERY_COMPLETED", "CONFIRMATION_COMPLETED")
    reversed_source = reversed_source.replace("recovery_", "confirmation_")
    if reversed_source != source:
        raise RecoveryPreflightFailure("registered transformations are not exactly reversible")
    return transformed, {
        "namespace_replacements_confirmation_to_recovery": namespace_count,
        "versioned_recovery_preflight_path_replacements": preflight_path_count,
        "round_trip_parser_replacements": residual_count,
        "stage_label_replacements": stage_count,
        "completion_status_label_replacements": status_count,
        "reverse_transform_is_byte_identical_to_failed_runner": True,
    }


def ast_nodes(source: str) -> tuple[dict[str, ast.AST], dict[str, ast.AST]]:
    tree = ast.parse(source)
    functions: dict[str, ast.AST] = {}
    constants: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value
    return functions, constants


def ast_hash(node: ast.AST) -> str:
    return sha256_bytes(ast.dump(node, annotate_fields=True, include_attributes=False).encode("utf-8"))


def build_diff_attestation(source: str, transformed: str, replacements: dict[str, Any]) -> dict[str, Any]:
    base_functions, base_constants = ast_nodes(source)
    recovery_functions, recovery_constants = ast_nodes(transformed)
    function_rows = []
    for name in PROTECTED_FUNCTIONS:
        base_hash = ast_hash(base_functions[name])
        recovery_hash = ast_hash(recovery_functions[name])
        if base_hash != recovery_hash:
            raise RecoveryPreflightFailure(f"protected function changed: {name}")
        function_rows.append({"name": name, "failed_ast_sha256": base_hash, "recovery_ast_sha256": recovery_hash})
    constant_rows = []
    for name in PROTECTED_CONSTANTS:
        base_hash = ast_hash(base_constants[name])
        recovery_hash = ast_hash(recovery_constants[name])
        if base_hash != recovery_hash:
            raise RecoveryPreflightFailure(f"protected constant changed: {name}")
        constant_rows.append({"name": name, "failed_ast_sha256": base_hash, "recovery_ast_sha256": recovery_hash})
    if ".fit(" in transformed or "HuberRegressor" in transformed:
        raise RecoveryPreflightFailure("recovery runtime unexpectedly contains a fitting path")
    return {
        "status": "DIFF_ATTESTATION_PASS",
        "failed_runner_path": str(FAILED_RUNNER.relative_to(REPO_ROOT)),
        "failed_runner_sha256": sha256_bytes(source.encode("utf-8")),
        "recovery_runtime_source_path": str(RECOVERY_RUNTIME_SOURCE.relative_to(REPO_ROOT)),
        "recovery_runtime_source_sha256": sha256_bytes(transformed.encode("utf-8")),
        "registered_transformations": replacements,
        "protected_function_ast_hashes": function_rows,
        "protected_constant_ast_hashes": constant_rows,
        "protected_claims": {
            "model_coefficients_and_thresholds_unchanged": True,
            "source_paths_predicates_and_columns_unchanged": True,
            "event_mapping_roster_pairing_and_joins_unchanged": True,
            "grading_stakes_pushes_and_prices_unchanged": True,
            "bootstrap_seed_draws_and_cluster_rules_unchanged": True,
            "secondary_scoring_and_clv_rules_unchanged": True,
            "primary_verdict_semantics_unchanged": True,
            "manifest_algorithm_unchanged_except_recovery_output_filename": True,
            "no_refit_or_reselection_path": True,
        },
    }


def verify_development_manifest() -> dict[str, Any]:
    path = RUN_ROOT / "output_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        artifact = RUN_ROOT / item["path"]
        if artifact.stat().st_size != item["bytes"] or sha256_file(artifact) != item["sha256"]:
            failures.append(item["path"])
    if failures:
        raise RecoveryPreflightFailure(f"development manifest failures: {failures}")
    return {"files_verified": len(manifest["files"]), "manifest_sha256": sha256_file(path)}


def verify_authorization_token(recipe: dict[str, Any]) -> str:
    stored = recipe["one_touch_confirmation_guard"]["authorization_token_sha256"]
    unsigned = copy.deepcopy(recipe)
    unsigned["one_touch_confirmation_guard"].pop("authorization_token_sha256")
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected = sha256_bytes(b"experiment12-confirmation:" + canonical)
    if stored != expected:
        raise RecoveryPreflightFailure("authorization token mismatch")
    return stored


def verify_original_failure_state() -> dict[str, Any]:
    if sha256_file(ORIGINAL_TOUCH) != ORIGINAL_TOUCH_SHA256:
        raise RecoveryPreflightFailure("original touch marker hash mismatch")
    if sha256_file(ORIGINAL_FAILURE) != ORIGINAL_FAILURE_SHA256:
        raise RecoveryPreflightFailure("original failure marker hash mismatch")
    if ORIGINAL_COMPLETION.exists():
        raise RecoveryPreflightFailure("original completion marker unexpectedly exists")
    marker = json.loads(ORIGINAL_TOUCH.read_text(encoding="utf-8"))
    failure = json.loads(ORIGINAL_FAILURE.read_text(encoding="utf-8"))
    history_lines = [json.loads(line) for line in ORIGINAL_HISTORY.read_text(encoding="utf-8").splitlines() if line]
    if marker.get("status") != "IN_PROGRESS" or marker.get("confirmation_runner_sha256") != FAILED_RUNNER_SHA256:
        raise RecoveryPreflightFailure("original touch marker content mismatch")
    if failure.get("status") != "FAILED_TOUCH_CONSUMED" or failure.get("error") != "frozen residual distribution content hash mismatch":
        raise RecoveryPreflightFailure("original failure content mismatch")
    if [row.get("status") for row in history_lines] != ["IN_PROGRESS", "FAILED_TOUCH_CONSUMED"]:
        raise RecoveryPreflightFailure("original history is not exactly IN_PROGRESS then FAILED_TOUCH_CONSUMED")
    failed_dir_files = []
    for path in sorted(ORIGINAL_FAILED_DIR.iterdir()):
        if path.is_file():
            failed_dir_files.append(
                {"path": str(path.relative_to(REPO_ROOT)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return {
        "touch_marker": {"path": str(ORIGINAL_TOUCH.relative_to(REPO_ROOT)), "sha256": ORIGINAL_TOUCH_SHA256, "content": marker},
        "history": {"path": str(ORIGINAL_HISTORY.relative_to(REPO_ROOT)), "sha256": sha256_file(ORIGINAL_HISTORY), "events": history_lines},
        "failure_marker": {"path": str(ORIGINAL_FAILURE.relative_to(REPO_ROOT)), "sha256": ORIGINAL_FAILURE_SHA256, "content": failure},
        "completion_marker_absent": True,
        "failed_output_files": failed_dir_files,
        "original_touch_verdict": "NO VERDICT -- INFRASTRUCTURE FAILURE",
    }


def load_runtime(transformed: str) -> types.ModuleType:
    module = types.ModuleType("experiment_12r_runtime")
    module.__file__ = str(Path(__file__).resolve())
    module.__package__ = None
    sys.modules[module.__name__] = module
    exec(compile(transformed, str(RECOVERY_RUNTIME_SOURCE), "exec"), module.__dict__)
    return module


def source_free_preflight(require_marker_absent: bool = True) -> tuple[dict[str, Any], dict[str, Any], str, Any]:
    if sha256_file(DEVELOPMENT_SCRIPT) != DEVELOPMENT_SCRIPT_SHA256:
        raise RecoveryPreflightFailure("development script hash mismatch")
    if sha256_file(FAILED_RUNNER) != FAILED_RUNNER_SHA256:
        raise RecoveryPreflightFailure("failed confirmation runner hash mismatch")
    if sha256_file(RUN_ROOT / "frozen_recipe.json") != FROZEN_RECIPE_SHA256:
        raise RecoveryPreflightFailure("frozen recipe hash mismatch")
    if require_marker_absent and RECOVERY_TOUCH.exists():
        raise RecoveryPreflightFailure("recovery touch marker already exists")
    if RECOVERY_COMPLETION.exists() or RECOVERY_FAILURE.exists():
        raise RecoveryPreflightFailure("recovery completion/failure marker exists before touch")

    manifest = verify_development_manifest()
    recipe = json.loads((RUN_ROOT / "frozen_recipe.json").read_text(encoding="utf-8"))
    token = verify_authorization_token(recipe)
    original_failure = verify_original_failure_state()
    source = FAILED_RUNNER.read_text(encoding="utf-8")
    if sha256_bytes(source.encode("utf-8")) != FAILED_RUNNER_SHA256:
        raise RecoveryPreflightFailure("failed runner text decode hash mismatch")
    transformed, replacements = transform_failed_source(source)
    attestation = build_diff_attestation(source, transformed, replacements)
    runtime = load_runtime(transformed)
    frozen = runtime.load_frozen_module()

    residual_path = RUN_ROOT / recipe["residual_distribution"]["artifact"]
    manifest_json = json.loads((RUN_ROOT / "output_manifest.json").read_text(encoding="utf-8"))
    residual_manifest = next(item for item in manifest_json["files"] if item["path"] == residual_path.name)
    residual_file_hash = sha256_file(residual_path)
    if residual_file_hash != residual_manifest["sha256"]:
        raise RecoveryPreflightFailure("residual byte hash does not match development manifest")
    residual_frame = pd.read_csv(residual_path, float_precision="round_trip")
    semantic_hash = frozen.canonical_frame_hash(residual_frame)
    if semantic_hash != recipe["residual_distribution"]["sha256"]:
        raise RecoveryPreflightFailure("round-trip residual semantic hash mismatch")
    synthetic = runtime.synthetic_preflight(frozen)

    preflight = {
        "experiment": "12R", "status": "RECOVERY_PREFLIGHT_PASS", "completed_at": utc_now(),
        "source_data_materialized": False, "source_data_paths_opened": [],
        "development_script_sha256": DEVELOPMENT_SCRIPT_SHA256,
        "failed_confirmation_runner_sha256": FAILED_RUNNER_SHA256,
        "recovery_runner_sha256": sha256_file(Path(__file__)),
        "frozen_recipe_sha256": FROZEN_RECIPE_SHA256,
        "authorization_token_verified": token,
        "development_manifest": manifest,
        "original_failed_touch": original_failure,
        "residual_preflight": {
            "path": str(residual_path.relative_to(REPO_ROOT)),
            "bytes": residual_path.stat().st_size,
            "byte_sha256": residual_file_hash,
            "manifest_byte_sha256": residual_manifest["sha256"],
            "round_trip_semantic_sha256": semantic_hash,
            "registered_semantic_sha256": recipe["residual_distribution"]["sha256"],
            "rows": len(residual_frame), "float_precision": "round_trip",
            "byte_and_semantic_checks": "PASS",
        },
        "synthetic_tests": synthetic,
        "diff_attestation_sha256": sha256_bytes(
            (json.dumps(attestation, indent=2, sort_keys=True) + "\n").encode("utf-8")
        ),
        "recovery_runtime_source_sha256": sha256_bytes(transformed.encode("utf-8")),
        "recovery_marker_absent": not RECOVERY_TOUCH.exists(),
        "original_markers_untouched": True,
        "superseded_source_free_preflight": {
            "reason": "The v1 orchestrator expected a non-existent runtime.run_recovery attribute; caught before marker or source access.",
            "preflight_path": str(SUPERSEDED_PREFLIGHT.relative_to(REPO_ROOT)),
            "preflight_sha256": sha256_file(SUPERSEDED_PREFLIGHT),
            "diff_path": str(SUPERSEDED_DIFF.relative_to(REPO_ROOT)),
            "diff_sha256": sha256_file(SUPERSEDED_DIFF),
            "runtime_path": str(SUPERSEDED_RUNTIME_SOURCE.relative_to(REPO_ROOT)),
            "runtime_sha256": sha256_file(SUPERSEDED_RUNTIME_SOURCE),
            "recovery_marker_created": False,
            "source_paths_opened": [],
        },
    }
    runtime.FROZEN = frozen
    return preflight, attestation, transformed, runtime


def persist_preflight(preflight: dict[str, Any], attestation: dict[str, Any], transformed: str) -> None:
    write_exclusive(RECOVERY_RUNTIME_SOURCE, transformed.encode("utf-8"))
    write_exclusive_json(RECOVERY_DIFF, attestation)
    write_exclusive_json(RECOVERY_PREFLIGHT, preflight)


def verify_persisted_preflight(preflight: dict[str, Any], attestation: dict[str, Any], transformed: str) -> None:
    if not (RECOVERY_PREFLIGHT.exists() and RECOVERY_DIFF.exists() and RECOVERY_RUNTIME_SOURCE.exists()):
        raise RecoveryPreflightFailure("persisted recovery preflight artifacts are incomplete")
    persisted = json.loads(RECOVERY_PREFLIGHT.read_text(encoding="utf-8"))
    if persisted.get("status") != "RECOVERY_PREFLIGHT_PASS":
        raise RecoveryPreflightFailure("persisted recovery preflight did not pass")
    if persisted.get("recovery_runner_sha256") != sha256_file(Path(__file__)):
        raise RecoveryPreflightFailure("persisted preflight runner hash mismatch")
    if sha256_file(RECOVERY_RUNTIME_SOURCE) != sha256_bytes(transformed.encode("utf-8")):
        raise RecoveryPreflightFailure("persisted recovery runtime source hash mismatch")
    if sha256_file(RECOVERY_DIFF) != sha256_bytes(
        (json.dumps(attestation, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ):
        raise RecoveryPreflightFailure("persisted diff attestation hash mismatch")
    if preflight["source_data_paths_opened"] or persisted["source_data_paths_opened"]:
        raise RecoveryPreflightFailure("preflight claims a source path was opened")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 12R one-touch recovery runner")
    parser.add_argument("--mode", required=True, choices=("preflight", "recover"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    preflight, attestation, transformed, runtime = source_free_preflight(require_marker_absent=True)
    if args.mode == "preflight":
        persist_preflight(preflight, attestation, transformed)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    verify_persisted_preflight(preflight, attestation, transformed)
    recipe = json.loads((RUN_ROOT / "frozen_recipe.json").read_text(encoding="utf-8"))
    output_dir = runtime.run_confirmation(preflight, runtime.FROZEN, recipe)
    print(f"Experiment 12R completed at {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RecoveryPreflightFailure as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(2)
