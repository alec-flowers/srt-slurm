# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for self-contained site recipes and reproducibility artifacts."""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from srtctl.cli.submit import migrate_site_config, show_config_details
from srtctl.core.config import load_config
from srtctl.core.lockfile import build_lock_section
from srtctl.core.runtime import RuntimeContext


def _write_yaml(tmp_path: Path, data: dict, name: str = "recipe.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _resources() -> dict:
    return {
        "gpu_type": "h100",
        "gpus_per_node": 8,
        "agg_nodes": 1,
        "agg_workers": 1,
    }


def _site_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "model": tmp_path / "models" / "kimi",
        "speculative_model": tmp_path / "models" / "eagle",
        "traces": tmp_path / "traces",
        "output": tmp_path / "outputs",
    }
    for path in paths.values():
        path.mkdir(parents=True)
    container = tmp_path / "containers" / "runtime.sqsh"
    container.parent.mkdir(parents=True)
    container.touch()
    paths["container"] = container
    return paths


def _minimal_site_recipe(paths: dict[str, Path]) -> dict:
    return {
        "name": "site-job",
        "schema_version": 2,
        "site": {
            "name": "lyris",
            "slurm": {
                "account": "site-account",
                "partition": "site-partition",
                "time_limit": "04:00:00",
                "network_interface": "ib0",
            },
            "output": {"path": str(paths["output"])},
            "model": {"path": str(paths["model"])},
            "container": {"path": str(paths["container"])},
        },
        "resources": _resources(),
    }


class TestSiteConfigLoading:
    def test_minimal_site_recipe_synthesizes_legacy_model_without_srtslurm(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe_path = _write_yaml(tmp_path, _minimal_site_recipe(paths))

        with patch("srtctl.core.config.load_cluster_config", side_effect=AssertionError("srtslurm.yaml used")):
            config = load_config(recipe_path)

        assert config.site is not None
        assert config.schema_version == 2
        assert config.site.name == "lyris"
        assert config.model.path == str(paths["model"])
        assert config.model.container == str(paths["container"])
        assert config.model.precision == "unknown"
        assert config.slurm.account == "site-account"
        assert config.slurm.partition == "site-partition"
        assert config.slurm.time_limit == "04:00:00"

    def test_legacy_recipe_without_schema_version_defaults_to_v1(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe = {
            "name": "legacy-job",
            "model": {
                "path": str(paths["model"]),
                "container": str(paths["container"]),
                "precision": "fp8",
            },
            "resources": _resources(),
        }
        recipe_path = _write_yaml(tmp_path, recipe)

        with patch("srtctl.core.config.load_cluster_config", return_value=None):
            config = load_config(recipe_path)

        assert config.schema_version == 1
        assert config.site is None

    def test_unversioned_site_recipe_defaults_to_v2_during_transition(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe = _minimal_site_recipe(paths)
        recipe.pop("schema_version")

        config = load_config(_write_yaml(tmp_path, recipe))

        assert config.schema_version == 2
        assert config.site is not None

    def test_schema_version_one_rejects_site_recipes(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe = _minimal_site_recipe(paths)
        recipe["schema_version"] = 1

        with pytest.raises(ValueError, match="schema_version: 2 is required"):
            load_config(_write_yaml(tmp_path, recipe))

    def test_schema_version_two_requires_site_block(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe = {
            "name": "legacy-job",
            "schema_version": 2,
            "model": {
                "path": str(paths["model"]),
                "container": str(paths["container"]),
                "precision": "fp8",
            },
            "resources": _resources(),
        }

        with pytest.raises(ValueError, match="schema_version: 2 recipes must use site"):
            load_config(_write_yaml(tmp_path, recipe))

    def test_unsupported_schema_version_is_rejected(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe = _minimal_site_recipe(paths)
        recipe["schema_version"] = 99

        with pytest.raises(ValueError, match="Unsupported schema_version 99"):
            load_config(_write_yaml(tmp_path, recipe))

    def test_site_metadata_is_optional_but_populates_identity_when_declared(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe = _minimal_site_recipe(paths)
        recipe["site"]["model"].update(
            {
                "hf_repo": "nvidia/Kimi-K2.5-NVFP4",
                "revision": "c0285e649c34",
                "precision": "fp4",
            }
        )
        recipe["site"]["container"].update(
            {
                "image": "gitlab.example/dynamo:trtllm",
                "digest": "sha256:abc",
                "frameworks": {"dynamo": "1.0.0", "tensorrt_llm": "1.3.0rc9"},
            }
        )
        recipe_path = _write_yaml(tmp_path, recipe)

        config = load_config(recipe_path)

        assert config.model.precision == "fp4"
        assert config.identity.model.repo == "nvidia/Kimi-K2.5-NVFP4"
        assert config.identity.model.revision == "c0285e649c34"
        assert config.identity.container.image == "gitlab.example/dynamo:trtllm"
        assert config.identity.frameworks == {"dynamo": "1.0.0", "tensorrt_llm": "1.3.0rc9"}
        assert config.site.container.digest == "sha256:abc"

    def test_site_plus_top_level_model_is_rejected(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe = _minimal_site_recipe(paths)
        recipe["model"] = {
            "path": "legacy-alias",
            "container": "legacy-container",
            "precision": "fp8",
        }
        recipe_path = _write_yaml(tmp_path, recipe)

        with pytest.raises(ValueError, match="move model/container fields under site"):
            load_config(recipe_path)


class TestSiteRuntime:
    def test_runtime_uses_site_paths_mounts_and_network_interface(self, tmp_path):
        paths = _site_paths(tmp_path)
        recipe = _minimal_site_recipe(paths)
        recipe["site"]["speculative_model"] = {"path": str(paths["speculative_model"])}
        recipe["site"]["mounts"] = [f"{paths['traces']}:/traces"]
        config = load_config(_write_yaml(tmp_path, recipe))

        slurm_env = {
            "SLURM_JOB_ID": "12345",
            "SLURM_JOBID": "12345",
            "SLURM_NODELIST": "gpu-[01]",
            "SLURM_JOB_NUM_NODES": "1",
        }

        def mock_scontrol(cmd, **kwargs):
            if cmd[:3] == ["scontrol", "show", "hostnames"]:
                result = MagicMock()
                result.stdout = "gpu-01\n"
                result.returncode = 0
                return result
            raise subprocess.CalledProcessError(1, cmd)

        with (
            patch.dict(os.environ, slurm_env),
            patch("subprocess.run", mock_scontrol),
            patch("srtctl.core.runtime.get_hostname_ip", return_value="10.0.0.1"),
        ):
            runtime = RuntimeContext.from_config(config, job_id="12345")

        assert runtime.model_path == paths["model"].resolve()
        assert runtime.container_image == paths["container"].resolve()
        assert runtime.network_interface == "ib0"
        assert runtime.container_mounts[runtime.log_dir.parent] == Path("/outputs")
        assert runtime.container_mounts[runtime.log_dir] == Path("/logs")
        assert runtime.container_mounts[paths["model"].resolve()] == Path("/model")
        assert runtime.container_mounts[paths["speculative_model"].resolve()] == Path("/speculative-model")
        assert runtime.container_mounts[paths["traces"].resolve()] == Path("/traces")

    def test_dry_run_shows_site_mounts_not_legacy_cluster_mounts(self, tmp_path, capsys):
        paths = _site_paths(tmp_path)
        recipe = _minimal_site_recipe(paths)
        recipe["site"]["speculative_model"] = {
            "path": str(paths["speculative_model"]),
            "target": "/draft-model",
        }
        recipe["site"]["mounts"] = [f"{paths['traces']}:/traces"]
        config = load_config(_write_yaml(tmp_path, recipe))

        with patch("srtctl.cli.submit.get_srtslurm_setting", side_effect=AssertionError("srtslurm.yaml used")):
            show_config_details(config)

        output = capsys.readouterr().out
        assert "/model" in output
        assert "/draft-model" in output
        assert "/traces" in output
        assert "site" in output
        assert "srtslurm.yaml" not in output


class TestSiteLockfileArtifacts:
    def test_lockfile_records_declared_and_observed_site_artifacts(self, tmp_path):
        paths = _site_paths(tmp_path)
        (paths["model"] / ".huggingface" / "refs").mkdir(parents=True)
        (paths["model"] / ".huggingface" / "refs" / "main").write_text("observed-model-revision")
        (paths["model"] / "config.json").write_text('{"_name_or_path": "nvidia/Kimi-K2.5-NVFP4"}')
        (paths["model"] / "tokenizer_config.json").write_text('{"tokenizer_class": "TestTokenizer"}')
        with (paths["model"] / "model.safetensors").open("wb") as f:
            f.truncate(64 * 1024 * 1024 + 1)
        (paths["speculative_model"] / ".huggingface").mkdir(parents=True)
        (paths["speculative_model"] / ".huggingface" / "download_metadata.json").write_text(
            '{"repo_id": "nvidia/Kimi-K2.5-Thinking-Eagle3", "commit_hash": "observed-spec-revision"}'
        )
        (paths["traces"] / "trace.jsonl").write_text('{"prompt": "hello"}\n')

        recipe = _minimal_site_recipe(paths)
        recipe["site"]["model"].update(
            {
                "hf_repo": "nvidia/Kimi-K2.5-NVFP4",
                "revision": "declared-model-revision",
                "precision": "fp4",
            }
        )
        recipe["site"]["speculative_model"] = {
            "path": str(paths["speculative_model"]),
            "hf_repo": "nvidia/Kimi-K2.5-Thinking-Eagle3",
            "revision": "declared-spec-revision",
        }
        recipe["site"]["container"].update(
            {
                "image": "gitlab.example/dynamo:trtllm",
                "digest": "sha256:abc",
                "frameworks": {"dynamo": "1.0.0"},
            }
        )
        recipe["site"]["mounts"] = [f"{paths['traces']}:/traces"]
        recipe["setup_script"] = "custom-setup.sh"
        source_dir = tmp_path / "source"
        (source_dir / "configs").mkdir(parents=True)
        (source_dir / "configs" / "custom-setup.sh").write_text("#!/bin/bash\necho setup\n")
        config = load_config(_write_yaml(tmp_path, recipe))
        resolved_log_dir = paths["output"] / "12345" / "logs"
        resolved_log_dir.mkdir(parents=True)
        (resolved_log_dir / "trtllm_config_prefill.yaml").write_text("kv_cache_config: {}\n")
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(recipe, sort_keys=False))
        (tmp_path / "config_resolved.yaml").write_text("name: resolved\n")
        (tmp_path / "sbatch_script.sh").write_text("#!/bin/bash\nsrtctl\n")

        with patch.dict(os.environ, {"SRTCTL_SOURCE_DIR": str(source_dir)}):
            lock = build_lock_section(
                config,
                worker_fingerprints={"agg_w0": {"frameworks": {"dynamo": "1.0.1", "tensorrt_llm": "1.3.0rc9"}}},
                resolved_log_dir=resolved_log_dir,
                output_dir=tmp_path,
                recipe_text=(tmp_path / "config.yaml").read_text().rstrip(),
            )

        assert lock["hashes"]["recipe_exact"]["status"] == "complete"
        assert lock["hashes"]["normalized_intent"]["status"] == "complete"
        assert lock["hashes"]["site_binding"]["status"] == "complete"
        assert lock["hashes"]["normalized_intent"]["value"] != lock["hashes"]["site_binding"]["value"]
        artifacts = lock["artifacts"]
        assert artifacts["model"]["declared"]["hf_repo"] == "nvidia/Kimi-K2.5-NVFP4"
        assert artifacts["model"]["declared"]["revision"] == "declared-model-revision"
        assert artifacts["model"]["observed"]["revision"] == "observed-model-revision"
        assert artifacts["model"]["observed"]["model_id"] == "nvidia/Kimi-K2.5-NVFP4"
        assert artifacts["model"]["manifest"]["hash"]["status"] == "partial"
        manifest_entries = {entry["path"]: entry for entry in artifacts["model"]["manifest"]["entries"]}
        assert manifest_entries["config.json"]["hash"]["status"] == "complete"
        assert manifest_entries["tokenizer_config.json"]["hash"]["status"] == "complete"
        assert manifest_entries["model.safetensors"]["hash"]["status"] == "skipped_large_file"
        assert artifacts["speculative_model"]["declared"]["target"] == "/speculative-model"
        assert artifacts["speculative_model"]["observed"]["hf_repo"] == "nvidia/Kimi-K2.5-Thinking-Eagle3"
        assert artifacts["speculative_model"]["observed"]["revision"] == "observed-spec-revision"
        assert artifacts["container"]["declared"]["digest"] == "sha256:abc"
        assert artifacts["container"]["observed"]["kind"] == "file"
        assert "hash" not in artifacts["container"]["observed"]
        assert artifacts["frameworks"]["declared"] == {"dynamo": "1.0.0"}
        assert artifacts["frameworks"]["observed"] == {"dynamo": "1.0.1", "tensorrt_llm": "1.3.0rc9"}
        assert artifacts["mounts"][0]["target"] == "/traces"
        assert artifacts["mounts"][0]["observed"]["exists"] is True
        assert artifacts["mounts"][0]["manifest"]["hash"]["status"] == "complete"
        assert artifacts["mounts"][0]["manifest"]["entries"][0]["path"] == "trace.jsonl"
        assert artifacts["output"]["declared"]["base_path"] == str(paths["output"])
        assert artifacts["output"]["observed"]["path"] == str(resolved_log_dir)
        runtime_code = artifacts["runtime_code"]
        assert runtime_code["files"]["submitted_config"]["hash"]["status"] == "complete"
        assert runtime_code["files"]["sbatch_script"]["hash"]["status"] == "complete"
        assert runtime_code["files"]["runtime_configs"][0]["relative_path"] == "config_resolved.yaml"
        assert runtime_code["files"]["generated_runtime_configs"][0]["relative_path"] == "trtllm_config_prefill.yaml"
        assert runtime_code["files"]["generated_runtime_configs"][0]["hash"]["status"] == "complete"
        assert runtime_code["files"]["setup_script"]["hash"]["status"] == "complete"

    def test_normalized_intent_ignores_site_paths_but_site_binding_tracks_them(self, tmp_path):
        paths_a = _site_paths(tmp_path / "cluster-a")
        paths_b = _site_paths(tmp_path / "cluster-b")
        recipe_a = _minimal_site_recipe(paths_a)
        recipe_b = _minimal_site_recipe(paths_b)
        for recipe in (recipe_a, recipe_b):
            recipe["site"]["model"].update({"hf_repo": "nvidia/Kimi-K2.5-NVFP4", "revision": "abc123"})
            recipe["site"]["container"].update({"image": "registry.example/dynamo:tag"})
            recipe["site"]["mounts"] = [f"{recipe['site']['output']['path']}:/job-output"]

        lock_a = build_lock_section(load_config(_write_yaml(tmp_path, recipe_a, name="a.yaml")))
        lock_b = build_lock_section(load_config(_write_yaml(tmp_path, recipe_b, name="b.yaml")))

        assert lock_a["hashes"]["normalized_intent"]["value"] == lock_b["hashes"]["normalized_intent"]["value"]
        assert lock_a["hashes"]["site_binding"]["value"] != lock_b["hashes"]["site_binding"]["value"]


class TestMigrateSiteCommand:
    def test_migrates_legacy_alias_recipe_to_self_contained_site_recipe(self, tmp_path):
        recipe_path = _write_yaml(
            tmp_path,
            {
                "name": "legacy-job",
                "model": {"path": "kimi", "container": "trtllm-runtime", "precision": "fp4"},
                "identity": {
                    "model": {"repo": "nvidia/Kimi-K2.5-NVFP4", "revision": "abc123"},
                    "container": {"image": "gitlab.example/dynamo:trtllm"},
                    "frameworks": {"dynamo": "1.0.0"},
                },
                "slurm": {"account": "recipe-account"},
                "extra_mount": ["/recipe/cache:/cache"],
                "resources": _resources(),
            },
        )
        srtslurm_path = _write_yaml(
            tmp_path,
            {
                "default_account": "cluster-account",
                "default_partition": "batch",
                "default_time_limit": "04:00:00",
                "network_interface": "eth0",
                "output_dir": "/cluster/outputs",
                "model_paths": {"kimi": "/cluster/models/kimi"},
                "containers": {"trtllm-runtime": "/cluster/containers/trtllm.sqsh"},
                "default_mounts": {"/cluster/traces": "/traces"},
            },
            name="srtslurm.yaml",
        )

        migrated = migrate_site_config(recipe_path, srtslurm_path, "lyris")

        assert "model" not in migrated
        assert "identity" not in migrated
        assert "extra_mount" not in migrated
        assert migrated["schema_version"] == 2
        assert migrated["site"]["name"] == "lyris"
        assert migrated["site"]["slurm"]["account"] == "recipe-account"
        assert migrated["site"]["slurm"]["partition"] == "batch"
        assert migrated["site"]["model"]["path"] == "/cluster/models/kimi"
        assert migrated["site"]["model"]["hf_repo"] == "nvidia/Kimi-K2.5-NVFP4"
        assert migrated["site"]["model"]["revision"] == "abc123"
        assert migrated["site"]["container"]["path"] == "/cluster/containers/trtllm.sqsh"
        assert migrated["site"]["container"]["image"] == "gitlab.example/dynamo:trtllm"
        assert migrated["site"]["container"]["frameworks"] == {"dynamo": "1.0.0"}
        assert migrated["site"]["output"]["path"] == "/cluster/outputs"
        assert migrated["site"]["mounts"] == ["/cluster/traces:/traces", "/recipe/cache:/cache"]
