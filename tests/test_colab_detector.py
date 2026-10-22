"""CPU-only orchestration checks; no downloads, installs or training."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import colab_detector as c
import yolox_train_entry as training_entry


class AutomationTests(unittest.TestCase):
    def test_frozen_repository_inputs(self):
        c.frozen_project()

    def test_launch_arguments(self):
        cmd = c.training_command(Path("initial.pth"))
        self.assertEqual(cmd[cmd.index("-b") + 1], "8")
        self.assertEqual(cmd[cmd.index("-d") + 1], "1")
        self.assertIn("--fp16", cmd)
        self.assertNotIn("--resume", cmd)
        self.assertEqual(cmd[-2:], ["output_dir", str(c.DRIVE / "outputs/detector")])
        self.assertEqual(cmd[cmd.index("-f") + 1], str(c.PROJECT / "exps" / (c.EXP + ".py")))

    def test_verify_only_never_prepares_assets(self):
        with contextlib.ExitStack() as stack:
            checks = [stack.enter_context(patch.object(c, name)) for name in
                      ("frozen_project", "verify_init", "verify_source", "verify_data")]
            readiness = stack.enter_context(patch.object(c, "runtime_check"))
            install = stack.enter_context(patch.object(c, "install_dependencies"))
            run = stack.enter_context(patch.object(c, "run"))
            c.setup(True)
            for check in checks:
                check.assert_called_once()
            install.assert_not_called()
            readiness.assert_not_called()
            run.assert_not_called()

    def test_frozen_patch_git_bytes(self):
        for spec, path in zip(c.compat.PATCH_SPECS, c.patches()):
            name = path.relative_to(c.PROJECT).as_posix()
            original = c.subprocess.check_output(["git", "-C", str(c.PROJECT), "show", f"{c.BASE}:{name}"])
            self.assertEqual(hashlib.sha256(original).hexdigest(), spec["sha256"])
            # Diagnose checkout conversion without relaxing production hash checks.
            self.assertEqual(path.read_bytes().replace(b"\r\n", b"\n"), original)

    def test_readiness_only_probes_imports(self):
        with patch.object(c, "run") as run:
            c.runtime_check()
        code = run.call_args.args[0][2]
        self.assertIn("from yolox.layers import COCOeval_opt", code)
        self.assertIn("from yolox import _C", code)
        self.assertNotIn("get_model(", code)
        self.assertNotIn(".train(", code)
        self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test_training_exception_is_not_suppressed(self):
        args = SimpleNamespace(exp_file="frozen.py", name=None, opts=["output_dir", "drive"],
                               experiment_name=None, devices=1, dist_url=None,
                               num_machines=1, machine_rank=0, dist_backend="nccl")
        from unittest.mock import Mock
        exp = SimpleNamespace(exp_name=c.EXP, merge=Mock())
        def underlying(exp, args):
            raise RuntimeError("simulated training failure")
        def suppressed(*args):
            self.fail("Exception-suppressing decorator was called")
        suppressed.__wrapped__ = underlying
        def launch(main, *unused, **kwargs):
            self.assertEqual(unused, (1, 1, 0))
            self.assertEqual(kwargs["backend"], "nccl")
            self.assertEqual(kwargs["dist_url"], "auto")
            main(*kwargs["args"])
        entry = {"make_parser": lambda: SimpleNamespace(parse_args=lambda: args),
                 "get_exp": lambda *args: exp, "main": suppressed, "launch": launch,
                 "torch": SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1))}
        with self.assertRaisesRegex(RuntimeError, "simulated training failure"):
            training_entry.launch_entry(entry)
        exp.merge.assert_called_once_with(args.opts)
        self.assertEqual(args.experiment_name, c.EXP)

    def test_status_separates_assets_from_readiness(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(c, "DRIVE", Path(temp)))
            for name in ("frozen_project", "verify_source", "verify_data", "verify_init", "run"):
                stack.enter_context(patch.object(c, name))
            stack.enter_context(patch.object(c.subprocess, "run", return_value=c.subprocess.CompletedProcess([], 1, "", "missing extension")))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(c.status(), 1)
            self.assertIn("Asset verification: PASS", output.getvalue())
            self.assertIn("Training readiness (imports, compiled evaluator, CUDA): FAIL", output.getvalue())

    def test_nonempty_training_output_refused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(c, "DRIVE", Path(temp)):
            output = c.DRIVE / "outputs/detector" / c.EXP
            output.mkdir(parents=True)
            (output / "latest_ckpt.pth").write_bytes(b"existing")
            with patch.object(c, "setup"), patch.object(c, "runtime_check"), patch.object(c, "run") as run:
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    c.train()
                run.assert_not_called()

    def test_dataset_hash_and_missing_image_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "annotations").mkdir()
            (root / "train2017").mkdir()
            annotation = root / "annotations/instances_train2017.json"
            annotation.write_text(json.dumps({"images": [{"file_name": "one.jpg"}], "annotations": []}))
            specs = {"train2017": (1, 0, c.compat.sha256(annotation))}
            with patch.object(c, "DATA_SPECS", specs):
                with self.assertRaisesRegex(RuntimeError, "Missing/empty image"):
                    c.verify_data(root)
                (root / "train2017/one.jpg").write_bytes(b"fixture")
                c.verify_data(root)
                annotation.write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    c.verify_data(root)

    def test_setup_reuses_prepared_dataset(self):
        with contextlib.ExitStack() as stack:
            for name in ("frozen_project", "verify_init", "verify_source", "verify_data",
                         "runtime_check", "install_dependencies"):
                stack.enter_context(patch.object(c, name))
            stack.enter_context(patch.object(Path, "exists", return_value=True))
            run = stack.enter_context(patch.object(c, "run"))
            c.setup()
            self.assertEqual(run.call_count, 1)
            self.assertIn("apply_yolox_training_compat.py", str(run.call_args.args[0][1]))

    def test_existing_invalid_dataset_is_not_reconstructed(self):
        with contextlib.ExitStack() as stack:
            for name in ("frozen_project", "verify_init", "verify_source",
                         "runtime_check", "install_dependencies"):
                stack.enter_context(patch.object(c, name))
            stack.enter_context(patch.object(c, "verify_data", side_effect=RuntimeError("invalid dataset")))
            stack.enter_context(patch.object(Path, "exists", return_value=True))
            run = stack.enter_context(patch.object(c, "run"))
            with self.assertRaisesRegex(RuntimeError, "invalid dataset"):
                c.setup()
            self.assertEqual(run.call_count, 1)

    def test_local_mutation_guard(self):
        with patch.object(c.sys, "platform", "win32"):
            with self.assertRaisesRegex(RuntimeError, "Colab"):
                c.require_colab()


if __name__ == "__main__":
    unittest.main()
