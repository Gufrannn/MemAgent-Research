import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ENTRY = REPO / "scripts/h20/preflight_qwen25_7b_mic_v2_e0.sh"
DATA_ENTRY = REPO / "scripts/h20/preflight_qwen25_7b_mic_v2_data_freeze.sh"


class MicV2RealEntryAdversarialTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="mic-v2-entry-")
        cls.root = Path(cls.temp.name)
        cls.clone = cls.root / "repo"
        subprocess.run(
            ["git", "clone", "--shared", "--no-checkout", str(REPO), str(cls.clone)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        cls.commit = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True,
        ).strip()
        subprocess.run(
            ["git", "-C", str(cls.clone), "checkout", "--detach", cls.commit],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        python_link = cls.root / ".venv/bin/python"
        python_link.parent.mkdir(parents=True)
        python_link.symlink_to(Path(sys.executable))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def environment(self, *, run_id="entry-test", commit=None):
        result = dict(os.environ)
        result.update({
            "MEMAGENT_MIC_V2_WORK_ROOT": str(self.root),
            "MEMAGENT_MIC_V2_REPO_DIR": str(self.clone),
            "MEMAGENT_MIC_V2_EXPECTED_COMMIT": commit or self.commit,
            "MEMAGENT_MIC_V2_RUN_ID": run_id,
        })
        result.pop("PYTHONOPTIMIZE", None)
        return result

    def invoke(self, environment):
        return subprocess.run(
            ["bash", str(ENTRY)], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    def data_environment(self, *, run_id="data-entry-test", commit=None):
        result = self.environment(run_id="unused-e0", commit=commit)
        result["MEMAGENT_MIC_V2_DATA_RUN_ID"] = run_id
        return result

    def invoke_data(self, environment):
        return subprocess.run(
            ["bash", str(DATA_ENTRY)], env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    def test_missing_environment_is_rejected_by_real_shell_entry(self):
        environment = {"PATH": os.environ.get("PATH", "")}
        result = self.invoke(environment)
        self.assertEqual(result.returncode, 40, result.stdout)
        self.assertIn("missing required environment", result.stdout)

    def test_wrong_commit_is_rejected_by_real_shell_entry(self):
        result = self.invoke(self.environment(commit="0" * 40))
        self.assertEqual(result.returncode, 42, result.stdout)
        self.assertIn("exact Git commit mismatch", result.stdout)

    def test_unsafe_run_id_is_rejected_by_real_shell_entry(self):
        result = self.invoke(self.environment(run_id="../escape"))
        self.assertEqual(result.returncode, 45, result.stdout)
        self.assertIn("safe stable identifier", result.stdout)

    def test_optimized_python_is_rejected_by_real_shell_entry(self):
        environment = self.environment(run_id="optimized")
        environment["PYTHONOPTIMIZE"] = "1"
        result = self.invoke(environment)
        self.assertEqual(result.returncode, 46, result.stdout)
        self.assertIn("PYTHONOPTIMIZE is forbidden", result.stdout)

    def test_dirty_tree_is_rejected_by_real_shell_entry(self):
        marker = self.clone / "UNTRACKED_MIC_V2_ATTACK"
        marker.write_text("attack\n")
        try:
            result = self.invoke(self.environment(run_id="dirty"))
        finally:
            marker.unlink()
        self.assertEqual(result.returncode, 43, result.stdout)
        self.assertIn("worktree is dirty", result.stdout)

    def test_existing_attempt_is_rejected_without_overwrite(self):
        attempt = self.root / "logs/mic_v2/replay"
        attempt.mkdir(parents=True)
        sentinel = attempt / "sentinel"
        sentinel.write_text("preserve\n")
        result = self.invoke(self.environment(run_id="replay"))
        self.assertEqual(result.returncode, 44, result.stdout)
        self.assertEqual(sentinel.read_text(), "preserve\n")

    def test_data_entry_missing_environment_is_rejected(self):
        result = self.invoke_data({"PATH": os.environ.get("PATH", "")})
        self.assertEqual(result.returncode, 40, result.stdout)
        self.assertIn("missing required environment", result.stdout)

    def test_data_entry_wrong_commit_is_rejected(self):
        result = self.invoke_data(self.data_environment(commit="0" * 40))
        self.assertEqual(result.returncode, 42, result.stdout)
        self.assertIn("exact Git commit mismatch", result.stdout)

    def test_data_entry_unsafe_run_id_is_rejected(self):
        result = self.invoke_data(self.data_environment(run_id="../escape"))
        self.assertEqual(result.returncode, 45, result.stdout)
        self.assertIn("safe stable identifier", result.stdout)

    def test_data_entry_optimized_python_is_rejected(self):
        environment = self.data_environment(run_id="data-optimized")
        environment["PYTHONOPTIMIZE"] = "1"
        result = self.invoke_data(environment)
        self.assertEqual(result.returncode, 46, result.stdout)
        self.assertIn("PYTHONOPTIMIZE is forbidden", result.stdout)

    def test_data_entry_dirty_tree_is_rejected(self):
        marker = self.clone / "UNTRACKED_MIC_V2_DATA_ATTACK"
        marker.write_text("attack\n")
        try:
            result = self.invoke_data(self.data_environment(run_id="data-dirty"))
        finally:
            marker.unlink()
        self.assertEqual(result.returncode, 43, result.stdout)
        self.assertIn("worktree is dirty", result.stdout)

    def test_data_entry_existing_attempt_is_preserved(self):
        attempt = self.root / "logs/mic_v2_data_freeze/data-replay"
        attempt.mkdir(parents=True)
        sentinel = attempt / "sentinel"
        sentinel.write_text("preserve\n")
        result = self.invoke_data(self.data_environment(run_id="data-replay"))
        self.assertEqual(result.returncode, 44, result.stdout)
        self.assertEqual(sentinel.read_text(), "preserve\n")


if __name__ == "__main__":
    unittest.main()
