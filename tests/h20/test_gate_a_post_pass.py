import re
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ACCEPT = REPO / "scripts/h20/accept_qwen25_7b_gatea_r5_pass.sh"
ARCHIVE = REPO / "scripts/h20/archive_qwen25_7b_gatea_r5_evidence.sh"
PHASE1 = REPO / "docs/h20/phase1_stable_identity_freeze_draft_20260821.md"


def embedded_python(script: str) -> str:
    match = re.search(r"<<'PY'\n(?P<body>.*?)\nPY(?:\n|$)", script, re.DOTALL)
    if match is None:
        raise AssertionError("script is missing its embedded Python validator")
    return match.group("body")


class GateAPostPassContractTests(unittest.TestCase):
    def test_acceptance_is_read_only_and_fail_closed(self):
        script = ACCEPT.read_text(encoding="utf-8")
        compile(embedded_python(script), str(ACCEPT), "exec")
        self.assertIn("--phase final", script)
        self.assertNotIn("--write-report", script)
        self.assertNotIn("run_gate_a.sh", script)
        self.assertNotIn("resume_qwen25_7b_gatea_step2_to3.sh", script)
        self.assertIn("c3f987be5513cad2a9e95622dd6773726a7bf12e", script)
        self.assertIn("GATE_A_POST_PASS_OK", script)
        self.assertIn("version_3_master_digest", script)
        self.assertIn("vllm_worker_ranks_acknowledged", script)

    def test_archive_is_atomic_whitelisted_and_does_not_copy_checkpoints(self):
        script = ARCHIVE.read_text(encoding="utf-8")
        compile(embedded_python(script), str(ARCHIVE), "exec")
        self.assertIn("mktemp -d", script)
        self.assertIn("mv -T --no-clobber", script)
        self.assertIn("git clone --quiet --shared --no-checkout", script)
        self.assertIn("bundle verify", script)
        self.assertIn("bundle list-heads", script)
        self.assertIn("gzip -t", script)
        self.assertIn("sha256sum -c", script)
        self.assertNotIn("cp -a", script)
        self.assertNotIn("*.log", script)
        self.assertNotIn("/tmp/gate_a_r5_readonly_reaudit.json", script)
        self.assertNotIn("--write-report", script)
        self.assertNotRegex(script, r"cp[^\n]*(?:model|optim|extra_state)_world_size")

    def test_phase1_draft_does_not_authorize_gpu_execution(self):
        draft = PHASE1.read_text(encoding="utf-8")
        self.assertIn("gpu_execution_authorized=false", draft)
        self.assertIn("strict vLLM", draft)
        self.assertIn("不构造新数据集", draft)
        self.assertIn("当前不启动", draft)
        self.assertNotIn("gpu_execution_authorized=true", draft)


if __name__ == "__main__":
    unittest.main()
