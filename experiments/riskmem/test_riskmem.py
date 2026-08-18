import importlib.util
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("riskmem_headroom_vllm.py")
SPEC = importlib.util.spec_from_file_location("riskmem_headroom", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RiskMemTests(unittest.TestCase):
    def test_version_link_and_retrieval(self):
        store = MODULE.VersionStore()
        store.add_session("user", 0, [{
            "index": 1, "memory_content": "User lives in Paris", "timestamp": "2025",
            "memory_type": "Persona", "importance": 0.8, "is_update": "False",
            "original_memories": [], "memory_source": "primary",
        }])
        store.add_session("user", 1, [{
            "index": 1, "memory_content": "User lives in Rome", "timestamp": "2026",
            "memory_type": "Persona", "importance": 0.8, "is_update": "True",
            "original_memories": ["User lives in Paris"], "memory_source": "primary",
        }])
        self.assertEqual(store.memories[0].superseded_by, [store.memories[1].uid])
        current = store.retrieve("Where does the user live?", 5, include_superseded=False)
        self.assertEqual([x.content for x in current], ["User lives in Rome"])

    def test_labels_do_not_leak_to_public_memory(self):
        memory = MODULE.Memory("m", "fact", "now", "event", 0.5, "interference", [])
        self.assertNotIn("source", memory.public())
        self.assertNotIn("interference", str(memory.public()))

    def test_paraphrased_version_link(self):
        store = MODULE.VersionStore()
        store.add_session("user", 0, [{
            "memory_content": "Martin wants a new job for better health and career goals.",
            "is_update": False, "original_memories": [],
        }])
        store.add_session("user", 1, [{
            "memory_content": "Martin seeks work aligned with well-being and humanitarian goals.",
            "is_update": True,
            "original_memories": ["Martin wants a new job for better health and professional goals."],
        }])
        self.assertEqual(store.memories[0].superseded_by, [store.memories[1].uid])

    def test_scoring_and_gate_json(self):
        self.assertGreater(MODULE.token_f1("The answer is Rome", "Rome"), 0)
        self.assertEqual(MODULE.parse_selected_ids('text {"selected_ids":["a","bad"]}', {"a"}), ["a"])

    def test_openai_compatible_vllm_client(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                request = json.loads(self.rfile.read(length))
                self.assert_request(request)
                body = json.dumps({"choices": [{"message": {"content": "pong"}}]}).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def assert_request(self, request):
                assert request["model"] == "fake-model"
                assert request["messages"][0]["content"] == "ping"

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        try:
            client = MODULE.VLLMClient(f"http://127.0.0.1:{server.server_port}", "fake-model", 3, 0)
            result = client._sync_chat([{"role": "user", "content": "ping"}], 8, 0.0)
            self.assertEqual(result, "pong")
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__":
    unittest.main()
