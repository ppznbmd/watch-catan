"""A local OpenAI-compatible server, so the provider path is actually exercised.

Everything from the request body to the parsed response goes through the real
`openai` SDK over a real socket. What it does not prove is that DeepSeek behaves
like its documentation — only that we handle it correctly when it does, and when
it misbehaves in the ways the docs warn about.
"""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer


def completion(content, finish_reason="stop", prompt_tokens=894, completion_tokens=120,
               reasoning_content=None):
    message = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        # DeepSeek's thinking trace, beside the answer rather than inside it
        message["reasoning_content"] = reasoning_content
    return {
        "id": "fake-1",
        "object": "chat.completion",
        "created": 0,
        "model": "fake",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def decision_json(choice=0, reasoning="scripted", say="", give=(), want=()):
    # `note` is the key the model is asked for (prompt v4); it lands in
    # Decision.reasoning.
    return json.dumps({
        "note": reasoning,
        "choice": choice,
        "offer_give": list(give),
        "offer_want": list(want),
        "say": say,
    })


@contextmanager
def fake_provider(responder):
    """`responder(request_body, call_index)` returns a dict payload, or an
    (status, dict) pair to answer with an HTTP error."""
    state = {"calls": 0, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            state["requests"].append(body)
            result = responder(body, state["calls"])
            state["calls"] += 1
            status, payload = result if isinstance(result, tuple) else (200, result)
            blob = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
