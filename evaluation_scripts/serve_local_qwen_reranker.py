#!/usr/bin/env python3
"""Serve a local Qwen model as a constrained ATC candidate reranker."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Optional compatible tokenizer path when a merged model tokenizer is not loadable.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    # 8765 is reserved for the evaluation platform. Keep the optional model
    # service on a separate port so both processes can run at the same time.
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    return parser.parse_args()


class QwenRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.model_path = str(args.model.resolve())
        self.max_input_tokens = int(args.max_input_tokens)
        self.max_new_tokens = int(args.max_new_tokens)
        self.lock = threading.Lock()
        tokenizer_path = str((args.tokenizer or args.model).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        if args.device == "cpu":
            device_map: Any = {"": "cpu"}
        elif args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            device_map = {"": 0}
        else:
            device_map = "auto"
        dtype_lookup = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if args.dtype == "auto":
            model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            model_dtype = dtype_lookup[args.dtype]
        kwargs: dict[str, Any] = {
            "local_files_only": True,
            "device_map": device_map,
            "low_cpu_mem_usage": True,
        }
        if args.load_in_4bit and args.load_in_8bit:
            raise ValueError("--load-in-4bit and --load-in-8bit are mutually exclusive")
        if args.load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif args.load_in_8bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = model_dtype
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **kwargs)
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)
        self.loaded_at = time.time()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model": self.model_path,
            "device": self.device,
            "cuda_available": bool(self.torch.cuda.is_available()),
            "loaded_at_epoch": self.loaded_at,
        }

    def rank(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "只完成受约束候选选择，不输出额外说明。"},
            {"role": "user", "content": prompt},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        model_device = next(self.model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        with self.lock, self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def handler_factory(runtime: QwenRuntime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}", flush=True)

        def send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/health":
                self.send_json(404, {"error": "not_found"})
                return
            self.send_json(200, runtime.health())

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/rank":
                self.send_json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                prompt = str(payload["prompt"])
                text = runtime.rank(prompt)
                self.send_json(200, {"text": text, "model": runtime.model_path})
            except Exception as exc:  # service boundary: return an auditable error envelope
                self.send_json(500, {"error": type(exc).__name__, "detail": str(exc)[:1000]})

    return Handler


def main() -> int:
    args = parse_args()
    runtime = QwenRuntime(args)
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(runtime))
    print(json.dumps(runtime.health(), ensure_ascii=False), flush=True)
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
