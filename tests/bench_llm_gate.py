"""Benchmark: chat latency vs background generation contention.

Requires Ollama running locally.
Run with: uv run python tests/bench_llm_gate.py
"""
from __future__ import annotations

import asyncio
import json
import time

import httpx

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:8b"

# Short prompts to keep each call fast (~2-5s instead of ~10s)
SHORT_CHAT = "Define 'hello' in 5 words."
SHORT_BG = "Define 'test' in 5 words."
LONG_BG = (
    "Generate a quiz question about 'ameliorate' as JSON with fields: "
    "stem, choices (4), correct_index, explanation."
)


async def _generate(prompt: str, system: str | None = None) -> dict:
    """Single Ollama /api/generate call with timing."""
    body: dict = {"model": MODEL, "prompt": prompt, "temperature": 0.3, "stream": True}
    if system:
        body["system"] = system
    t0 = time.monotonic()
    first = None
    tokens = 0
    async with httpx.AsyncClient(timeout=120.0) as c:
        async with c.stream("POST", f"{OLLAMA_URL}/api/generate", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                tok = json.loads(line).get("response", "")
                if tok:
                    if first is None:
                        first = time.monotonic() - t0
                    tokens += 1
    total = time.monotonic() - t0
    return {"first": first or total, "total": total, "tokens": tokens}


async def scenario_baseline():
    """Chat with no contention."""
    r = await _generate(SHORT_CHAT, system="Be brief.")
    print(f"  First token: {r['first']:.2f}s  Total: {r['total']:.2f}s")
    return r["first"]


async def scenario_contention():
    """Chat while a long background request is in-flight."""
    async def bg():
        return await _generate(LONG_BG)

    async def chat():
        await asyncio.sleep(0.3)
        return await _generate(SHORT_CHAT, system="Be brief.")

    bg_t, chat_r = await asyncio.gather(asyncio.create_task(bg()), asyncio.create_task(chat()))
    print(f"  BG total: {bg_t['total']:.2f}s")
    print(f"  Chat first token: {chat_r['first']:.2f}s  Total: {chat_r['total']:.2f}s")
    return chat_r["first"]


async def scenario_cancel():
    """Cancel background, then chat."""
    async def bg():
        try:
            return await _generate(LONG_BG)
        except (asyncio.CancelledError, httpx.ReadError):
            return None

    task = asyncio.create_task(bg())
    await asyncio.sleep(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    r = await _generate(SHORT_CHAT, system="Be brief.")
    print(f"  Chat first token: {r['first']:.2f}s  Total: {r['total']:.2f}s")
    return r["first"]


async def main():
    print("Warming up (loading model)...")
    await _generate("Hi")
    print()

    print("1. Baseline (no contention):")
    baseline = await scenario_baseline()

    print("\n2. Contention (chat during background):")
    contention = await scenario_contention()

    print("\n3. Cancel background, then chat:")
    cancelled = await scenario_cancel()

    print(f"\nSummary:")
    print(f"  Baseline:    {baseline:.2f}s")
    print(f"  Contention:  {contention:.2f}s  ({contention/baseline:.1f}x baseline)")
    print(f"  Cancel:      {cancelled:.2f}s  ({cancelled/baseline:.1f}x baseline)")

    if contention > baseline * 1.5:
        print("\n  Contention adds significant latency — Ollama serializes GPU work.")
    if cancelled < contention * 0.7:
        print("  Cancellation effectively frees the GPU for chat.")


if __name__ == "__main__":
    asyncio.run(main())
