import httpx, yaml
from typing import Dict, Any

CONFIG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))

async def call_llm(tier: str, prompt: Dict[str, str]) -> str:
    model = CONFIG["llm"]["tier1_model"] if tier=="tier1" else CONFIG["llm"]["tier2_model"]
    max_tokens = CONFIG["llm"]["max_output_tokens_tier1"] if tier=="tier1" else CONFIG["llm"]["max_output_tokens_tier2"]
    return await _ollama_chat(model, prompt, max_tokens)

async def _ollama_chat(model: str, prompt: Dict[str, str], max_tokens: int) -> str:
    url = "http://localhost:11434/api/chat"
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(url, json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2}
        })
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "") or str(data)
