import json
import os
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL")
model = os.getenv("DEEPSEEK_MODEL")

if not api_key:
    raise RuntimeError(".env 中缺少 DEEPSEEK_API_KEY")
        
payload = json.dumps(
    {
        "model": model,
        "messages": [{"role": "user", "content": "你好，请简单回复：API 调用成功。"}],
    }
).encode("utf-8")

request = Request(
    f"{base_url}/chat/completions",
    data=payload,
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    method="POST",
)

with urlopen(request, timeout=60) as response:
    result = json.load(response)

print(result["choices"][0]["message"]["content"])
