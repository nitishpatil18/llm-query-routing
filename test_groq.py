import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

for model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is 17 * 24? Answer with just the number."}],
    )
    msg = resp.choices[0].message.content
    usage = resp.usage
    print(f"{model}: answer={msg!r}, prompt_tokens={usage.prompt_tokens}, completion_tokens={usage.completion_tokens}")
