import asyncio
import os
from dotenv import load_dotenv

from llm.llama_client import LlamaClient
from utils.prompts import SYSTEM_PROMPT


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        raise RuntimeError("SAMBANOVA_API_KEY is not set. Populate .env and try again.")

    client = LlamaClient()

    print("=== Non-stream test ===")
    try:
        text = await client.acomplete(
            "Скажи 'привет' и назови модель, которую ты используешь.",
            system=SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=64,
        )
        print(text)
    except Exception as e:
        print("Non-stream call failed:", e)

    print("\n=== Stream test ===")
    try:
        async for chunk in client.astream(
            "Коротко дополни фразу: 'Вкусный чай — ' (сделай большой рассказ)",
            system=SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=1200,
        ):
            print(chunk, end="", flush=True)
        print()
    except Exception as e:
        print("Stream call failed:", e)

    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
