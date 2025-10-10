import asyncio

from telegram.bot import run_bot


async def main() -> None:
    await run_bot()


if __name__ == "__main__":
    asyncio.run(main())
