import asyncio
import logging

from telegram.bot import run_bot


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    await run_bot()


if __name__ == "__main__":
    asyncio.run(main())
