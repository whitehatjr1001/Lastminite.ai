"""Run the supervisory agent service end-to-end."""

import asyncio
import logging
from dotenv import load_dotenv

from lastminute_api.application.agent_service.service import (
    configure_agent_logging,
    run_revision_agent,
    summarise_agent_result,
)


async def main() -> None:
    load_dotenv()
    configure_agent_logging(logging.DEBUG)

    query = "Generate a concise explanation of CRISPR and include a diagram prompt."
    state = await run_revision_agent(query)
    summary = summarise_agent_result(state)

    print("Query type:", summary.get("query_type"))
    print("Answer:\n", summary.get("answer"))
    image_url = summary.get("image_url")
    if image_url:
        print("Image URL:", image_url[:120] + "...")


if __name__ == "__main__":
    asyncio.run(main())
