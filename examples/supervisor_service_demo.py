"""Run the supervisory agent service end-to-end."""

import asyncio
from dotenv import load_dotenv

from lastminute_api.application.agent_service.service import (
    run_revision_agent,
    summarise_agent_result,
)


async def main() -> None:
    load_dotenv()

    query = "Create a mind map of the key concepts in AI, including machine learning, neural networks, and natural language processing. Also, generate a simple diagram illustrating a neural network."
    state = await run_revision_agent(query)
    summary = summarise_agent_result(state)

    print("Query type:", summary.get("query_type"))
    print("Answer:\n", summary.get("answer"))
    image_url = summary.get("image_url")
    if image_url:
        print("Image URL:", image_url[:120] + "...")


if __name__ == "__main__":
    asyncio.run(main())
