import asyncio
from datasets import load_dataset
import enum
from google import genai
import itertools
import json
import pydantic


class Response(pydantic.BaseModel):
    original: int
    square: int


with open("gemini-api-key.txt") as f:
    API_KEY = f.read()


MODEL = "gemini-2.0-flash-exp"
CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": genai.types.Content(
        parts=[
            genai.types.Part(
                text="""The input you receive will be exclusively positive integers.
                You always reply in **valid JSON format** like this:

                {"original": <number>, "square": <number squared>}
                """
            )
        ]
    )
}

client = genai.Client(api_key=API_KEY, http_options={'api_version': 'v1alpha'})


async def test():
    async with client.aio.live.connect(model=MODEL, config=CONFIG) as session:
        numbers = [10, 4, 2, 5, 6, 1, 2, 3, 8, 4]

        # Send all requests
        for i in numbers:
            print(f"new number: {i}")
            await session.send(input=str(i), end_of_turn=True)

            # Receive responses in the correct order
            async for response in session.receive():
                if response.text is not None:
                    print(f"{response.text}", end="")


if __name__ == "__main__":
    asyncio.run(test())

