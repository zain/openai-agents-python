import asyncio

from agents import Agent, Runner, ModelSettings, OpenAIResponsesModel


async def main():
    agent = Agent(
        name="English agent",
        instructions="You only speak English",
        model=OpenAIResponsesModel(
            model="gpt-4o",
            model_settings=ModelSettings(
                store=False,
            )
        )
    )

    result = await Runner.run(agent, "Tell me about recursion in programming.")
    print(result.final_output)
    # Function calls itself,
    # Looping in smaller pieces,
    # Endless by design.


if __name__ == "__main__":
    asyncio.run(main())
