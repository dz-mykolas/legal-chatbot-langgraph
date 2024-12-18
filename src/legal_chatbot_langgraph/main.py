from dotenv import load_dotenv
load_dotenv(override=True)

from legal_chatbot_langgraph.graph import create_graph

graph = create_graph()

async def main():
    print("Starting the graph")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
