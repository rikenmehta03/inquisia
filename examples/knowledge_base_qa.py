import asyncio
from dotenv import load_dotenv

load_dotenv()

from inquisia.knowledge_base_qa import KnowledgeBaseQA

model_name = "gpt-4-1106-preview"

kb_qa = KnowledgeBaseQA(model=model_name, knowledge_base="test_kb")

# print(
#     kb_qa.generate_answer(
#         "What are the potential rooms of improvements of bitsandbytes?"
#     )
# )


# main coroutine
async def main():
    # create the async generator
    gen = kb_qa.generate_stream_answer(
        "What are the potential rooms of improvements of bitsandbytes?"
    )
    tokens = ""
    async for token in gen:
        tokens += token
    print(tokens)


asyncio.run(main())
