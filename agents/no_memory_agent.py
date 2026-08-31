from __future__ import annotations

import os

from .base_agent import BaseAgent
from .concat_agent import QA_PROMPT


class NoMemoryAgent(BaseAgent):
    """Answer from the query only.

    This is a depth-0 control for Adaptive Memory Computation. It keeps the
    same final QA prompt style as UMA's concat agent but does not expose any
    stored memory chunks to the model.
    """

    async def add_memory_async(self, chunk: str):
        return None

    def add_memory(self, chunk: str):
        return None

    def reset(self) -> None:
        return None

    async def QA_batch_async(self, query_list: list[str], batch_size: int = 5) -> list[str]:
        return [await self.QA_async(query) for query in query_list]

    async def QA_async(self, query: str) -> str:
        try:
            prompt = f"Your memory:\nNo previous memory\n\n{QA_PROMPT.format(query)}"
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,
                temperature=float(os.getenv("UMA_TEMPERATURE", "0.7")),
                top_p=float(os.getenv("UMA_TOP_P", "1.0")),
            )
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("Received empty response from API")
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            return content.strip()
        except Exception as exc:
            return self._handle_api_error(exc, query)
