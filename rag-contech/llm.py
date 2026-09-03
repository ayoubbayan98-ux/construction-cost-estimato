import os
from typing import Optional

from anthropic import Anthropic
from openai import OpenAI


class LLMClient:
    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

        if self.provider == "openai":
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

        elif self.provider == "anthropic":
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        elif self.provider == "kimi":
            self.client = OpenAI(
                api_key=os.getenv("KIMI_API_KEY"),
                base_url="https://api.moonshot.ai/v1"
            )

        else:
            raise ValueError(
                f"Unsupported LLM_PROVIDER: {self.provider}"
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1500,
    ) -> str:

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt or "",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        elif self.provider == "kimi":
            response = self.client.chat.completions.create(
                model=os.getenv("KIMI_MODEL"),
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt or "",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=max_tokens,
            )

            return response.choices[0].message.content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=os.getenv("CLAUDE_MODEL"),
                max_tokens=max_tokens,
                system=system_prompt or "",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            return response.content[0].text