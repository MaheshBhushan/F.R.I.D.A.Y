"""Optional local Hugging Face generation transport for the reasoning seam."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

from friday.brain import LLMEvent, ToolCall


class TransformersTransport:
    """Generate with a local Transformers model; tool calls use a JSON envelope.

    Models should return ``{"text": "...", "tool_calls": [...]}`` and tool
    calls must use the names and inputs from the supplied tool schemas.
    Generation is non-streaming; the Transport protocol still emits its final
    text through the normal event path.
    """

    def __init__(self, model: str, *, device: str = "auto", max_new_tokens: int = 512):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install FRIDAY's local extra: pip install 'friday[local]'") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model, device_map=device)
        self._max_new_tokens = max_new_tokens

    @staticmethod
    def _messages(request: dict) -> list[dict]:
        system = "\n".join(block["text"] for block in request.get("system", []))
        tools = request.get("tools") or []
        if tools:
            system += "\nAvailable tools (request only as JSON tool_calls):\n" + json.dumps(tools)
        messages = [{"role": "system", "content": system}]
        for message in request.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(item.get("text", item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            messages.append({"role": message["role"], "content": content})
        return messages

    def _generate(self, request: dict) -> tuple[str, tuple[ToolCall, ...]]:
        import torch

        messages = self._messages(request)
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text, ()
        if not isinstance(payload, dict) or not isinstance(payload.get("tool_calls", []), list):
            return text, ()
        if "text" not in payload and "tool_calls" not in payload:
            return text, ()
        calls = tuple(
            ToolCall(
                id=f"local-{index}",
                name=call["name"],
                input=call.get("input", {}),
            )
            for index, call in enumerate(payload["tool_calls"])
            if isinstance(call, dict) and isinstance(call.get("name"), str)
            and isinstance(call.get("input", {}), dict)
        )
        return str(payload.get("text", "")), calls

    async def stream(self, request: dict) -> AsyncIterator[LLMEvent]:
        text, calls = await asyncio.to_thread(self._generate, request)
        if text:
            yield LLMEvent(kind="text", text=text)
        content = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.input}
            for call in calls
        ]
        yield LLMEvent(
            kind="turn_end", tool_calls=calls,
            stop_reason="tool_use" if calls else "end_turn",
            assistant_content=content,
        )
