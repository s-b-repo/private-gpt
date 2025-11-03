# mypy: ignore-errors
from __future__ import annotations

import io
import json
import logging
from typing import TYPE_CHECKING, Any

import boto3  # type: ignore
from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.core.llms import (
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from llama_index.callbacks import CallbackManager
    from llama_index.llms import (
        ChatMessage,
        ChatResponse,
        ChatResponseGen,
        CompletionResponseGen,
    )

logger = logging.getLogger(__name__)


class LineIterator:
    r"""A lighter-weight, faster line iterator for TGI-style event streams.

    Notes on changes (kept behavior):
    - Uses a bytearray buffer instead of io.BytesIO to avoid stream/seeking overhead.
    - Uses index arithmetic and in-place buffer truncation to reduce allocations.
    - Preserves semantics of only yielding lines that end with ``\n`` (same as original).
    """

    def __init__(self, stream: Any) -> None:
        """Line iterator initializer.

        `stream` must be an iterable yielding dict-like events with
        a ``PayloadPart`` key containing a ``Bytes`` value.
        """
        self._byte_iter = iter(stream)
        self._buf = bytearray()
        self._read_pos = 0

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> bytes:
        # Localize frequently used names for speed
        buf = self._buf
        read_pos = self._read_pos
        byte_iter = self._byte_iter

        while True:
            nl = buf.find(b"\n", read_pos)
            if nl != -1:
                # Found a full line (ending in \n) -- return it without the newline
                line = bytes(buf[read_pos:nl])
                read_pos = nl + 1
                # If buffer has grown large, drop consumed prefix to avoid unbounded memory
                if read_pos > 4096:
                    # delete consumed prefix and reset read_pos
                    del buf[:read_pos]
                    read_pos = 0
                # update instance state then return
                self._buf = buf
                self._read_pos = read_pos
                return line

            # No newline available; try to read next chunk from the underlying iterator
            try:
                chunk = next(byte_iter)
            except StopIteration:
                # No more input coming. Match original behavior: do not return a
                # partial (unterminated) line -- raise StopIteration.
                raise

            # Validate structure quickly and append bytes
            if not chunk or "PayloadPart" not in chunk:
                logger.warning("Unknown event type=%s", chunk)
                continue

            data_bytes = chunk["PayloadPart"]["Bytes"]
            if data_bytes:
                buf.extend(data_bytes)
            # loop will check for newline again
            # keep local read_pos in sync for next loop iteration
            self._buf = buf
            self._read_pos = read_pos


class SagemakerLLM(CustomLLM):
    """Sagemaker Inference Endpoint models.

    To use, you must supply the endpoint name from your deployed
    Sagemaker model & the region where it is deployed.
    """

    endpoint_name: str = Field(description="")
    temperature: float = Field(description="The temperature to use for sampling.")
    max_new_tokens: int = Field(description="The maximum number of tokens to generate.")
    context_window: int = Field(
        description="The maximum number of context tokens for the model."
    )
    messages_to_prompt: Any = Field(
        description="The function to convert messages to a prompt.", exclude=True
    )
    completion_to_prompt: Any = Field(
        description="The function to convert a completion to a prompt.", exclude=True
    )
    generate_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    verbose: bool = Field(description="Whether to print verbose output.")

    # Keep a module-level boto3 client to avoid recreating it repeatedly.
    _boto_client: Any = boto3.client(
        "sagemaker-runtime",
    )

    def __init__(
        self,
        endpoint_name: str | None = "",
        temperature: float = 0.1,
        max_new_tokens: int = 512,  # to review defaults
        context_window: int = 2048,  # to review defaults
        messages_to_prompt: Any = None,
        completion_to_prompt: Any = None,
        callback_manager: CallbackManager | None = None,
        generate_kwargs: dict[str, Any] | None = None,
        model_kwargs: dict[str, Any] | None = None,
        verbose: bool = True,
    ) -> None:
        """SagemakerLLM initializer."""
        model_kwargs = model_kwargs or {}
        model_kwargs.update({"n_ctx": context_window, "verbose": verbose})

        messages_to_prompt = messages_to_prompt or {}
        completion_to_prompt = completion_to_prompt or {}

        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {"temperature": temperature, "max_tokens": max_new_tokens}
        )

        super().__init__(
            endpoint_name=endpoint_name,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
        )

        # Ensure instance reuses the module-level client (no expensive call here)
        # This keeps behavior identical while making attribute access slightly faster.
        self._boto_client = self.__class__._boto_client

    @property
    def inference_params(self):
        # TODO expose the rest of params
        return {
            "do_sample": True,
            "top_p": 0.7,
            "temperature": self.temperature,
            "top_k": 50,
            "max_new_tokens": self.max_new_tokens,
        }

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_new_tokens,
            model_name="Sagemaker LLama 2",
        )

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # Ensure non-streaming mode for complete()
        self.generate_kwargs.update({"stream": False})

        is_formatted = kwargs.pop("formatted", False)
        if not is_formatted:
            prompt = self.completion_to_prompt(prompt)

        request_params = {
            "inputs": prompt,
            "stream": False,
            "parameters": self.inference_params,
        }

        # Localize boto client for faster attribute lookup
        client = self._boto_client
        resp = client.invoke_endpoint(
            EndpointName=self.endpoint_name,
            Body=json.dumps(request_params),
            ContentType="application/json",
        )

        response_body = resp["Body"]
        response_str = response_body.read().decode("utf-8")
        response_dict = json.loads(response_str)

        # Return only the generated portion (same behavior)
        return CompletionResponse(
            text=response_dict[0]["generated_text"][len(prompt) :], raw=resp
        )

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        def get_stream():
            text = ""

            request_params = {
                "inputs": prompt,
                "stream": True,
                "parameters": self.inference_params,
            }

            client = self._boto_client
            resp = client.invoke_endpoint_with_response_stream(
                EndpointName=self.endpoint_name,
                Body=json.dumps(request_params),
                ContentType="application/json",
            )

            event_stream = resp["Body"]
            start_json = b"{"
            stop_token = "<|endoftext|>"
            first_token = True

            for line in LineIterator(event_stream):
                # Avoid processing empty lines
                if not line:
                    continue

                # Search for a JSON start and decode only the slice that contains it
                idx = line.find(start_json)
                if idx == -1:
                    continue

                try:
                    data = json.loads(line[idx:].decode("utf-8"))
                except Exception:
                    logger.exception("Failed to parse json from stream line")
                    continue

                token = data.get("token", {})
                special = token.get("special", False)
                token_text = token.get("text", "")
                stop = token_text == stop_token

                if not special and not stop:
                    delta = token_text
                    # trim the leading space for the first token if present
                    if first_token:
                        delta = delta.lstrip()
                        first_token = False
                    text += delta
                    yield CompletionResponse(delta=delta, text=text, raw=data)

        return get_stream()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)
