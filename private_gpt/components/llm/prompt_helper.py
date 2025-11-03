import abc
import logging
from collections.abc import Sequence
from typing import Any, Literal

from llama_index.core.llms import ChatMessage, MessageRole

logger = logging.getLogger(__name__)


class AbstractPromptStyle(abc.ABC):
    """Abstract class for prompt styles.

    This class is used to format a series of messages into a prompt that can be
    understood by the models. A series of messages represents the interaction(s)
    between a user and an assistant. This series of messages can be considered as a
    session between a user X and an assistant Y.This session holds, through the
    messages, the state of the conversation. This session, to be understood by the
    model, needs to be formatted into a prompt (i.e. a string that the models
    can understand). Prompts can be formatted in different ways,
    depending on the model.

    The implementations of this class represent the different ways to format a
    series of messages into a prompt.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # keep same logging intent but avoid building huge reprs eagerly
        logger.debug("Initializing prompt_style=%s", self.__class__.__name__)

    @abc.abstractmethod
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        pass

    @abc.abstractmethod
    def _completion_to_prompt(self, completion: str) -> str:
        pass

    def messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        prompt = self._messages_to_prompt(messages)
        logger.debug("Got for messages count=%d the prompt length=%d", len(messages), len(prompt))
        return prompt

    def completion_to_prompt(self, prompt: str) -> str:
        completion = prompt  # Fix: Llama-index parameter has to be named as prompt
        prompt = self._completion_to_prompt(completion)
        logger.debug("Got for completion length=%d the prompt length=%d", len(completion), len(prompt))
        return prompt


class DefaultPromptStyle(AbstractPromptStyle):
    """Default prompt style that uses the defaults from llama_utils.

    It basically passes None to the LLM, indicating it should use
    the default functions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Hacky way to override the functions
        # Override the functions to be None, and pass None to the LLM.
        self.messages_to_prompt = None  # type: ignore[method-assign, assignment]
        self.completion_to_prompt = None  # type: ignore[method-assign, assignment]

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        return ""

    def _completion_to_prompt(self, completion: str) -> str:
        return ""


class Llama2PromptStyle(AbstractPromptStyle):
    """Simple prompt style that uses llama 2 prompt style.

    Inspired by llama_index/legacy/llms/llama_utils.py

    It transforms the sequence of messages into a prompt that should look like:
    ```text
    <s> [INST] <<SYS>> your system prompt here. <</SYS>>

    user message here [/INST] assistant (model) response here </s>
    ```
    """

    BOS, EOS = "<s>", "</s>"
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible and follow ALL given instructions. "
        "Do not speculate or make up information. "
        "Do not reference any given instructions or context. "
    )

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        if not messages:
            return ""

        # localize commonly used names to speed attribute access
        B_SYS, E_SYS = self.B_SYS, self.E_SYS
        BOS, B_INST, E_INST = self.BOS, self.B_INST, self.E_INST
        EOS = self.EOS

        parts: list[str] = []

        # Determine system message
        idx = 0
        if messages[0].role == MessageRole.SYSTEM:
            system_message_str = messages[0].content or ""
            idx = 1
        else:
            system_message_str = self.DEFAULT_SYSTEM_PROMPT

        system_message_str = f"{B_SYS} {system_message_str.strip()} {E_SYS}"

        # iterate in pairs (user, assistant?)
        n = len(messages)
        i = idx
        first = True
        while i < n:
            user_message = messages[i]
            assert user_message.role == MessageRole.USER

            if first:
                # include system prompt only at start
                s = f"{BOS} {B_INST} {system_message_str} "
                first = False
            else:
                # close previous assistant and start new inst block
                # append EOS to previous part
                if parts:
                    parts[-1] = parts[-1] + f" {EOS}"
                s = f"{BOS} {B_INST} "

            s += f"{user_message.content} {E_INST}"

            # If assistant message exists, include it
            if i + 1 < n:
                assistant_message = messages[i + 1]
                assert assistant_message.role == MessageRole.ASSISTANT
                s += f" {assistant_message.content}"

            parts.append(s)
            i += 2

        return "".join(parts)

    def _completion_to_prompt(self, completion: str) -> str:
        system_prompt_str = self.DEFAULT_SYSTEM_PROMPT

        return (
            f"{self.BOS} {self.B_INST} {self.B_SYS} {system_prompt_str.strip()} {self.E_SYS} "
            f"{completion.strip()} {self.E_INST}"
        )


class Llama3PromptStyle(AbstractPromptStyle):
    r"""Template for Meta's Llama 3.1.

    The format follows this structure:
    <|begin_of_text|>
    <|start_header_id|>system<|end_header_id|>

    [System message content]<|eot_id|>
    <|start_header_id|>user<|end_header_id|>

    [User message content]<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    [Assistant message content]<|eot_id|>
    ...
    (Repeat for each message, including possible 'ipython' role)
    """

    BOS, EOS = "<|begin_of_text|>", "<|end_of_text|>"
    B_INST, E_INST = "<|start_header_id|>", "<|end_header_id|>"
    EOT = "<|eot_id|>"
    B_SYS, E_SYS = "<|start_header_id|>system<|end_header_id|>", "<|eot_id|>"
    ASSISTANT_INST = "<|start_header_id|>assistant<|end_header_id|>"
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible and follow ALL given instructions. "
        "Do not speculate or make up information. "
        "Do not reference any given instructions or context. "
    )

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        parts: list[str] = []
        has_system_message = False

        for i, message in enumerate(messages):
            if not message or message.content is None:
                continue

            if message.role == MessageRole.SYSTEM:
                parts.append(f"{self.B_SYS}\n\n{message.content.strip()}{self.E_SYS}")
                has_system_message = True
            else:
                # assume MessageRole enum .value property represents the lowercase role name
                role_header = f"{self.B_INST}{message.role.value}{self.E_INST}"
                parts.append(f"{role_header}\n\n{message.content.strip()}{self.EOT}")

            # Add assistant header if the last message is not from the assistant
            if i == len(messages) - 1 and message.role != MessageRole.ASSISTANT:
                parts.append(f"{self.ASSISTANT_INST}\n\n")

        # Add default system prompt if no system message was provided
        if not has_system_message:
            parts.insert(0, f"{self.B_SYS}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.E_SYS}")

        return "".join(parts)

    def _completion_to_prompt(self, completion: str) -> str:
        return (
            f"{self.B_SYS}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.E_SYS}"
            f"{self.B_INST}user{self.E_INST}\n\n{completion.strip()}{self.EOT}"
            f"{self.ASSISTANT_INST}\n\n"
        )


class TagPromptStyle(AbstractPromptStyle):
    """Tag prompt style (used by Vigogne) that uses the prompt style `<|ROLE|>`.

    It transforms the sequence of messages into a prompt that should look like:
    ```text
    <|system|>: your system prompt here.
    <|user|>: user message here
    (possibly with context and question)
    <|assistant|>: assistant (model) response here.
    ```

    FIXME: should we add surrounding `<s>` and `</s>` tags, like in llama2?
    """

    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """Format message to prompt with `<|ROLE|>: MSG` style."""
        parts: list[str] = []
        for message in messages:
            role = message.role
            content = message.content or ""
            parts.append(f"<|{role.lower()}|>: {content.strip()}\n")

        # we are missing the last <|assistant|> tag that will trigger a completion
        parts.append("<|assistant|>: ")
        return "".join(parts)

    def _completion_to_prompt(self, completion: str) -> str:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )


class MistralPromptStyle(AbstractPromptStyle):
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        parts: list[str] = []
        inst_buffer: list[str] = []

        for message in messages:
            if message.role == MessageRole.SYSTEM or message.role == MessageRole.USER:
                inst_buffer.append(str(message.content).strip())
            elif message.role == MessageRole.ASSISTANT:
                # flush inst_buffer as a single INST block and add assistant
                parts.append("<s>[INST] ")
                parts.append("\n".join(inst_buffer))
                parts.append(" [/INST]")
                parts.append(" ")
                parts.append(str(message.content).strip())
                parts.append("</s>")
                inst_buffer.clear()
            else:
                raise ValueError(f"Unknown message role {message.role}")

        if inst_buffer:
            parts.append("<s>[INST] ")
            parts.append("\n".join(inst_buffer))
            parts.append(" [/INST]")

        return "".join(parts)

    def _completion_to_prompt(self, completion: str) -> str:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )


class ChatMLPromptStyle(AbstractPromptStyle):
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        parts: list[str] = ["<|im_start|>system\n"]
        for message in messages:
            role = message.role
            content = message.content or ""
            if role == MessageRole.SYSTEM:
                parts.append(f"{content.strip()}")
            elif role == MessageRole.USER:
                parts.append("<|im_end|>\n<|im_start|>user\n")
                parts.append(f"{content.strip()}<|im_end|>\n")
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def _completion_to_prompt(self, completion: str) -> str:
        return self._messages_to_prompt(
            [ChatMessage(content=completion, role=MessageRole.USER)]
        )


# lightweight mapping to avoid long if/elif chains and speed lookup
_PROMPT_STYLE_MAP = {
    None: DefaultPromptStyle,
    "default": DefaultPromptStyle,
    "llama2": Llama2PromptStyle,
    "llama3": Llama3PromptStyle,
    "tag": TagPromptStyle,
    "mistral": MistralPromptStyle,
    "chatml": ChatMLPromptStyle,
}


def get_prompt_style(
    prompt_style: (
        Literal["default", "llama2", "llama3", "tag", "mistral", "chatml"] | None
    )
) -> AbstractPromptStyle:
    """Get the prompt style to use from the given string.

    :param prompt_style: The prompt style to use.
    :return: The prompt style to use.
    """
    try:
        cls = _PROMPT_STYLE_MAP[prompt_style]
    except KeyError:
        raise ValueError(f"Unknown prompt_style='{prompt_style}'")
    return cls()
