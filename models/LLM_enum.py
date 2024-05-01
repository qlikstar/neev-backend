from enum import Enum


class OllamaModelIdentifier(Enum):
    GEMMA_2B = "gemma:2b"
    GEMMA_7B = "gemma:7b"
    LLAMA3_70B = "llama3:70b"
    LLAMA3_8B = "llama3:8b"


'''
Anthropic Models: https://docs.anthropic.com/claude/docs/models-overview
'''


class AnthropicModelIdentifier(Enum):
    CLAUDE3_OPUS = "claude-3-opus-20240307"
    CLAUDE3_SONNET = "claude-3-sonnet-20240307"
    CLAUDE3_HAIKU = "claude-3-haiku-20240307"
