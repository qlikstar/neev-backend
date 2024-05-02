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


class HuggingFaceModelIdentifier(Enum):
    HF_FINANCE_13B = "AdaptLLM/finance-LLM-13B"
    HF_BLOKE_FIN_GGUP = "TheBloke/finance-LLM-GGUF"
    HF_SUJET_FIN_8B = "sujet-ai/Sujet-Finance-8B-v0.1"
    HF_ADAPT_FIN_CHAT = "AdaptLLM/finance-chat"
