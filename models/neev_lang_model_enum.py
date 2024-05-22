from enum import Enum


class NeevLangModelIdentifier(Enum):
    OPENAI_GPT_35 = "OpenAI GPT-3.5"
    ANTHROPIC_CLAUDE3 = "Anthropic Claude3"
    OLLAMA_MISTRAL = "Ollama Mistral"
    TOGETHER_AI = "TogetherAI"
    # HUGGING_FACE = "HuggingFaceHub"