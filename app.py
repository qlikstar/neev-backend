import streamlit as st

from models.neev_lang_model_enum import NeevLangModelIdentifier
from models.embedding_enum import VoyageEmbedIdentifier
from models.large_lang_model import OpenAIModel, OllamaModel, AnthropicModel, HuggingFaceModel, TogetherAiModel
from models.LLM_enum import OllamaModelIdentifier, AnthropicModelIdentifier, HuggingFaceModelIdentifier, \
    TogetherAiIdentifier
from service.conversation import ConversationalChainService
from service.doc_processor import DocProcessorService
from service.vector_store import FaissVectorStore, ChromaVectorStore
from util.startup import initialize


def invoke_user_form(number, st, conv_service):
    key = "user_form" + str(number)
    with st.form(key=str(number)):
        user_question = st.text_area("Ask a Question from the uploaded files", key="question" + str(number))
        submitted = st.form_submit_button("Submit")

        if key not in st.session_state:
            st.session_state[key] = False

        if (user_question and submitted) or st.session_state[key]:
            st.session_state[key] = True
            response = conv_service.get_response(user_question)
            st.write("Reply: " + response["output_text"])
            return True


if __name__ == "__main__":

    initialize()
    st.set_page_config("Neev AI")
    st.header(f"🤖 Neev: FinOps Intelligence 🌟")

    options = tuple(member.value for member in NeevLangModelIdentifier)
    default_index = options.index(NeevLangModelIdentifier.ANTHROPIC_CLAUDE3.value)

    option = st.selectbox("Please select a Large language model",
                          options,
                          index=default_index,
                          placeholder="Select an LLM")

    if option == NeevLangModelIdentifier.ANTHROPIC_CLAUDE3.value:
        ll_model = AnthropicModel(AnthropicModelIdentifier.CLAUDE3_HAIKU, VoyageEmbedIdentifier.VOYAGE_2)
    elif option == NeevLangModelIdentifier.OPENAI_GPT_35.value:
        ll_model = OpenAIModel()
    elif option == NeevLangModelIdentifier.OLLAMA_MISTRAL.value:
        ll_model = OllamaModel(OllamaModelIdentifier.MISTRAL)
    elif option == NeevLangModelIdentifier.TOGETHER_AI.value:
        ll_model = TogetherAiModel(TogetherAiIdentifier.LLAMA3_70B_CHAT)
    # elif option == NeevLangModelIdentifier.HUGGING_FACE:
    #     model = HuggingFaceModel(HuggingFaceModelIdentifier.HF_ADAPT_FIN_CHAT)
    else:
        st.error("Invalid selection")
        exit(0)

    vector_store = ChromaVectorStore(ll_model)

    conv_service = ConversationalChainService(ll_model, vector_store)
    num = 1
    while invoke_user_form(num, st, conv_service):
        num += 1

    with st.sidebar:
        st.title("Menu:")
        data_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button",
                                      accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                doc_processor_service = DocProcessorService()
                chunked_docs = doc_processor_service.get_chunked_documents(data_files)
                vector_store.create_vector_embeddings(chunked_docs)
                st.success("Done")

        st.markdown("""---""")
