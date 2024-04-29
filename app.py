import streamlit as st
from dotenv import load_dotenv

from models.EmbeddingModelEnum import VoyageEmbedIdentifier
from models.LargeLangModel import OpenAIModel, OllamaModel, AnthropicModel
from models.LLModelEnum import OllamaModelIdentifier, AnthropicModelIdentifier
from service.ConversationService import ConversationalChainService
from service.EmbedService import EmbeddingCreatorService

load_dotenv()

if __name__ == "__main__":

    model = AnthropicModel(AnthropicModelIdentifier.CLAUDE3_HAIKU, VoyageEmbedIdentifier.VOYAGE_2)
    # model = OllamaModel(OllamaModelIdentifier.GEMMA_2B)
    # model = OpenAIModel()

    st.set_page_config("Neev AI")
    st.header(f"Neev Fintelligence with {model.get_model_name()}💁")

    user_question = st.text_input("Ask a Question from the uploaded files")
    if user_question:
        conv_service = ConversationalChainService(model)
        response = conv_service.get_response(user_question)
        st.write("Reply: ", response["output_text"])

    with st.sidebar:
        st.title("Menu:")
        data_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button",
                                      accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                embed_service = EmbeddingCreatorService(model)
                file_paths = embed_service.save_files_to_temp(data_files)
                embed_service.create_embeddings(file_paths)
                st.success("Done")
