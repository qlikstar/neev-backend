import streamlit as st
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel

from models.LLM_enum import OllamaModelIdentifier, AnthropicModelIdentifier, TogetherAiIdentifier
from models.embedding_enum import VoyageEmbedIdentifier
from models.large_lang_model import OpenAIModel, OllamaModel, AnthropicModel, TogetherAiModel
from models.neev_lang_model_enum import NeevLangModelIdentifier
from service.doc_processor import DocProcessorService
from service.vector_store import FaissVectorStore, ChromaVectorStore, BaseVectorStore
from templates import FinOpsTemplate
from tools.python_repl_tool import get_python_repl_tool
from tools.retriever_tool import VectorStoreRetrieverTool
from tools.web_search_tool import DuckDuckGoSearchTool, TavilySearchTool
from util.csv_reader_df import convert_csv_to_df
from util.startup import initialize


class AgentInputs(BaseModel):
    input: str


def invoke_user_form(number, st):
    with st.form(key=str(number)):
        user_question = st.text_area("Ask a Question from the uploaded files", key="question" + str(number))
        submitted = st.form_submit_button("Submit")

        if user_question and submitted:

            all_xls_data, df_dict = convert_csv_to_df()
            # Tools
            repl_tool = get_python_repl_tool(df_dict)
            retriever_tool = VectorStoreRetrieverTool(vector_store.get_vector_store()).get_tool()
            search_tool = DuckDuckGoSearchTool().get_tool()
            # search_tool = TavilySearchTool().get_tool()
            tools = [repl_tool, retriever_tool, search_tool]

            # Prompt
            template = FinOpsTemplate.FIN_OPS_TEMPLATE.format(xls_data=all_xls_data)
            # print(template)
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", template),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ("human", f"{user_question}"),
                ]
            )

            # Agent
            agent = create_openai_functions_agent(
                llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), prompt=prompt, tools=tools
            )
            agent_executor = AgentExecutor(
                agent=agent, tools=tools, max_iterations=10,
                early_stopping_method="generate",
                return_intermediate_steps= True,
                verbose=True,
            ) | (lambda x: x["output"])
            agent_executor = agent_executor.with_types(input_type=AgentInputs)

            print("Invoking ... Agent")
            try:
                st.write("Reply: " + agent_executor.invoke({"input": user_question}))
            except:
                st.write("Error: Could not find an answer. Please check error logs" )
            return True


if __name__ == "__main__":

    initialize()
    st.set_page_config(page_title="Your App Title", layout="wide")
    st.header(f"🤖Neev: FinOps Intelligence🌟")

    options = tuple(member.value for member in NeevLangModelIdentifier)
    default_index = options.index(NeevLangModelIdentifier.OPENAI_GPT_35.value)

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

    doc_processor = DocProcessorService()
    vector_store: BaseVectorStore = ChromaVectorStore(ll_model, doc_processor)
    invoke_user_form(0, st)

    with st.sidebar:
        st.title("Menu:")
        data_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button",
                                      accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                vector_store.create_vector_embeddings(data_files)
                st.success("Done")

        st.markdown("""---""")

        if st.button("Clear all files"):
            with st.spinner("Processing..."):
                doc_processor.clear_all()
                st.error("Cleared uploaded files")