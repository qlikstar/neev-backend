from time import sleep

import streamlit as st
from langchain.agents import AgentExecutor, create_openai_functions_agent, OpenAIFunctionsAgent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
from util.constants import SAMPLE_QUESTIONS, OUTPUT_FILE_PATH, INPUT_FILE_PATH
from util.csv_reader_df import convert_csv_to_df
from util.startup import initialize


class AgentInputs(BaseModel):
    input: str


def add_question_ans(question, ans):
    if question:
        st.session_state.question_ans.append({"question": question, "ans": ans})


def get_tools(df_dict, vector_store):
    # Tools
    repl_tool = get_python_repl_tool(df_dict)
    retriever_tool = VectorStoreRetrieverTool(vector_store.get_vector_store()).get_tool()
    search_tool = DuckDuckGoSearchTool().get_tool()
    # search_tool = TavilySearchTool().get_tool()
    return [repl_tool, retriever_tool, search_tool]


def populate_filenames_in_buffer(doc_processor):
    files = doc_processor.get_all_file_names(INPUT_FILE_PATH)
    if len(files) == 0:
        st.error("**Please upload a file to process!**")
    else:
        files_present_message = "**Files present in the memory:**\n"
        for file in files:
            files_present_message += f"- {file}\n"
        st.success(files_present_message)


def populate_filenames_in_output(doc_processor):
    files = doc_processor.get_all_file_names(OUTPUT_FILE_PATH)
    if len(files) > 0:
        files_present_message = "**Files present in the output:**\n"
        for file in files:
            files_present_message += f"- {file}\n"
        st.success(files_present_message)


def main():
    st.set_page_config(page_title="Neev", layout="wide")
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
    else:
        st.error("Invalid selection")
        exit(0)

    st.warning(SAMPLE_QUESTIONS)

    doc_processor = DocProcessorService()
    vector_store: BaseVectorStore = ChromaVectorStore(ll_model, doc_processor)

    for i, question_ans in enumerate(st.session_state['question_ans']):
        with st.container(border=True):
            st.write("**Question:** " + question_ans["question"])
            st.markdown("""---""")
            st.write(question_ans["ans"])

    with st.form(key="form"):
        user_question = st.text_area("Ask a Question from the uploaded files")
        submitted = st.form_submit_button("Submit")

        if user_question and submitted:

            all_xls_data, df_dict = convert_csv_to_df()
            tools = get_tools(df_dict, vector_store)
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
            agent = OpenAIFunctionsAgent(
                llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"), prompt=prompt, tools=tools
            )
            agent_executor = AgentExecutor(
                agent=agent, tools=tools, max_iterations=10,
                early_stopping_method="generate",
                return_intermediate_steps=True,
                verbose=True,
                handle_parsing_errors=True
            ) | (lambda x: x["output"])
            agent_executor = agent_executor.with_types(input_type=AgentInputs)

            print("Invoking ... Agent")

            try:
                answer = agent_executor.invoke({"input": user_question})
            except Exception as e:
                st.write(f"Error: Could not find an answer. Please check error logs: {e}")
            else:
                st.write("Reply: " + answer)
                add_question_ans(user_question, answer)
            finally:
                st.rerun()

    with st.sidebar:
        st.title("Menu:")
        populate_filenames_in_buffer(doc_processor)
        populate_filenames_in_output(doc_processor)

        st.markdown("""---""")
        data_files = st.file_uploader("Upload your Files and Click on the Submit & Process Button",
                                      accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                vector_store.create_vector_embeddings(data_files)
                st.success("Done")
                sleep(1)
                st.rerun()

        st.markdown("""---""")
        if st.button("Clear all files"):
            with st.spinner("Processing..."):
                doc_processor.clear_all()
                st.session_state.question_ans = []
                st.error("Cleared uploaded files")
                st.rerun()


if __name__ == "__main__":
    initialize()
    main()
