from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

from models.large_lang_model import LargeLanguageModel
from service.vector_store import VectorStore


class ConversationalChainService:
    def __init__(self, large_lang_model: LargeLanguageModel, vector_store: VectorStore):
        self.large_lang_model = large_lang_model
        self.vector_store = vector_store

        self.prompt_template = """
        You are an expert AI assistant on finance domain. 
        Answer the question as detailed as possible from the provided context, 
        make sure to provide all the details, if the answer is not in provided context 
        just say, "Answer is not available in the context" \n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        self.model = large_lang_model.get_llm()
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        self.chain = load_qa_chain(self.model, prompt=self.prompt)
        self.vector_db = vector_store.get_db()

    def get_response(self, users_question):
        docs = self.vector_db.similarity_search(users_question)
        response = self.chain.invoke({"input_documents": docs, "question": users_question}, return_only_outputs=True)
        return response
