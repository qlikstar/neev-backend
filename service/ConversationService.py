from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

from models.LargeLangModel import LargeLanguageModel


class ConversationalChainService:
    def __init__(self, large_lang_model: LargeLanguageModel):
        self.large_lang_model = large_lang_model

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
        self.vector_db = FAISS.load_local(self.large_lang_model.get_vector_store_name(),
                                          self.large_lang_model.get_embeddings(),
                                          allow_dangerous_deserialization=True)

    def get_response(self, users_question):
        docs = self.vector_db.similarity_search(users_question)
        response = self.chain({"input_documents": docs, "question": users_question}, return_only_outputs=True)
        return response
