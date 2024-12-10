from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
import warnings
warnings.filterwarnings("ignore")
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from typing import Tuple, List, Optional
from langchain_openai import AzureChatOpenAI

import os

os.environ['OPENAI_API_KEY'] = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# Configurations
NEO4J_URI = "neo4j+s://05163178.xxxxxxxxxxxx"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "xxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxx"
hf_api_key = "xxxxxxxxxxxxxxxxxxxxxxx"
llm = AzureChatOpenAI(
    openai_api_version="xxxxxx",
    azure_deployment="xxxxxx",
    temperature=0,
    azure_endpoint="xxxxxxxxxxxxx",
    api_key="xxxxxxxx"
)


class GraphRAGPipeline:
    def __init__(self):
        self.graph = None
        self.llm = None
        self.embedding_model = None

    def setup(self):
        self.graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        self.llm = AzureChatOpenAI(
            openai_api_version="xxxxxxxxxx",
            azure_deployment="xxxxxxxxxxxx",
            temperature=0,
            azure_endpoint="xxxxxxxxxxxxxxx",
            api_key="xxxxxxxxxxxxxxxxx"
        )

    def load_and_split_documents(self, pdf_files):
        documents = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
        return text_splitter.split_documents(documents)

    def add_documents_to_graph(self, documents):
        llm_transformer = LLMGraphTransformer(llm=self.llm)
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        self.graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )


class EntityExtractor(BaseModel):
    names: List[str] = Field(..., description="Extracted entities")


def generate_full_text_query(input_text):
    words = [el for el in remove_lucene_chars(input_text).split() if el]
    query = " AND ".join([f"{word}~2" for word in words])
    return query


class Retriever:
    def __init__(self, graph, entity_chain):
        self.graph = graph
        self.entity_chain = entity_chain
        self.vector_index = Neo4jVector.from_existing_graph(
            OpenAIEmbeddings(),
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )

    def structured_retriever(self, question: str) -> str:
        result = ""
        entities = self.entity_chain.invoke({"question": question})
        for entity in entities.names:
            response = self.graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result



    def retriever(self, question: str):
        print(f"Search query: {question}")
        structured_data = self.structured_retriever(question)
        unstructured_data = [el.page_content for el in self.vector_index.similarity_search(question)]
        final_data = f"""Structured data: {structured_data}
Unstructured data:
{"#Document ".join(unstructured_data)}
        """
        return final_data

    def format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer


def main():
    pipeline = GraphRAGPipeline()
    pipeline.setup()

    pdf_files = ["files here to be uploaded"]
    documents = pipeline.load_and_split_documents(pdf_files)
    pipeline.add_documents_to_graph(documents)

    prompt_1 = ChatPromptTemplate.from_messages([
        ("system", "You are extracting entities."),
        ("human", "Extract from {question}")
    ])
    entity_chain = prompt_1 | pipeline.llm.with_structured_output(EntityExtractor)

    retriever = Retriever(pipeline.graph, entity_chain)

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
    in its original language.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    Use natural language and be concise.
    Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    _search_query = RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            RunnablePassthrough.assign(
                chat_history=lambda x: Retriever.format_chat_history(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser(),
        ),
        RunnableLambda(lambda x: x["question"]),
    )

    chain = (
            RunnableParallel(
                {
                    "context": _search_query | retriever.retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
    )




    result =chain.invoke({"question": "Can you tell me who approves my time sheets?"})
    print(result)


if __name__ == "__main__":
    main()
