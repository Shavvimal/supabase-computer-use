import os
import asyncio
import requests
from urllib.parse import quote
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START
from dotenv import load_dotenv
from logging import getLogger
from brave import Brave
from rag.similarity_search import SimilaritySearch

# Ensure environment variables are loaded
_ = load_dotenv()


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: The user's original question.
        web_search: Indicator for whether to perform a web search.
        documents: List of documents retrieved from searches.
    """
    question: str
    web_search: str
    documents: List[Document]
    retry_count: int
    computer_use_instructions: str


class AgenticRAG:
    def __init__(self):
        self.workflow = None
        self.llm = self.setup_llm()
        # Chains
        self.retrieval_grader = None
        self.question_rewriter = None
        self.logger = getLogger(f"API.{__name__}")
        # Similarity search tool
        self.similarity_search_tool = SimilaritySearch()

        # Set up the workflow
        self.setup_retrieval_grader()
        self.setup_question_rewriter()
        self.setup_workflow()

    def setup_llm(self):
        return AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_ID"),
            api_version=os.getenv("AZURE_OPENAI_CHAT_API_VERSION"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # organization="...",
            # model="gpt-35-turbo",
            # model_version="0125",
            # other params...
        )

    def setup_retrieval_grader(self):
        class GradeDocuments(BaseModel):
            """Binary score for relevance check on retrieved documents."""
            binary_score: str = Field(
                description="Documents are relevant to the question, 'yes' or 'no'"
            )

        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        system = """You are a grader assessing the relevance of retrieved content to a user's question. The user is seeking instructions or information about how to use Supabase to achieve a specific goal. If the document contains relevant information that can help answer the user's question, grade it as relevant. Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n{document}\n\nUser question: {question}"),
            ]
        )

        self.retrieval_grader = grade_prompt | structured_llm_grader

    def setup_question_rewriter(self):
        system = """You are a question re-writer that converts an input question to a better query optimized for fetching documentation to carry out actions in Supabase. Look at the input, try to understand the underlying intent, and make the question more specific to Supabase."""

        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n{question}\nFormulate an improved question.",
                ),
            ]
        )

        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()

    def brave_search(self, query: str, num_results: int = 3):
        brave = Brave()
        search_results = brave.search(q=query, count=num_results)

        urls = [f"https://r.jina.ai/{quote(str(doc.get('url')), safe=':/')}"
                for doc in search_results.web_results]

        fetched_contents = []
        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                fetched_contents.append({'url': url, 'status_code': response.status_code, 'content': response.text})
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching content from {url}: {e}")
        return fetched_contents

    def tavily_search(self, query: str, num_results: int = 3):
        tavily_search_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"), k=num_results)
        search_results = tavily_search_tool.invoke({'query': query})
        urls = [f"https://r.jina.ai/{result['url']}" for result in search_results]

        fetched_contents = []
        for url in urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                fetched_contents.append({'url': url, 'status_code': response.status_code, 'content': response.text})
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error fetching content from {url}: {e}")

        return fetched_contents

    def web_search(self, state):
        """
        Retrieve documents using a web search. Start with Brave, fallback to Tavily.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with retrieved documents.
        """

        self.logger.info("---RETRIEVE FROM WEB SEARCH---")
        question = state["question"] + " in Supabase"
        documents = []

        try:
            brave_search_res = self.brave_search(question, num_results=1)
            if not brave_search_res:
                raise ValueError("Brave search returned no results.")

            for result in brave_search_res:
                documents.append(Document(page_content=result["content"]))
            self.logger.info(f"---SUCCESS: RETRIEVED {len(brave_search_res)} RESULTS FROM BRAVE---")

        except Exception as e:
            self.logger.info(f"---FAILED: BRAVE SEARCH ERROR: {e}, FALLING BACK TO TAVILY---")

            try:
                tavily_search_res = self.tavily_search(question)
                if not tavily_search_res:
                    raise ValueError("Tavily search returned no results.")

                for result in tavily_search_res:
                    documents.append(Document(page_content=result["content"]))
                self.logger.info(f"---SUCCESS: RETRIEVED {len(tavily_search_res)} RESULTS FROM TAVILY---")

            except Exception as e:
                self.logger.info(f"---FAILED: TAVILY SEARCH ERROR: {e}---")
                documents.append(Document(page_content="No results found from web search."))

        return {"documents": documents, "question": state["question"]}  # Keep the original question

    def grade_documents(self, state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with filtered relevant documents.
        """

        self.logger.info("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = self.retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = score.binary_score.lower()
            if grade == "yes":
                self.logger.info("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                self.logger.info("---GRADE: DOCUMENT NOT RELEVANT---")

        if not filtered_docs:
            web_search = "Yes"

        return {"documents": filtered_docs, "question": question, "web_search": web_search}

    def similarity_search(self, state):
        """
        Perform similarity search using the question and update the state with retrieved documents.
        """
        self.logger.info("---RUNNING SIMILARITY SEARCH---")
        question = state["question"]

        # Run the similarity search
        documents = asyncio.run(self.similarity_search_tool.query_similar_docs(question))

        # Update state
        state_documents = state.get("documents", [])
        state_documents.extend(documents)

        return {"documents": state_documents, "question": question}

    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state.

        Returns:
            state (dict): Updated state with the rephrased question.
        """
        self.logger.info("---TRANSFORM QUERY---")
        question = state["question"]
        better_question = self.question_rewriter.invoke({"question": question})
        retry_count = state.get('retry_count', 0) + 1
        return {
            "documents": state["documents"],
            "question": better_question,
            "retry_count": retry_count
        }

    def decide_to_generate(self, state):
        """
        Edge that determines whether to generate an answer or re-generate a question.

        Args:
            state (dict): The current graph state.

        Returns:
            str: Decision for the next node to call.
        """

        self.logger.info("---ASSESS GRADED DOCUMENTS---")
        retry_count = state.get('retry_count', 0)
        if retry_count >= 2:  # Set your desired max retries
            self.logger.info("---DECISION: MAX RETRIES REACHED, PROCEEDING TO SIMILARITY SEARCH---")
            return "similarity_search"
        web_search = state["web_search"]
        if web_search == "Yes":
            self.logger.info("---DECISION: DOCUMENTS NOT RELEVANT, TRANSFORM QUERY---")
            return "transform_query"
        else:
            self.logger.info("---DECISION: PROCEED TO SIMILARITY SEARCH---")
            return "similarity_search"

    def generate(self, state):
        """
        Generate the final answer using the retrieved documents.
        """
        self.logger.info("---GENERATING FINAL ANSWER---")
        question = state["question"]
        documents = state["documents"]
        # Concatenate the content of the documents
        context = "\n\n".join([doc.page_content for doc in documents])

        # Create the prompt
        prompt_template = PromptTemplate(
            template="""
            You are a teacher. Your job is to teach students on how to use Supabase. Supabase is an open-source platform for building and deploying web and mobile apps built on PostgreSQL. You will be taking a user's question on how to use the platform and you will return detailed step-by-step instructions on how the user will achieve their goal using the GUI of Supabase.

            Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know.

            Question: {question}
            Context: {context}

            Instructions:
            """,
            input_variables=["question", "context"],
        )

        prompt = prompt_template.format(question=question, context=context)

        # Generate the final answer using the LLM
        final_answer = self.llm.invoke(prompt)
        # Get the response from content of AIMessage type
        final_answer = final_answer.content

        return {"computer_use_instructions": final_answer}

    def setup_workflow(self):
        workflow = StateGraph(GraphState)
        # Define the nodes
        workflow.add_node("web_search_node", self.web_search)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("similarity_search", self.similarity_search)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("web_search_retry", self.web_search)
        workflow.add_node("generate", self.generate)
        # Build graph
        workflow.add_edge(START, "web_search_node")
        workflow.add_edge("web_search_node", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "similarity_search": "similarity_search",
            },
        )
        workflow.add_edge("transform_query", "web_search_retry")
        workflow.add_edge("web_search_retry", "grade_documents")
        workflow.add_edge("similarity_search", "generate")
        workflow.add_edge("generate", END)
        # Compile
        self.workflow = workflow.compile()

    def invoke(self, question: str):
        if not self.workflow:
            raise ValueError("Workflow not set up.")

        res = self.workflow.invoke({"question": question})
        return res["computer_use_instructions"]


if __name__ == '__main__':
    agent = AgenticRAG()
    result = agent.invoke("How do I create a bucket?")
    print(result)