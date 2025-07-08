"""
RAG Pipeline Module for Financial Complaint Analysis

This module implements a modular Retrieval-Augmented Generation (RAG) pipeline:
- Retriever: retrieves top-k relevant complaint chunks using FAISS and the embedding model
- Prompt Engineering: robust prompt template for LLM guidance
- Generator: combines context and question, sends to LLM, returns answer
- Evaluation: utility for qualitative evaluation of the pipeline

Dependencies: sentence-transformers, faiss, transformers (Hugging Face), torch
"""
import sys; sys.path.append('src')




import logging
from typing import List, Dict, Any, Optional
from embedding_indexing import ComplaintEmbeddingIndexer, create_embedding_indexer

try:
    from transformers import pipeline, Pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Modular Retrieval-Augmented Generation (RAG) pipeline for financial complaint analysis.
    """
    def __init__(self,
                 indexer: Optional[ComplaintEmbeddingIndexer] = None,
                 llm_model: str = "google/flan-t5-base",
                 device: int = -1,
                 top_k: int = 5):
        """
        Initialize the RAG pipeline.

        Args:
            indexer (ComplaintEmbeddingIndexer, optional): Pre-initialized embedding indexer. If None, a new one is created.
            llm_model (str): Hugging Face model name for generation.
            device (int): Device for LLM (-1 for CPU, 0+ for GPU).
            top_k (int): Number of chunks to retrieve.
        """
        self.indexer = indexer or create_embedding_indexer()
        self.top_k = top_k
        self.llm_model = llm_model
        self.device = device
        self.prompt_template = (
            "You are a financial analyst assistant for CrediTrust. "
            "Your task is to answer questions about customer complaints. "
            "Use the following retrieved complaint excerpts to formulate your answer. "
            "If the context doesn't contain the answer, state that you don't have enough information.\n"
            "Context: {context}\nQuestion: {question}\nAnswer:"
        )
        if TRANSFORMERS_AVAILABLE:
            try:
                self.generator = pipeline("text2text-generation", model=llm_model, device=device)
            except Exception as e:
                logger.error(f"Error loading LLM pipeline: {e}")
                self.generator = None
        else:
            logger.error("transformers library not available. LLM generation will not work.")
            self.generator = None

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant complaint chunks for a user question.

        Args:
            question (str): User's question.
        Returns:
            List[dict]: List of retrieved chunk dicts (id, text, metadata, distance).
        """
        if not question or not isinstance(question, str):
            logger.error("Question must be a non-empty string.")
            return []
        try:
            results = self.indexer.search_similar_chunks(question, n_results=self.top_k)
            return results
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []

    def build_prompt(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Build the prompt for the LLM using the template, context, and question.
        Args:
            question (str): User's question.
            retrieved_chunks (List[dict]): Retrieved chunk dicts.
        Returns:
            str: Prompt string for the LLM.
        """
        context = "\n---\n".join([chunk['text'] for chunk in retrieved_chunks])
        prompt = self.prompt_template.format(context=context, question=question)
        return prompt

    def generate(self, prompt: str) -> str:
        """
        Generate an answer from the LLM given the prompt.
        Args:
            prompt (str): Prompt string for the LLM.
        Returns:
            str: Generated answer.
        """
        if not self.generator:
            logger.error("LLM generator pipeline is not available.")
            return "[LLM not available]"
        try:
            output = self.generator(prompt, max_length=256, do_sample=False)
            if isinstance(output, list) and output:
                return output[0].get('generated_text', output[0].get('text', '')).strip()
            return str(output)
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return "[Generation error]"

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve, build prompt, generate answer.
        Args:
            question (str): User's question.
        Returns:
            dict: { 'question', 'answer', 'retrieved_sources' }
        """
        retrieved = self.retrieve(question)
        prompt = self.build_prompt(question, retrieved)
        answer = self.generate(prompt)
        return {
            'question': question,
            'answer': answer,
            'retrieved_sources': retrieved[:2]  # Show top 2 for reporting
        }

    @staticmethod
    def qualitative_evaluation(rag_pipeline, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Run qualitative evaluation for a list of questions.
        Args:
            rag_pipeline (RAGPipeline): The RAG pipeline instance.
            questions (List[str]): List of questions to evaluate.
        Returns:
            List[dict]: List of evaluation results with question, answer, sources.
        """
        results = []
        for q in questions:
            try:
                result = rag_pipeline.answer_question(q)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating question '{q}': {e}")
                results.append({'question': q, 'answer': '[Error]', 'retrieved_sources': []})
        return results

    @staticmethod
    def evaluation_table(results: List[Dict[str, Any]]) -> str:
        """
        Create a Markdown evaluation table for reporting.
        Args:
            results (List[dict]): List of evaluation results.
        Returns:
            str: Markdown table as a string.
        """
        header = "| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |\n"
        header += "|---|---|---|---|---|\n"
        rows = []
        for r in results:
            sources = "<br>".join([s['text'][:120].replace("|", " ") + ("..." if len(s['text']) > 120 else "") for s in r.get('retrieved_sources', [])])
            rows.append(f"| {r['question']} | {r['answer']} | {sources} |  |  |")
        return header + "\n".join(rows) 