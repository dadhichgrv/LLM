from backend.src.graph.workflow import app as self_rag_app

def run_self_rag_workflow():
 
    initial_state = {#"query": "What are the specifications of Youtube Ad Video?",
                     "query": "How do video ad and compliance rules interact in borderline scenarios?",
                     #"query": "What is the capital of Italy?",
                     #"query": "Using only the retrieved context, return a JSON object with keys: deal_value_usd, announcement_date, board_approval_id, and direct_quotes (exactly 3 verbatim quotes with source line references). If any field is missing in context, do not guess",
                     "answer": "",
                     "retrieve_or_not": "",
                     "retrieved_docs": [],
                     "relevant_docs": [],
                     "final_scores": [],
                     "final_reasons": [],
                     "context": "",
                     "original_query": "",
                     "relevant_or_not": "",
                     "is_response_grounded": "",
                     "is_useful": "",
                     "retrieve_counter": 0,
                     "generate_answer_counter": 0,
                     "revise_counter": 0
                     }
    
    result = self_rag_app.invoke(initial_state)

    print("query: \n", initial_state["query"])
    print("answer: \n", result["answer"])
    print("retrieve_or_not: \n", result["retrieve_or_not"])
    print("retrieved_docs: \n", [doc.page_content for doc in result["retrieved_docs"]])
    print("relevant_docs: \n", [doc.page_content for doc in result["relevant_docs"]])
    print("final_scores: \n", result["final_scores"])
    print("final_reasons: \n", result["final_reasons"])
    print("context: \n", result["context"])
    print("is_response_grounded: \n", result["is_response_grounded"])
    print("is_useful: \n", result["is_useful"])
    print("original_query: \n", result["original_query"])
    print("relevant_or_not: \n", result["relevant_or_not"])
    print("Retrieve Counter: \n", result["retrieve_counter"])
    print("Generate Answer Counter: \n", result["generate_answer_counter"])
    print("Revise Counter: \n", result["revise_counter"])


if __name__ == "__main__":
    run_self_rag_workflow()    