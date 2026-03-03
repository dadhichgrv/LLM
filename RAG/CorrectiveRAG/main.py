from backend.src.graph.workflow import app


def _print_section(title: str):
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def _preview_text(text: str, max_len: int = 400) -> str:
    value = (text or "").strip().replace("\n", " ")
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}..."


def _print_docs(label: str, docs):
    _print_section(label)
    if not docs:
        print("None")
        return
    for idx, doc in enumerate(docs, start=1):
        doc_id = (
            doc.metadata.get("id")
            or doc.metadata.get("chunk_id")
            or doc.metadata.get("source_id")
            or idx
        )
        content = (doc.page_content or "").strip()
        print(f"id: {doc_id}")
        print(f"content: {content}")
        print("-" * 60)

def run_rag_workflow():
    initial_inputs = {#"query": "How do video ad and compliance rules interact in borderline scenarios?",
                      #"query": "Which are the neighbouring countries to Iran",
                      #"query": "Compare specifications for ADs on Youtube vs Facebook",
                      "query": "What are the compliance rules for video ads on Facebook and Nexa AI, and how do they differ in borderline scenarios?",
                      "is_retrieval_needed": False,
                      "retrieved_docs": [],
                      "web_context": "",
                      "relevant_docs": [],
                      "verdict": None,
                      "doc_scores": [],
                      "reason": [],
                      "answer": "",
                      "web_query": "",
                      "refined_context": ""
                      
                      }
    
    result = app.invoke(initial_inputs)

    _print_section("INPUT")
    print(f"Query: {initial_inputs['query']}")

    _print_section("RETRIEVAL")
    print(f"Is Retrieval Needed: {result['is_retrieval_needed']}")
    print(f"Verdict: {result['verdict']}")
    print(f"Document Scores: {result['doc_scores']}")

    _print_section("RELEVANCE REASONS")
    if result["reason"]:
        for idx, reason in enumerate(result["reason"], start=1):
            print(f"[{idx}] {_preview_text(reason, max_len=300)}")
    else:
        print("None")

    _print_docs("RETRIEVED DOCS", result["retrieved_docs"])
    _print_docs("RELEVANT DOCS", result["relevant_docs"])

    _print_section("WEB")
    print(f"Web Query: {result['web_query']}")
    print(f"Web Context: {_preview_text(result['web_context'], max_len=600)}")

    _print_section("REFINED CONTEXT")
    print(_preview_text(result["refined_context"], max_len=1000) or "None")

    _print_section("FINAL ANSWER")
    print(result["answer"])

if __name__ == "__main__":
    run_rag_workflow()
    