
from langgraph.graph import StateGraph, START, END

from backend.src.graph.nodes import shud_retrieve_node, generate_answer_node, retrieve_node, route_after_evaluation, \
check_docs_relevant_node, no_answer_found, generate_answer_with_retrieved_docs_node, router_after_checking_relevant_docs, \
is_grounded_node, router_after_grounded_check, is_useful_node, router_after_usefulness_check, rewrite_question_node, revise_answer_node

from backend.src.graph.state import SelfRAGState


def self_rag_workflow():

    graph = StateGraph(SelfRAGState)

    graph.add_node("shud_retrieve_node", shud_retrieve_node)
    graph.add_node("generate_answer_node", generate_answer_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("check_docs_relevant_node", check_docs_relevant_node)
    graph.add_node("no_answer_found", no_answer_found)
    graph.add_node("generate_answer_with_retrieved_docs_node", generate_answer_with_retrieved_docs_node)
    graph.add_node("is_grounded_node", is_grounded_node)
    graph.add_node("is_useful_node", is_useful_node)
    graph.add_node("rewrite_question_node", rewrite_question_node)
    graph.add_node("revise_answer_node", revise_answer_node)

    graph.add_edge(START, "shud_retrieve_node")
    graph.add_conditional_edges("shud_retrieve_node", route_after_evaluation,
                            {"retrieve_node": "retrieve_node", 
                             "generate_answer_node": "generate_answer_node"})
    graph.add_edge("generate_answer_node", END)
    
    graph.add_edge("retrieve_node", "check_docs_relevant_node")
    graph.add_conditional_edges("check_docs_relevant_node", router_after_checking_relevant_docs, 
                             {"relevant": "generate_answer_with_retrieved_docs_node",
                              "not_relevant": "no_answer_found"})
    
    graph.add_edge("no_answer_found", END)
    graph.add_edge("generate_answer_with_retrieved_docs_node", "is_grounded_node")
    graph.add_conditional_edges("is_grounded_node", router_after_grounded_check,
                             {"grounded": "is_useful_node",
                              "not_grounded":"revise_answer_node"})
    graph.add_edge("revise_answer_node", "is_grounded_node")
    graph.add_conditional_edges("is_useful_node", router_after_usefulness_check,
                             {"useful": END,
                              "not_useful": "rewrite_question_node"})
    graph.add_edge("rewrite_question_node", "retrieve_node")

    workflow = graph.compile()

    # Save workflow diagram
    try:
        # Get the graph diagram as PNG using mermaid
        diagram = workflow.get_graph().draw_mermaid_png()
        with open("workflow_diagram.png", "wb") as f:
            f.write(diagram)
        print("Workflow diagram saved as 'workflow_diagram.png'")
    except Exception as e:
        print(f"Could not save diagram: {e}")
        # Alternative: Save as mermaid text
        mermaid_text = workflow.get_graph().draw_mermaid()
        with open("workflow_diagram.md", "w") as f:
            f.write(f"```mermaid\n{mermaid_text}\n```")
        print("Workflow diagram saved as 'workflow_diagram.md' (mermaid format)")

    return workflow

app = self_rag_workflow()



