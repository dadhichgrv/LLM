from langgraph.graph import StateGraph, START, END
from backend.src.graph.nodes import is_retrieve_node, retrieve_node, router_after_is_retrieve_node, web_search_node, \
     is_doc_relevant_node, rewrite_query, refine_node, generate_answer_node, route_after_is_doc_relevant_node
from backend.src.graph.states import CRAGState

def CRAGWorkflow():
    
    graph = StateGraph(CRAGState)

    graph.add_node(is_retrieve_node, "is_retrieve_node")
    graph.add_node(retrieve_node, "retrieve_node")
    graph.add_node(web_search_node, "web_search_node")
    graph.add_node(is_doc_relevant_node, "is_doc_relevant_node")
    graph.add_node(rewrite_query, "rewrite_query")
    graph.add_node(refine_node, "refine_node")
    graph.add_node(generate_answer_node, "generate_answer_node")

    graph.add_edge(START, "is_retrieve_node")
    graph.add_conditional_edges("is_retrieve_node", router_after_is_retrieve_node,
                               {"retrieve": "retrieve_node",
                                "not_retrieve":"rewrite_query"})
    
    graph.add_edge("web_search_node", END)
    graph.add_edge("retrieve_node", "is_doc_relevant_node")
    graph.add_conditional_edges("is_doc_relevant_node", route_after_is_doc_relevant_node,
                                {"refine_node": "refine_node",
                                 "rewrite_query": "rewrite_query"})
    
    graph.add_edge("refine_node", "generate_answer_node")
    
    graph.add_edge("rewrite_query", "web_search_node")
    graph.add_edge("web_search_node", "refine_node")

    graph.add_edge("generate_answer_node", END)

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


app = CRAGWorkflow()



