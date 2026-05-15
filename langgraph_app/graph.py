"""Budowanie i kompilacja grafu LangGraph dla symulacji rozmowy lekarza."""

from langgraph.graph import END, StateGraph

from .nodes import (
    node_analyze,
    node_build_directives,
    node_detect_context,
    node_ethics_stop,
    node_finalize,
    node_generate_response,
    node_plan_turn,
    node_policy_check,
    node_retrieve_context,
    node_time_stop,
    node_update_state,
    route_after_policy,
    route_after_update,
)
from .state import ConversationState


def build_graph() -> StateGraph:
    """Buduje graf przetwarzania jednej tury rozmowy.

    Przepływ:
        detect_context → analyze → policy_check
            ├─[hard_stop]──→ ethics_stop → END
            └─[continue]──→ update_state
                ├─[time_limit]──→ time_stop → END
                └─[continue]──→ build_directives → plan_turn → retrieve_context → generate_response → finalize → END
    """
    graph = StateGraph(ConversationState)

    graph.add_node("detect_context", node_detect_context)
    graph.add_node("analyze", node_analyze)
    graph.add_node("policy_check", node_policy_check)
    graph.add_node("ethics_stop", node_ethics_stop)
    graph.add_node("update_state", node_update_state)
    graph.add_node("time_stop", node_time_stop)
    graph.add_node("build_directives", node_build_directives)
    graph.add_node("plan_turn", node_plan_turn)
    graph.add_node("retrieve_context", node_retrieve_context)
    graph.add_node("generate_response", node_generate_response)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("detect_context")

    graph.add_edge("detect_context", "analyze")
    graph.add_edge("analyze", "policy_check")

    graph.add_conditional_edges(
        "policy_check",
        route_after_policy,
        {"ethics_stop": "ethics_stop", "update_state": "update_state"},
    )
    graph.add_edge("ethics_stop", END)

    graph.add_conditional_edges(
        "update_state",
        route_after_update,
        {"time_stop": "time_stop", "build_directives": "build_directives"},
    )
    graph.add_edge("time_stop", END)

    graph.add_edge("build_directives", "plan_turn")
    graph.add_edge("plan_turn", "retrieve_context")
    graph.add_edge("retrieve_context", "generate_response")
    graph.add_edge("generate_response", "finalize")
    graph.add_edge("finalize", END)

    return graph


# Skompilowany graf wielokrotnego użytku (bez checkpointera — sesje zarządzane przez service)
compiled_graph = build_graph().compile()
