from dataclasses import dataclass


@dataclass(frozen=True)
class CausalEdge:
    parent: str
    child: str


SCM_EDGES = (
    CausalEdge("economic_context", "network_exposure"),
    CausalEdge("network_exposure", "aml_state"),
    CausalEdge("economic_context", "transaction_volume"),
    CausalEdge("aml_state", "transaction_volume"),
    CausalEdge("network_exposure", "cross_border_ratio"),
    CausalEdge("aml_state", "cross_border_ratio"),
    CausalEdge("aml_state", "structuring_score"),
    CausalEdge("structuring_score", "transaction_volume"),
)