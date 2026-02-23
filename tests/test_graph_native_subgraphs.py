import importlib
from pathlib import Path


def _load_graph_module():
    graph = importlib.import_module("deepresearch.graph")
    return importlib.reload(graph)


def test_native_subgraph_builders_compile():
    graph = _load_graph_module()

    researcher_subgraph = graph.build_researcher_subgraph()
    supervisor_subgraph = graph.build_supervisor_subgraph()

    assert hasattr(researcher_subgraph, "ainvoke")
    assert hasattr(supervisor_subgraph, "ainvoke")


def test_main_graph_routes_through_supervisor_and_final_report_nodes():
    graph = _load_graph_module()
    compiled = graph.app.get_graph()

    node_names = set(compiled.nodes.keys())
    assert "research_supervisor" in node_names
    assert "final_report_generation" in node_names
    assert "research_manager_node" not in node_names

    edges = {(edge.source, edge.target) for edge in compiled.edges}
    assert ("write_research_brief", "research_supervisor") in edges
    assert ("research_supervisor", "final_report_generation") in edges
    assert ("final_report_generation", "__end__") in edges


def test_deepagents_removed_from_runtime_and_dependencies():
    graph_source = Path("src/deepresearch/graph.py").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "deepagents" not in graph_source
    assert "deepagents" not in pyproject
