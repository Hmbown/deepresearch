import importlib


def _load_graph_module():
    graph = importlib.import_module("deepresearch.graph")
    return importlib.reload(graph)


def test_native_subgraph_builders_compile(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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


def test_researcher_subgraph_uses_deep_agent(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from deepresearch.researcher_subgraph import build_researcher_subgraph

    agent = build_researcher_subgraph()
    assert hasattr(agent, "ainvoke")
