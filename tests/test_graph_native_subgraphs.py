import importlib

from langgraph.checkpoint.memory import MemorySaver


def _load_graph_module():
    graph = importlib.import_module("deepresearch.graph")
    return importlib.reload(graph)


def test_native_subgraph_builders_compile(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    researcher_subgraph = importlib.import_module("deepresearch.researcher_subgraph")
    fake_researcher_graph = type("FakeResearcherGraph", (), {"ainvoke": lambda self, *_args, **_kwargs: {}})()
    monkeypatch.setattr(researcher_subgraph, "create_deep_agent", lambda **kwargs: fake_researcher_graph)
    graph = _load_graph_module()

    researcher_subgraph = graph.build_researcher_subgraph()
    supervisor_subgraph = graph.build_supervisor_subgraph()

    assert hasattr(researcher_subgraph, "ainvoke")
    assert hasattr(supervisor_subgraph, "ainvoke")


def test_main_graph_routes_through_supervisor_and_final_report_nodes():
    graph = _load_graph_module()
    compiled = graph.app.get_graph()

    node_names = set(compiled.nodes.keys())
    assert "route_turn" not in node_names
    assert "scope_intake" in node_names
    assert "write_research_brief" not in node_names
    assert "research_supervisor" in node_names
    assert "final_report_generation" in node_names
    assert "research_manager_node" not in node_names

    edges = {(edge.source, edge.target) for edge in compiled.edges}
    assert ("__start__", "scope_intake") in edges
    assert ("scope_intake", "research_supervisor") in edges
    assert ("research_supervisor", "final_report_generation") in edges
    assert ("final_report_generation", "__end__") in edges


def test_researcher_subgraph_uses_deep_agent(monkeypatch):
    monkeypatch.setenv("SEARCH_PROVIDER", "none")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from deepresearch import researcher_subgraph

    captured = {}
    fake_graph = type("FakeResearcherGraph", (), {"ainvoke": lambda self, *_args, **_kwargs: {}})()

    def _fake_create_deep_agent(**kwargs):
        captured["kwargs"] = kwargs
        return fake_graph

    monkeypatch.setattr(researcher_subgraph, "create_deep_agent", _fake_create_deep_agent)

    agent = researcher_subgraph.build_researcher_subgraph()
    assert hasattr(agent, "ainvoke")
    assert captured["kwargs"]["name"] == "deep-researcher"
    # model should be a pre-configured ChatModel, not a raw string
    assert not isinstance(captured["kwargs"]["model"], str)


def test_build_app_accepts_optional_checkpointer():
    graph = _load_graph_module()
    app = graph.build_app(checkpointer=MemorySaver())
    assert app is not None
