# Design

## Preparation

First, I reverse engineered Claude's current Researcher mode by making multiple different queries and watching thinking, planning, and spawning sub-agents. I've found from my own usage that it covers the most ground in deep research and still comes up with a coherent answer.

I noticed that it went through breadth vs depth ideas and after creating a research brief sent off individual tasks to sub-agents specialized in that area.

I started with no deliberate scope / user intake process and quickly ran into issues with that.

After getting an initial prototype working through vibe coding, I watched the LangChain deep research classes (LangGraph Academy) and refactored against the formal vocabulary and patterns. This mapped my intuitive architecture onto three canonical phases: scope classification → research → report creation.

## Scope / Intake

I underestimated the importance of this quite a bit and it quickly became the most important part of the project for me. Early iterations with no intake / scope resulted in worthless responses where the response spent most of the time indicating what it wish it had had clarified.

Moved to a not-specific-enough "ask clarifying question" prompt — this led to loops where the model would keep asking individual questions. Finally landed on specific up-to-three questions and only-one-clarifying-question limits that are working well now.

The scoping phase is intentionally front-loaded — I found that putting more effort here was more valuable than adding extra research rounds.

## Research Supervisor

Issues I ran into include context limits, tool usage bleeding over into delivery, generating report before all agents completed tasks, and lack of clarity on when (if at all) to send out new agents to get more information.

Originally the orchestrator could decide to end early while a wave of research was still outgoing — a "soft" termination. I switched to hard decision nodes: all sub-agents must return before the orchestrator makes a new decision. This was cleaner, more deliberate, and better aligned with the deep research mindset.

I started off with much more rigid constraints to try to avoid context bloating, but it ended up resulting in poor response quality. After comparing outputs against ChatGPT 5.2 (GPT subscription) with the same query, it became obvious I was limiting the agent's capabilities with overly strict tool-call and search limits. The more artificial constraints I put on the agent, the worse the output tended to be. Landed on a happy medium of a maximum number of sub agents to be called in parallel + number in total while leaving other constraints open.

## Researchers

Number of researchers went through a few iterations — tried everything from tying it to "type" of query (too restrictive and not enough information, I found) determining number of sub-agents to doing tiered multiple waves — 5 - 3 - 1.

In the end, leaving the models to do what the models do best has worked the best for me. 4 sub-agents in parallel is working as a happy medium where in some instances one wave can satisfy conditions but in others it leaves opportunity for multiple iterations where quality of sub-agent prompting can improve (hopefully) over time.

## Agent vs. Deep Agent

While building the research supervisor subgraph and adding tool calls manually, I realized that the current deep agent package takes care of a lot of this. I kept the beginning intake the same to keep it from trying to perform searches / tool calls early (had issues with that).

I tried giving every agent `create_deep_agent` abilities, but this caused infinite looping — deep agents can spawn sub-agents, which can spawn sub-agents, etc. After going back and forth, I ended up making the researcher node a `create_deep_agent` call. The researcher is the node that benefits most from independence and deeper reasoning — it's the one that actually needs to think through search results, evaluate relevance, and decide how to dig further. The supervisor stays as a hand-built LangGraph loop that dispatches research tool calls.

`create_deep_agent()` returns a compiled LangGraph graph with middleware baked in: a planning/todo tool, a virtual filesystem for context management, and sub-agent spawning via the `task` tool.

## Things That Didn't Work

- **More model subdivisions** — tried creating finer-grained model breakdowns, but too many variables made it hard to understand why results were different.
- **Minimax model** — has a weird "thinking label" issue. Worked with it initially, but switched to OpenAI models which eliminated that problem.
- **Accidental parallel routing bug** — sequential sub-agent plans were accidentally flowing from both the initial sub-agents and the orchestrator to the next sub-agent. The orchestrator could inadvertently decide to finish early even though it had just requested a follow-up sub-agent.

## Final Report

I debated whether to add specifics on output style related to the type of deep research request. Like most other things, I found that when I got out of the model's way and gave it just enough information it was successful. Initial attempts included ideas such as limiting number of topics, number of bullet points, etc., ended up just being extra chatter and got in the way of good outputs.

## Web Search

Tested both Tavily and Exa — Exa had great latency and Tavily a bit more detail from my experience. Landing on OpenAI web search with the new Responses API was more of a cost thing for myself, and it lets me take advantage of what the model can already do.

## Model Choice

OpenAI GPT 5.2 by default; considered using GPT 5-mini for sub-agents but the research tasks were too intensive to get high quality output.

## Evals

Evals are set up using an LLM-as-judge framework but were not useful in the early phase — I still needed manual tuning and to roll back early restrictions to get the agent to the quality level I wanted before evals could tell me anything meaningful.

## Config / Env

Configurable via env. LangSmith tracing, LangSmith API, OpenAI API, Responses API configuration, Exa/Tavily search API, max number of researchers, max number of waves.

## CLI

Included a lightweight CLI.

## Report Quality — CellCog Comparison

I ran the same airport socioeconomic impact query through both this system and CellCog and compared them side by side.

Our report came out more academic — hypothesis framing, evidence for vs against, peer-reviewed econometric studies, careful about gross vs net. The citations held up when I spot-checked them which was nice. CellCog's came out more like a consulting deliverable — parametric model with sensitivity analysis, leakage-adjusted tables, significance benchmarking by region size, closure counterfactuals. More actionable if you're a policymaker.

Both landed on basically the same conclusion (measurable but context-dependent) but got there differently. Looking at how CellCog structures their answers made me think it could be worth making our reports more structured / professional — things like quantitative modeling sections, graduated significance tables, explicit policy implications. That's more of a report prompt engineering question than an architecture one though.

## If I Had More Time...

1. **Context engineering** — I am interested in an Exa-deep-research-like approach where context is maintained across all agents deliberately.

2. **Sandboxing / RLM-like behavior** — Related to above, I am interested in if utilizing a sandbox and encouraging recursive REPL behavior + storing data with symbols to pull later — it seems like this could be a great breakthrough in deep research efficiency and reliability with source citing.

3. I wanted to tune the intake process a bit more and allow for an easier possibility of inferring information from the first answers and not feeling like the research report had to be presented. I.e. — maybe if it's obvious that the user is ready for the agent to perform research from their response to the questions, they would continue onward with making the research brief and assigning tasks automatically.
