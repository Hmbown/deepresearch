# Design

## Preparation

First, I reverse engineered Claude's current Researcher mode by making multiple different queries and watching thinking, planning, and spawning sub-agents. I've found from my own usage that it covers the most ground in deep research and still comes up with a coherent answer.

I noticed that it went through breadth vs depth ideas and after creating a research brief sent off individual tasks to sub-agents specialized in that area.

I started with no deliberate scope / user intake process and quickly ran into issues with that.

## Scope / Intake

I underestimated the importance of this quite a bit and it quickly became the most important part of the project for me. Early iterations with no intake / scope resulted in worthless responses where the response spent most of the time indicating what it wish it had had clarified.

Moved to a not-specific-enough "ask clarifying question" prompt — this led to loops where the model would keep asking individual questions. Finally landed on specific up-to-three questions and only-one-clarifying-question limits that are working well now.

## Research Supervisor

Issues I ran into include context limits, tool usage bleeding over into delivery, generating report before all agents completed tasks, and lack of clarity on when (if at all) to send out new agents to get more information.

I started off with much more rigid constraints to try to avoid context bloating, but it ended up resulting in poor response quality. Landed on a happy medium of a maximum number of sub agents to be called in parallel + number in total while leaving other constraints open.

## Researchers

Number of researchers went through a few iterations — tried everything from tying it to "type" of query (too restrictive and not enough information, I found) determining number of sub-agents to doing tiered multiple waves — 5 - 3 - 1.

In the end, leaving the models to do what the models do best has worked the best for me. 4 sub-agents in parallel is working as a happy medium where in some instances one wave can satisfy conditions but in others it leaves opportunity for multiple iterations where quality of sub-agent prompting can improve (hopefully) over time.

## Final Report

I debated whether to add specifics on output style related to the type of deep research request. Like most other things, I found that when I got out of the model's way and gave it just enough information it was successful. Initial attempts included ideas such as limiting number of topics, number of bullet points, etc., ended up just being extra chatter and got in the way of good outputs.

## Agent vs. Deep Agent

While building the research supervisor subgraph and adding tool calls manually, I realized that the current deep agent package takes care of a lot of this. I kept the beginning intake the same to keep it from trying to perform searches / tool calls early (had issues with that).

## Web Search

Exa + Tavily are both configured — OpenAI Responses Web Search on by default when GPT 5.2 configured.

## Model Choice

OpenAI GPT 5.2 by default; considered using GPT 5-mini for sub-agents but the research tasks were too intensive to get high quality output.

## Config / Env

Configurable via env. LangSmith tracing, LangSmith API, OpenAI API, Responses API configuration, Exa/Tavily search API, max number of researchers, max number of waves.

## CLI

Included a lightweight CLI.

## If I Had More Time...

1. **Context engineering** — I am interested in an Exa-deep-research-like approach where context is maintained across all agents deliberately.

2. **Sandboxing / RLM-like behavior** — Related to above, I am interested in if utilizing a sandbox and encouraging recursive REPL behavior + storing data with symbols to pull later — it seems like this could be a great breakthrough in deep research efficiency and reliability with source citing.

3. I wanted to tune the intake process a bit more and allow for an easier possibility of inferring information from the first answers and not feeling like the research report had to be presented. I.e. — maybe if it's obvious that the user is ready for the agent to perform research from their response to the questions, they would continue onward with making the research brief and assigning tasks automatically.
