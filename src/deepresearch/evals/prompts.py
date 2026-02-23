"""LLM-as-judge prompt templates for online evaluation."""

ANSWER_QUALITY_PROMPT = """\
You are an expert evaluator for deep research reports. Score the following research report \
on five dimensions, each on a 1-5 scale.

## User Query
{inputs}

## Research Report
{outputs}

## Scoring Dimensions

1. **Citation quality** (weight 0.25): Are claims backed by inline citations like [1], [2] \
with a Sources/References section that maps numbers to URLs? \
Score 1 if no citations at all, 5 if every major claim cites a specific source with URL.

2. **Completeness** (weight 0.25): Does the report address the full scope of the user's \
question? Score 1 if it only partially answers, 5 if it thoroughly covers all aspects.

3. **Structure** (weight 0.15): Does the report have clear organization — executive summary, \
key findings, evidence sections, and sources? Score 1 if unstructured wall of text, \
5 if well-organized with clear sections and hierarchy.

4. **Evidence specificity** (weight 0.20): Does the report include concrete facts \
(names, dates, numbers, statistics) rather than vague claims? Score 1 if entirely vague, \
5 if packed with specific, verifiable details.

5. **Uncertainty handling** (weight 0.15): Does the report flag weak evidence, contradictions, \
or knowledge gaps honestly? Score 1 if it states everything with false certainty, \
5 if it transparently qualifies uncertain claims.

## Instructions
Score each dimension 1-5. Compute the weighted average and normalize to a 0-1 scale \
using: score = (weighted_avg - 1) / 4.

Respond with your reasoning, then a final score between 0 and 1."""

PROCESS_QUALITY_PROMPT = """\
You are an expert evaluator for research agent execution quality. Score the following \
research process summary on four dimensions, each on a 1-5 scale.

## Research Process Summary
{outputs}

The summary includes both observed behavior and configured budgets, so score should be
contextual to those constraints.

## Scoring Dimensions

1. **Source diversity** (weight 0.30): How many distinct domains/sources were consulted, relative to the
configured run budget? \
Score 1 if only 1-2 domains and no clear breadth planning, 3 if 5-8 domains across available search calls, \
5 if 10+ domains and broad evidence coverage. \
If the run is clearly capped by budget (low search or tool budget), do not penalize for not reaching high
absolute coverage; judge breadth against what was possible.

2. **Search strategy** (weight 0.25): Was the think_tool used between searches for \
reflection and iterative refinement? Were queries diverse and progressively focused? \
Score 1 if random queries with no reflection, 5 if systematic with disciplined think_tool usage and bounded iteration.

3. **Tool efficiency** (weight 0.20): Were search calls within budget? Was fetch_url \
used only when snippets were insufficient? Score 1 if excessive redundant tool calls for the budgeted windows, \
5 if the run is tight and efficient within MAX_REACT_TOOL_CALLS, search budgets, and iteration caps.

4. **Coverage** (weight 0.25): Were multiple research units dispatched for complex \
multi-faceted topics? Did different researchers cover different angles? Score 1 if only \
one narrow research path, 5 if coverage is comprehensive given MAX_CONCURRENT_RESEARCH_UNITS and MAX_RESEARCHER_ITERATIONS.

## Instructions
Score each dimension 1-5. Compute the weighted average and normalize to a 0-1 scale \
using: score = (weighted_avg - 1) / 4.

Respond with your reasoning, then a final score between 0 and 1."""
