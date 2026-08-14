# AI Agents and Tool Use

An AI agent is a language model wrapped in a loop that lets it take actions in the
world through tools, rather than only producing text. The key idea is that the
model decides *what* to do, and ordinary code *executes* that decision.

## The route-act-answer pattern
A simple and reliable agent design has three steps:

1. **Route** - the model reads the user's request and chooses one tool plus its
   arguments. Returning this choice as structured JSON makes it easy to parse and
   works across different model providers.
2. **Act** - the application runs the chosen tool. The model never executes code
   itself; this keeps the system safe and debuggable.
3. **Answer** - the tool's output is fed back to the model, which writes the final
   grounded response.

## Common tools
- A retrieval/search tool for looking things up in a knowledge base (RAG).
- A calculator for arithmetic, since language models are unreliable at math.
- A web search tool for current events the model was not trained on.

## Routers vs. multi-agent systems
A router agent picks one of several tools. More advanced systems use multiple
specialized agents (for example a researcher and a writer) that hand work to each
other. Start with a single router; add agents only when one model in a loop
genuinely cannot do the job.
