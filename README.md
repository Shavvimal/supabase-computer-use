# Supabase Computer Use

1. User will ask a Question
2. Agentic RAG for Expansion will:
   - Take in Query
   - See if RAG is required
   - Fetch relevant Documentation using RAG
   - Checks Relevance
   - If not relevant, rewrite the query and retry
   - Generate as detailed instructions for Computer Use to use the GUI

![img.png](assets/img.png)

1. Add Anthropic-defined computer use tools to your API request, as well as the user prompt
2. Claude loads the stored computer use tool definitions and assesses if any tools can help with the user’s query.
   - If yes, Claude constructs a properly formatted tool use request.
   - The API response has a `stop_reason` of `tool_use`,
3. On our side, we extract the tool name and input from Claude’s request.
4. Use tool on computer
5. Claude analyzes the tool results to determine if more tool use is needed or the task has been completed.
   - If Claude decides it needs another tool, it responds with another `tool_use` stop_reason and you should return to step 2
   - Otherwise, it crafts a text response to the user.
6. Continue the conversation with a new user message containing a `tool_result` content block.
