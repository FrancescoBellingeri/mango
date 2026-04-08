# Server API

Mango exposes a FastAPI server with a streaming SSE endpoint and a session-based multi-turn conversation system.

---

## Starting the Server

```python
from mango.servers.fastapi import MangoFastAPIServer

server = MangoFastAPIServer(agent)
server.run()  # http://localhost:8000
```

Default: `host=0.0.0.0`, `port=8000`. Override:

```python
server.run(host="127.0.0.1", port=9000)
```

---

## Endpoints

### `POST /api/v1/ask/stream`

Ask a question and receive a real-time SSE stream.

**Request body:**

```json
{
  "question": "How many orders were placed last week?",
  "session_id": "optional-session-id-for-multi-turn"
}
```

**Response** — `text/event-stream`, one JSON object per `data:` line:

| Event type | Fields | Description |
|------------|--------|-------------|
| `session` | `session_id` | Sent first. Use this ID for follow-up questions. |
| `tool_call` | `tool_name`, `tool_args` | The agent is calling a tool. |
| `tool_result` | `tool_name`, `success`, `preview` | Result of a tool call. |
| `answer` | `text` | The final natural language answer (may be streamed in chunks). |
| `done` | `iterations`, `input_tokens`, `output_tokens` | Summary of the interaction. |
| `error` | `message` | Something went wrong. |

**Example stream:**

```
data: {"type": "session",     "session_id": "abc123"}
data: {"type": "tool_call",   "tool_name": "list_collections", "tool_args": {}}
data: {"type": "tool_result", "tool_name": "list_collections", "success": true, "preview": "orders, customers, products..."}
data: {"type": "tool_call",   "tool_name": "run_mql", "tool_args": {"operation": "aggregate", "collection": "orders", ...}}
data: {"type": "tool_result", "tool_name": "run_mql", "success": true, "preview": "[{\"total\": 1247}]"}
data: {"type": "answer",      "text": "1,247 orders were placed in the last 7 days."}
data: {"type": "done",        "iterations": 2, "input_tokens": 1820, "output_tokens": 94}
```

---

## Multi-turn Conversations

Pass the `session_id` from the first response to continue a conversation:

```bash
# First question
curl -X POST http://localhost:8000/api/v1/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How many orders were placed last week?"}'

# Follow-up (use the session_id from the first response)
curl -X POST http://localhost:8000/api/v1/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "And how many were delivered?", "session_id": "abc123"}'
```

Mango maintains the conversation history for each session and prunes it automatically to keep token usage stable.

---

## JavaScript / Frontend

```js
const response = await fetch("http://localhost:8000/api/v1/ask/stream", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question: "How many users signed up this month?" }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  const lines = decoder.decode(value).split("\n");
  for (const line of lines) {
    if (!line.startsWith("data: ")) continue;
    const event = JSON.parse(line.slice(6));

    if (event.type === "answer") console.log(event.text);
    if (event.type === "tool_call") console.log("Using tool:", event.tool_name);
    if (event.type === "done") console.log("Tokens used:", event.input_tokens + event.output_tokens);
  }
}
```

---

## Health Check

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```
