# Mango — Web Frontend

The Nuxt 4 web UI for Mango. Connects to the Mango FastAPI backend and lets you query MongoDB in natural language from a browser.

## Prerequisites

The Python backend must be running before starting the frontend. See the [installation guide](https://mango.francescobellingeri.com/docs/getting-started/installation) to set it up:

```bash
pip install mango-ai[anthropic]  # or openai / gemini
```

Then start the FastAPI server (default: `http://localhost:8000`):

```python
from mango.servers.fastapi import MangoFastAPIServer
MangoFastAPIServer(agent).run()
```

## Setup

```bash
npm install
```

## Development

```bash
npm run dev   # http://localhost:3000
```

## Production

```bash
npm run build
npm run preview
```
