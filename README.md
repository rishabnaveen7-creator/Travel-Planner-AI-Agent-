# Voyager — AI Travel Planning Agent

A small full-stack demo: a Python (FastAPI) backend that runs an agent loop
against the Claude API with travel tools (flights, hotels, weather,
activities), and a plain HTML/JS chat frontend.

```
travel-planner/
├── backend/
│   ├── main.py            FastAPI app + agent loop + tools
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html          Chat UI (no build step needed)
└── README.md
```

## How it works

1. The frontend posts `{ message, history }` to `POST /api/chat`.
2. The backend sends the conversation to Claude along with 4 tool
   definitions: `search_flights`, `search_hotels`, `get_weather`,
   `search_activities`.
3. When Claude calls a tool, the backend runs the matching Python function
   and returns the result to the model, looping until Claude produces a
   final text answer.
4. The full message history (including tool calls/results) is passed back
   to the frontend and resent on the next turn, so the agent has full
   context.

The tool functions currently return **realistic mock data** (seeded
randomness, so results are stable within a conversation) so you can run the
whole thing with nothing but an API key. See "Going live" below to wire up
real providers.

## Requirements

- **Python 3.14** (verify with `python3.14 --version` or `python --version`)
- An Anthropic API key from [console.anthropic.com](https://console.anthropic.com)

## Setup

### 1. Backend

```bash
cd backend
python3.14 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env and add your ANTHROPIC_API_KEY
uvicorn main:app --reload --port 8000
```

> **Note on 3.14:** `requirements.txt` uses `>=` version floors rather than
> exact pins, because some of these packages (notably `pydantic-core`, which
> is Rust-backed) only shipped prebuilt wheels for Python 3.14 in their more
> recent releases. If `pip install` ever tries to compile something from
> source, upgrade pip first (`pip install --upgrade pip`) so it can find the
> newer wheels.

The API is now at `http://localhost:8000`. Check `http://localhost:8000/api/health`.

### 2. Frontend

No build step — just open `frontend/index.html` in a browser, or serve it:

```bash
cd frontend
python -m http.server 5500
```

Then visit `http://localhost:5500`. The page calls the backend at
`http://localhost:8000/api/chat` (edit the `API_URL` constant at the top of
the `<script>` block in `index.html` if you deploy the backend elsewhere).

## Going live: replacing the mock tools

Each tool in `backend/main.py` is a plain Python function with a matching
JSON schema in `TOOLS`. To use real data, swap the function body for a call
to a real provider and keep the return shape roughly the same:

| Tool               | Suggested real provider                      |
|---------------------|-----------------------------------------------|
| `search_flights`    | Amadeus Self-Service API, Skyscanner, Duffel   |
| `search_hotels`     | Amadeus, Booking.com Affiliate API, Expedia    |
| `get_weather`       | OpenWeatherMap, Open-Meteo (free, no key)      |
| `search_activities` | Google Places API, Viator, TripAdvisor Content |

Since the agent loop only cares about the tool's name, JSON schema, and
return value, none of the surrounding orchestration code needs to change.

## Notes

- CORS is wide open (`allow_origins=["*"]`) for local development — lock
  this down before deploying publicly.
- The agent loop caps at 6 model calls per user turn as a safety limit
  against runaway tool-use chains.
- `ANTHROPIC_MODEL` in `.env` defaults to `claude-sonnet-5`; override it if
  you want a different model.
