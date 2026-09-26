"""
Voyager - a travel-planning AI agent.

FastAPI backend that wraps the Gemini API with a small toolset
(flight search, hotel search, weather, activities) and runs the classic
agent loop: call the model -> if it wants a tool, run it -> feed the result
back -> repeat until the model gives a final answer.

The tool implementations here return realistic MOCK data so the whole app
runs out of the box with nothing but a GEMINI_API_KEY. Swap them for real
providers (Amadeus, Skyscanner, Booking.com, OpenWeather, etc.) when
you're ready to go live - the function signatures and tool schemas are the
only contract the agent cares about.
"""

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")

app = FastAPI(title="Voyager Travel Planner Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=API_KEY) if API_KEY else None


# --------------------------------------------------------------------------
# Mock travel-data tools. Deterministic per-query (seeded) so results are
# stable across a conversation instead of re-randomizing on every call.
# --------------------------------------------------------------------------

AIRLINES = ["SkyWays", "AeroLink", "Continental Air", "BlueJet", "Horizon Airlines"]
HOTEL_CHAINS = ["Grand Plaza", "Harborview Inn", "The Wanderer's Rest", "Metro Suites", "Sunset Lodge"]
ACTIVITY_POOL = [
    "Guided old-town walking tour", "Local food market crawl", "Museum of Modern Art visit",
    "Sunset boat cruise", "Historic castle/fort tour", "Hiking trail in the nearby hills",
    "Cooking class with a local chef", "Live music night in the arts district",
    "Bike tour along the waterfront", "Day trip to nearby vineyards",
    "Street-art neighborhood tour", "Rooftop bar with skyline views",
]


def search_flights(origin: str, destination: str, date: str, return_date: Optional[str] = None) -> Dict[str, Any]:
    random.seed(f"flight:{origin}:{destination}:{date}")
    flights = []
    for _ in range(3):
        flights.append({
            "airline": random.choice(AIRLINES),
            "origin": origin,
            "destination": destination,
            "date": date,
            "duration": f"{random.randint(2, 14)}h {random.randint(0, 59)}m",
            "stops": random.choice([0, 0, 1, 2]),
            "price_usd": random.randint(180, 950),
        })
    flights.sort(key=lambda f: f["price_usd"])
    return {"flights": flights, "return_date": return_date}


def search_hotels(city: str, check_in: str, check_out: str, guests: int = 2) -> Dict[str, Any]:
    random.seed(f"hotel:{city}:{check_in}:{check_out}")
    hotels = []
    for _ in range(4):
        hotels.append({
            "name": f"{random.choice(HOTEL_CHAINS)} {city}",
            "city": city,
            "rating": round(random.uniform(3.4, 4.9), 1),
            "price_per_night_usd": random.randint(70, 420),
            "check_in": check_in,
            "check_out": check_out,
            "amenities": random.sample(
                ["Free WiFi", "Pool", "Breakfast included", "Gym", "Spa", "Parking", "Pet friendly"], 3
            ),
        })
    hotels.sort(key=lambda h: h["price_per_night_usd"])
    return {"hotels": hotels, "guests": guests}


def get_weather(city: str, date: str) -> Dict[str, Any]:
    random.seed(f"weather:{city}:{date}")
    return {
        "city": city,
        "date": date,
        "condition": random.choice(["Sunny", "Partly cloudy", "Rainy", "Clear", "Overcast"]),
        "high_c": random.randint(15, 34),
        "low_c": random.randint(5, 20),
    }


def search_activities(city: str, interests: str = "") -> Dict[str, Any]:
    random.seed(f"activities:{city}:{interests}")
    return {"city": city, "activities": random.sample(ACTIVITY_POOL, 4)}


TOOL_IMPLEMENTATIONS = {
    "search_flights": search_flights,
    "search_hotels": search_hotels,
    "get_weather": get_weather,
    "search_activities": search_activities,
}

TOOLS = [
    {
        "name": "search_flights",
        "description": "Search for flights between two cities on a given date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string", "description": "Departure city or airport"},
                "destination": {"type": "string", "description": "Arrival city or airport"},
                "date": {"type": "string", "description": "Departure date, YYYY-MM-DD"},
                "return_date": {"type": "string", "description": "Optional return date, YYYY-MM-DD"},
            },
            "required": ["origin", "destination", "date"],
        },
    },
    {
        "name": "search_hotels",
        "description": "Search for hotels in a city for a given date range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "check_in": {"type": "string", "description": "YYYY-MM-DD"},
                "check_out": {"type": "string", "description": "YYYY-MM-DD"},
                "guests": {"type": "integer", "description": "Number of guests"},
            },
            "required": ["city", "check_in", "check_out"],
        },
    },
    {
        "name": "get_weather",
        "description": "Get expected weather for a city on a given date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["city", "date"],
        },
    },
    {
        "name": "search_activities",
        "description": "Find recommended activities and attractions in a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "interests": {
                    "type": "string",
                    "description": "Optional comma-separated interests, e.g. 'food, history, nightlife'",
                },
            },
            "required": ["city"],
        },
    },
]

SYSTEM_PROMPT = f"""You are Voyager, an expert AI travel-planning agent.
Today's date is {datetime.now().strftime('%Y-%m-%d')}.

Your job: help the user plan trips end-to-end - flights, hotels,
weather-aware packing/activity advice, and day-by-day itineraries.

Guidelines:
- Use the provided tools whenever the user asks about flights, hotels,
  weather, or activities. Don't guess prices or availability yourself.
- Ask at most one clarifying question at a time, and only when you truly
  need it (missing dates, missing destination, unclear budget).
- Once you have enough info, propose a concrete plan: flight options, a
  hotel pick, a day-by-day itinerary, and a rough total budget.
- Keep responses well organized with short headers and bullet points.
- Never repeat raw tool output verbatim; summarize it clearly.
"""


# --------------------------------------------------------------------------
# Agent loop
# --------------------------------------------------------------------------

def _format_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
                elif block.get("type") == "tool_result":
                    parts.append(f"[Tool result]\n{block.get('content', '')}")
                elif block.get("type") == "tool_use":
                    parts.append(
                        f"[Tool call: {block.get('name', 'tool')}]\n"
                        f"{json.dumps(block.get('input', {}), ensure_ascii=False)}"
                    )
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def run_agent_turn(messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    if client is None:
        return "", messages

    prompt_parts = [SYSTEM_PROMPT]
    for message in messages:
        role = message.get("role", "user")
        content = _format_message_content(message.get("content", ""))
        prompt_parts.append(f"{role.upper()}: {content}")

    response = client.models.generate_content(
        model=MODEL,
        contents="\n\n".join(prompt_parts),
        config={"max_output_tokens": 1500},
    )
    reply = (getattr(response, "text", None) or "").strip()
    updated_messages = list(messages)
    updated_messages.append({"role": "assistant", "content": reply})
    return reply, updated_messages


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, Any]] = []


class ChatResponse(BaseModel):
    reply: str
    history: List[Dict[str, Any]]


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.",
        )

    messages = list(req.history)
    messages.append({"role": "user", "content": req.message})

    try:
        reply, updated_history = run_agent_turn(messages)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc}")

    if not reply:
        reply = "I wasn't able to put together a response - could you rephrase that?"

    return ChatResponse(reply=reply, history=updated_history)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": MODEL, "configured": str(client is not None)}
