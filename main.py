"""
LangGraph Multi-Agent Travel Booking System with MCP and Memory
Part 2: Integrates Model Context Protocol (MCP) for real-time Flights, Weather, and Web Search.
"""

import os
from typing import TypedDict, Annotated
import operator
import asyncio
import uuid
import psycopg
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    try:
        from langgraph_checkpoint_postgres import PostgresSaver
    except ImportError:
        PostgresSaver = None

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq

from mcp_client import (
    tavily_mcp_search,
    get_airports,
    get_airlines,
    aviation_mcp_call,
    extract_destination,
    weather_mcp_search,
    forecast_mcp_search,
)

load_dotenv(override=True)

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def get_llm():
    """Safely initialize Groq LLM with api_key validation."""
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        return None
    try:
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"⚠️ Warning initializing Groq LLM: {e}")
        return None


llm = get_llm()


# State definition
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str
    final_response: str
    llm_calls: int


# Flight Agent Prompts
FLIGHT_AGENT_PROMPT = """You are an expert travel flight planner.

User Query:
{query}

Airport Information from Aviation MCP Server:
{airport_data}

Airline Information from Aviation MCP Server:
{airline_data}

Please generate:
1. Likely departure airport & arrival airport
2. Recommended airlines serving this route
3. Typical flight duration & routing
4. Estimated airfare range (Economy & Business)
5. Peak season travel pricing warnings
6. Practical flight booking tips

Return structured, concise travel guidance.
"""


def flight_agent(state: TravelState):
    """Query Aviation MCP Server and synthesize flight recommendations."""
    print("\n[Agent 1/5] ✈️ Flight Agent running (MCP Aviation)...")
    query = state["user_query"]
    active_llm = llm or get_llm()

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        if active_llm:
            prompt = FLIGHT_AGENT_PROMPT.format(
                query=query,
                airport_data=str(airports)[:3000],
                airline_data=str(airlines)[:3000],
            )
            response = active_llm.invoke([
                SystemMessage(content="You are an expert international flight travel planner."),
                HumanMessage(content=prompt),
            ])
            flight_data = response.content
        else:
            flight_data = (
                f"### ✈️ Flight Recommendations for: {query}\n\n"
                f"**Aviation MCP Airport Network:**\n{airports}\n\n"
                f"**Aviation MCP Airline Carriers:**\n{airlines}\n\n"
                f"💡 *Tip: Configure `GROQ_API_KEY` in `.env` to enable full AI flight route generation.*"
            )
    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated via Aviation MCP")],
        "llm_calls": state.get("llm_calls", 0) + (1 if active_llm else 0),
    }


def hotel_agent(state: TravelState):
    """Query Tavily MCP Server for accommodations."""
    print("\n[Agent 2/5] 🏨 Hotel Agent running (MCP Tavily)...")
    query = f"Best hotels for {state['user_query']}"
    active_llm = llm or get_llm()

    try:
        hotel_results = asyncio.run(tavily_mcp_search(query))
        if isinstance(hotel_results, list) and len(hotel_results) > 0 and isinstance(hotel_results[0], dict):
            hotel_results = hotel_results[0].get("text", str(hotel_results))
        elif isinstance(hotel_results, list):
            hotel_results = "\n".join([str(item) for item in hotel_results])
    except Exception as e:
        hotel_results = f"Hotel data unavailable: {e}"

    return {
        "hotel_results": str(hotel_results),
        "messages": [AIMessage(content="Hotel information fetched via Tavily MCP")],
        "llm_calls": state.get("llm_calls", 0),
    }


def weather_agent(state: TravelState):
    """Query Weather MCP Server for live climate and forecasts."""
    print("\n[Agent 3/5] ⛅ Weather Agent running (MCP Weather)...")
    query = state["user_query"]

    city = extract_destination(query)
    try:
        weather_data = asyncio.run(weather_mcp_search(city))
        forecast_data = asyncio.run(forecast_mcp_search(city))

        formatted_weather = (
            f"### ⛅ Weather & Climate Report for {city.title()}\n\n"
            f"**Current Conditions:**\n{weather_data}\n\n"
            f"**Travel Forecast & Advisories:**\n{forecast_data}"
        )
    except Exception as e:
        formatted_weather = f"Weather data unavailable for {city}: {e}"

    return {
        "weather_results": formatted_weather,
        "messages": [AIMessage(content="Weather information fetched via Weather MCP")],
        "llm_calls": state.get("llm_calls", 0),
    }


def itinerary_agent(state: TravelState):
    """Synthesize flights, hotels, and weather into a day-by-day plan."""
    print("\n[Agent 4/5] 🗓️ Itinerary Agent running (LangGraph)...")
    active_llm = llm or get_llm()

    if not active_llm:
        fallback_itinerary = (
            f"### 🗓️ Recommended Itinerary for {state['user_query']}\n\n"
            f"- **Day 1:** Arrival, check-in, orientation walk, and welcome dinner.\n"
            f"- **Day 2:** Iconic landmark tours and historical exploration.\n"
            f"- **Day 3:** Cultural districts, local markets, and culinary experiences.\n"
            f"- **Day 4:** Day excursion or scenic nature sightseeing.\n"
            f"- **Day 5:** Shopping, souvenirs, and farewell evening.\n"
            f"- **Day 6-7:** Leisure time, checkout, and flight departure.\n\n"
            f"💡 *Add `GROQ_API_KEY` to your `.env` for customized day-by-day AI itineraries.*"
        )
        return {
            "itinerary": fallback_itinerary,
            "messages": [AIMessage(content=fallback_itinerary)],
            "llm_calls": state.get("llm_calls", 0),
        }

    prompt = f"""You are a master itinerary travel planner.

Create a detailed day-by-day itinerary based on the real-time research below:

User Query:
{state['user_query']}

Flight Information (from Aviation MCP):
{state['flight_results']}

Hotel Recommendations (from Tavily MCP):
{state['hotel_results']}

Weather & Climate (from Weather MCP):
{state['weather_results']}

Please generate:
- Day-by-day schedule with morning, afternoon, and evening activities
- Meal and restaurant suggestions
- Packing and climate advice based on the weather
- Estimated daily budget breakdown
"""

    response = active_llm.invoke([
        SystemMessage(content="You are an expert luxury and leisure travel itinerary designer."),
        HumanMessage(content=prompt),
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def final_agent(state: TravelState):
    """Produce the final executive travel summary package."""
    print("\n[Agent 5/5] 🧠 Final Agent synthesizing travel plan...")
    active_llm = llm or get_llm()

    if not active_llm:
        final_summary = (
            f"# ✈️ Complete AI Travel Plan\n\n"
            f"**Trip Request:** {state['user_query']}\n\n"
            f"### Summary Highlights\n"
            f"- Flights and airport hubs retrieved via AviationStack MCP Server.\n"
            f"- Accommodations researched via Tavily MCP Server.\n"
            f"- Weather conditions provided via OpenWeather MCP Server.\n"
            f"- Itinerary generated and coordinated via LangGraph multi-agent pipeline.\n"
        )
        return {
            "final_response": final_summary,
            "messages": [AIMessage(content=final_summary)],
            "llm_calls": state.get("llm_calls", 0),
        }

    final_prompt = f"""You are the Chief Travel Concierge.
Synthesize the complete travel proposal into an engaging, cohesive, executive trip plan:

User Request: {state['user_query']}

Flight Overview:
{state['flight_results']}

Hotel Overview:
{state['hotel_results']}

Weather Forecast:
{state['weather_results']}

Itinerary:
{state['itinerary']}

Generate a polished markdown response with:
1. Trip Title & Overview
2. Key Highlights & Travel Tips
3. Flight & Accommodation Summary
4. Weather & Packing Guidance
5. Day-by-Day Schedule Summary
6. Estimated Total Budget & Cost Saving Tips
"""

    response = active_llm.invoke([
        SystemMessage(content="You are a professional travel concierge producing elite travel proposals."),
        HumanMessage(content=final_prompt),
    ])

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Assemble LangGraph Workflow
graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

# Checkpointer configuration with PostgreSQL & graceful in-memory fallback
checkpointer = None
try:
    if PostgresSaver and DATABASE_URL and not DATABASE_URL.startswith("postgresql://postgres:postgres@localhost"):
        _conn = psycopg.connect(DATABASE_URL)
        checkpointer = PostgresSaver(_conn)
        checkpointer.setup()
        print("Connected to PostgreSQL checkpointer.")
    else:
        raise ValueError("PostgresSaver unavailable or DATABASE_URL unset/placeholder.")
except Exception as e:
    print(f"⚠️ Note: PostgreSQL checkpointer unavailable ({e}). Using in-memory checkpointer.")
    checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    session_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }

    print("=" * 60)
    print("🌍 AI Travel Booking System (LangGraph + MCP)")
    print(f"Session Thread ID: {session_id}")
    print("=" * 60)

    user_input = input("\nEnter travel request (or press Enter for default Tokyo demo): ").strip()
    if not user_input:
        user_input = "Plan a complete 7 days Japan trip including flights, hotels and sightseeing under 2 lakhs."

    print(f"\nProcessing query: '{user_input}'\n")

    result = app.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    print("\n" + "=" * 60)
    print("🎉 FINAL TRAVEL PLAN GENERATED:")
    print("=" * 60)
    final_output = result.get("final_response") or result.get("itinerary")
    print(final_output)
