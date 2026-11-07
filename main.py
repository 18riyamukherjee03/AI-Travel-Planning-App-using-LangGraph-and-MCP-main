import os
from typing import TypedDict, Annotated
import operator

import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from langchain_groq import ChatGroq

from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Safe LLM initialization
def get_llm():
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("your_"):
        return None
    try:
        return ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize Groq LLM: {e}")
        return None

llm = get_llm()

# State
class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int

# Flight Agent
def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)
    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight results fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = tavily_search(query)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Itinerary Agent
def itinerary_agent(state: TravelState):
    active_llm = llm or get_llm()
    if not active_llm:
        msg = "⚠️ GROQ_API_KEY is not configured in .env. Please add your Groq API key from https://console.groq.com."
        return {
            "itinerary": msg,
            "messages": [AIMessage(content=msg)],
            "llm_calls": state.get("llm_calls", 0)
        }

    prompt = f"""
    Create a travel itinerary.
    User Query:
    {state['user_query']}

    Flight Results:
    {state['flight_results']}

    Hotel Results:
    {state['hotel_results']}
    """

    response = active_llm.invoke([
        SystemMessage(
            content="You are an expert travel planner"
        ),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Final Response Agent
def final_agent(state: TravelState):
    active_llm = llm or get_llm()
    if not active_llm:
        msg = "⚠️ GROQ_API_KEY is not configured in .env. Please add your Groq API key to generate the final response."
        return {
            "messages": [AIMessage(content=msg)],
            "llm_calls": state.get("llm_calls", 0)
        }

    final_prompt = f"""
    Generate final travel response.

    Flights:
    {state['flight_results']}

    Hotels:
    {state['hotel_results']}

    Itinerary:
    {state['itinerary']}
    """

    response = active_llm.invoke([
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }


graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


# Checkpointer setup with graceful fallback
checkpointer = None
try:
    if DATABASE_URL and not DATABASE_URL.startswith("postgresql://postgres:postgres@"):
        _conn = psycopg.connect(DATABASE_URL)
        checkpointer = PostgresSaver(_conn)
        checkpointer.setup()
        print("Connected to PostgreSQL checkpointer.")
    else:
        raise ValueError("DATABASE_URL contains placeholder credentials or is unset.")
except Exception as e:
    print(f"⚠️ Note: PostgreSQL not ready ({e}). Falling back to in-memory checkpointer.")
    checkpointer = MemorySaver()

app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {
        "configurable": {
            "thread_id": "user_aarohi"
        }
    }

    user_input = input("Enter travel request: ")

    result = app.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    print("\nFINAL RESPONSE:\n")

    for msg in result["messages"]:
        print(msg.content)
