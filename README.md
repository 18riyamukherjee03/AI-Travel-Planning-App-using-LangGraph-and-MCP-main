# Multi-Agent AI Travel Booking System using LangGraph & MCP

A full-stack, autonomous Multi-Agent Travel Planning & Booking application built with **LangGraph**, **Model Context Protocol (MCP)**, **Groq LLaMA 3.3 70B**, and **Streamlit**.

This project represents **Part 2** of the system, upgrading the core travel architecture with **Model Context Protocol (MCP)** servers for real-time flight tracking, global weather forecasting, and live web research.

---

## 🌟 Project Evolution: Part 1 vs Part 2

| Capability | Part 1 (Direct APIs) | Part 2 (LangGraph + MCP) |
|---|---|---|
| **Multi-Agent Orchestration** | 4 Sequential LangGraph Agents | 5 Specialized LangGraph Agents |
| **Flight Intelligence** | Direct REST API calls | **AviationStack MCP Server** (stdio) |
| **Weather & Forecasts** | None | **OpenWeatherMap MCP Server** (stdio) |
| **Hotel Research** | Direct Tavily API | **Tavily MCP Server** (streamable HTTP) |
| **Memory / Checkpointing** | PostgreSQL with In-Memory fallback | PostgreSQL with In-Memory fallback |
| **Interactive UI** | Streamlit Modern Dashboard | Streamlit Modern Dashboard with 5 Agents |

* **Part 1 Repository:** [https://github.com/18riyamukherjee03/AI-Travel-Planning-System-using-LangGraph](https://github.com/18riyamukherjee03/AI-Travel-Planning-System-using-LangGraph)

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    User["👤 User Request"] --> UI["💻 Streamlit Web UI / Terminal CLI"]
    UI --> Graph["🔗 LangGraph Workflow (Compiled with Checkpointer)"]
    
    subgraph Agents["🤖 Specialized AI Agent Team"]
        A1["① ✈️ Flight Agent"]
        A2["② 🏨 Hotel Agent"]
        A3["③ ⛅ Weather Agent"]
        A4["④ 🗓️ Itinerary Agent"]
        A5["⑤ 🧠 Final Concierge Agent"]
    end
    
    Graph --> A1
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5
    
    subgraph MCPServers["⚡ Model Context Protocol (MCP) Infrastructure"]
        MCPAviation["✈️ AviationStack MCP Server (stdio)"]
        MCPWeather["⛅ Weather MCP Server (stdio)"]
        MCPTavily["🔍 Tavily MCP Server (streamable_http)"]
    end
    
    A1 <-->|"list_airports, list_airlines"| MCPAviation
    A2 <-->|"tavily_search"| MCPTavily
    A3 <-->|"get_current_weather, get_forecast"| MCPWeather
    
    A4 --> LLM["🧠 Groq LLaMA 3.3 70B"]
    A5 --> LLM
    
    A5 --> Plan["📄 Complete Travel Plan (.md Download & Display)"]
    Graph <--> DB[("🐘 PostgreSQL / In-Memory Checkpointer")]
```

---

## 🚀 Key Features

- **5 Specialized AI Agents:**
  1. **Flight Agent:** Discovers optimal departure and arrival hubs, carrier networks, and flight guidance via AviationStack MCP.
  2. **Hotel Agent:** Identifies top luxury, boutique, and budget accommodations via Tavily MCP.
  3. **Weather Agent:** Evaluates destination climate and multi-day forecasts via OpenWeather MCP to provide tailored packing advice.
  4. **Itinerary Agent:** Crafts an immersive day-by-day itinerary balancing cultural sightseeing, activities, and dining.
  5. **Final Concierge Agent:** Assembles an executive-ready proposal with budget estimates and cost-saving tips.
- **Model Context Protocol (MCP) Standards:** Standardized stdio and streamable HTTP tool discovery and execution via `langchain-mcp-adapters`.
- **Fault-Tolerant Resilience:** Includes graceful fallback mechanisms for API quotas, network limits, and optional PostgreSQL database storage.
- **Modern UI & Report Generation:** Beautiful dark-themed Streamlit web interface with real-time streaming status widgets and automatic Markdown plan export.

---

## 📋 Prerequisites & APIs

Get your free API keys:
- **Groq Cloud:** [https://console.groq.com](https://console.groq.com)
- **Tavily AI:** [https://tavily.com](https://tavily.com)
- **AviationStack:** [https://aviationstack.com](https://aviationstack.com)
- **OpenWeatherMap:** [https://openweathermap.org](https://openweathermap.org)

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/18riyamukherjee03/AI-Travel-Planning-App-using-LangGraph-and-MCP-main.git
cd AI-Travel-Planning-App-using-LangGraph-and-MCP-main
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and fill in your API credentials:
```env
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
AVIATIONSTACK_API_KEY=...
OPENWEATHER_API_KEY=...

# Optional: PostgreSQL Database (leave default for in-memory checkpointer)
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/langgraph_memory_demo
```

---

## 🖥️ Usage

### Run the Streamlit Web Application
```bash
streamlit run frontend.py
```
Open your browser at `http://localhost:8501`.

### Run via Terminal CLI
```bash
python main.py
```

### Run MCP Client Self-Test
Verify all MCP servers and tools directly:
```bash
python mcp_client.py
```

---

## 📁 Project Directory Structure

```text
├── custom_aviation_mcp_server.py # FastMCP Aviation server (airports, airlines, status)
├── custom_weather_mcp_server.py  # FastMCP Weather server (current conditions, forecasts)
├── mcp_client.py                 # MultiServerMCPClient manager & tool router
├── main.py                       # LangGraph StateGraph pipeline & 5 agent nodes
├── frontend.py                   # Streamlit interactive dark-theme dashboard
├── tools/                        # Fallback direct tools (Part 1 compatibility)
│   ├── flight_tool.py            # AviationStack direct REST helper
│   └── tavily_tool.py            # Tavily direct search helper
├── requirements.txt              # Project dependencies
├── .env.example                  # Environment configuration template
├── .gitignore                    # Git ignore rules
└── README.md                     # Documentation
```

---

## 🛡️ License & Attribution
Authored by **Riya Mukherjee**. Developed with LangGraph and Model Context Protocol.
