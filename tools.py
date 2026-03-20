import logging
import xml.etree.ElementTree as ET

import httpx
from livekit.agents import function_tool

from config import TAVILY_API_KEY, HF_TOKEN

logger = logging.getLogger(__name__)


@function_tool
async def search_web(query: str) -> str:
    """Search the web for ML papers, docs, benchmarks, or any current information"""
    logger.info("TOOL CALLED: search_web | query: %s", query)

    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "max_results": 3,
            },
        )

    logger.info("Tavily status: %s | body: %s", r.status_code, r.text[:300])

    if r.status_code != 200:
        return f"Search failed: {r.status_code} - {r.text[:100]}"

    data = r.json()
    results = data.get("results", [])

    if not results:
        return f"No results found for: {query}"

    output = []
    for res in results[:2]:
        title   = res.get("title", "No title")
        content = res.get("content", "No content")[:200]
        url     = res.get("url", "")
        output.append(f"{title}: {content} ({url})")

    result = "\n\n".join(output)
    logger.info("TOOL RESULT: search_web | %s", result[:100])
    return result


@function_tool
async def calculate(expression: str) -> str:
    """Evaluate a math expression — latency, cost, parameter counts etc"""
    logger.info("TOOL CALLED: calculate | expression: %s", expression)
    try:
        return str(eval(expression))  # safe for simple math
    except Exception as e:
        logger.error("TOOL ERROR: calculate | error: %s", e)
        return f"Error: {e}"


@function_tool
async def search_arxiv(query: str) -> str:
    """Search arxiv for ML/AI research papers"""
    logger.info("TOOL CALLED: search_arxiv | query: %s", query)

    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "max_results": 3},
        )

    root    = ET.fromstring(r.text)
    ns      = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)

    if not entries:
        return "No papers found."

    results = []
    for entry in entries[:3]:
        title   = entry.find("atom:title", ns).text.strip()
        summary = entry.find("atom:summary", ns).text.strip()[:150]
        link    = entry.find("atom:id", ns).text.strip()
        results.append(f"- {title}: {summary}... ({link})")

    result = "\n".join(results)
    logger.info("TOOL RESULT: search_arxiv | %s", result[:100])
    return result


@function_tool
async def lookup_model(model_name: str) -> str:
    """Look up a HuggingFace model's stats"""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"https://huggingface.co/api/models/{model_name}",
            headers=headers,
        )

    if r.status_code != 200:
        return f"Model not found: {model_name}"

    data = r.json()
    return f"Downloads: {data.get('downloads')}, Likes: {data.get('likes')}, Tags: {data.get('tags')}"