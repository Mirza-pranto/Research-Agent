from typing import List
from ddgs import DDGS
from schemas import ExtractedSource


def execute_web_search(sub_queries: List[str]) -> List[ExtractedSource]:
    """Search the web for each sub-query using the ddgs package and return structured sources."""
    results: List[ExtractedSource] = []

    # Initialize DDGS search client
    ddgs = DDGS()

    for query in sub_queries:
        try:
            # Perform text search
            raw_results = ddgs.text(query, max_results=3)

            if raw_results:
                for item in raw_results:
                    if isinstance(item, dict):
                        results.append(
                            ExtractedSource(
                                title=item.get("title", query),
                                url=item.get("href", item.get("link", "")),
                                snippet=item.get("body", item.get("content", "")),
                                relevance=f"Search results for: {query}",
                            )
                        )
                    else:
                        results.append(
                            ExtractedSource(
                                title=str(item),
                                url="",
                                snippet=str(item),
                                relevance=f"Search results for: {query}",
                            )
                        )
        except Exception as e:
            print(f"Error executing web search for query '{query}': {e}")
            continue

    return results