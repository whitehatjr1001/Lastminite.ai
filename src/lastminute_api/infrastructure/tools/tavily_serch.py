from langchain_tavily import TavilySearch

def web_search_tool(query: str) -> str:
    
    return TavilySearch(tavily_api_key=api_key).search(query)
