from .graph_tool import DynamicGraph
from pydantic import BaseModel
from typing import Dict, List, Tuple
from langchain_core.tools import tool 
 
class MindmapShema(BaseModel):
    """Schema for mindmap tool input."""
    node_names: List[str]
    edge_map: List[Tuple[str, str]]

@tool(args_schema=MindmapShema)
def create_mindmap_graph(node_names: List[str], edge_map: List[Tuple[str, str]]) -> DynamicGraph:
    """Create a DynamicGraph instance for the mindmap."""
    graph = DynamicGraph.create_graph(node_names, edge_map)
    return graph