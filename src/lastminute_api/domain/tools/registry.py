# src/lastminute_api/domain/tools/registry.py
from typing import List
from langchain_core.tools import tool
import json
import re
import hashlib
from datetime import datetime

# Global cache for mindmap data
MINDMAP_STORE = {}

# Import DynamicGraph class
try:
    from .graph_tool import DynamicGraph
except ImportError:
    try:
        from src.lastminute_api.domain.tools.graph_tool import DynamicGraph
    except ImportError:
        class DynamicGraph:
            @staticmethod
            def create_graph(nodes, edges, **kwargs):
                return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

def _generate_ref_id(input_text: str) -> str:
    """Generate unique reference ID."""
    timestamp = datetime.now().strftime("%H%M%S")
    content_hash = hashlib.md5(input_text.encode()).hexdigest()[:6]
    return f"mm_{timestamp}_{content_hash}"

def _store_mindmap(ref_id: str, base64_url: str, input_text: str, topic: str, subtopics: List[str]):
    """Store mindmap data and save file."""
    # Store in memory cache
    MINDMAP_STORE[ref_id] = {
        "base64": base64_url,
        "input": input_text,
        "topic": topic,
        "subtopics": subtopics,
        "created": datetime.now().isoformat()
    }
    
    # Save PNG file
    if base64_url.startswith("data:image/png;base64,"):
        import base64
        with open(f"{ref_id}.png", "wb") as f:
            f.write(base64.b64decode(base64_url.split("data:image/png;base64,")[1]))

@tool
def create_mindmap(input_text: str) -> str:
    """Create mind map and return reference ID instead of base64."""
    try:
        topic, subtopics = _parse_input(input_text)
        nodes = [topic] + subtopics[:8]
        edges = [(topic, subtopic) for subtopic in subtopics[:8]]
        
        # Generate actual mindmap (full base64)
        base64_url = DynamicGraph.create_graph(
            nodes=nodes, edges=edges, title=f"Mind Map: {topic}",
            node_color='lightblue', node_size=2000, figsize=(10, 8),
            font_size=11, edge_color='darkblue'
        )
        
        # Create reference and store data
        ref_id = _generate_ref_id(input_text)
        _store_mindmap(ref_id, base64_url, input_text, topic, subtopics)
        
        # Return SHORT reference to LLM
        return f"✅ Mindmap '{topic}' created! Reference: {ref_id} | Topics: {', '.join(subtopics[:3])}{'...' if len(subtopics) > 3 else ''}"
        
    except Exception as e:
        return f"❌ Error: {str(e)[:80]}"

@tool
def simple_mindmap(topic: str, subtopics: str = "") -> str:
    """Create simple mindmap with reference return."""
    try:
        # Parse subtopics
        subtopic_list = [s.strip() for s in subtopics.split(',') if s.strip()] if subtopics else _get_default_subtopics(topic)
        while len(subtopic_list) < 3:
            subtopic_list.append(f"Topic {len(subtopic_list) + 1}")
        
        nodes = [topic] + subtopic_list[:6]
        edges = [(topic, sub) for sub in subtopic_list[:6]]
        
        base64_url = DynamicGraph.create_graph(
            nodes=nodes, edges=edges, title=f"Mind Map: {topic}",
            node_color='lightgreen', figsize=(10, 8)
        )
        
        ref_id = _generate_ref_id(f"{topic}:{subtopics}")
        _store_mindmap(ref_id, base64_url, f"{topic}:{subtopics}", topic, subtopic_list)
        
        return f"✅ Simple mindmap '{topic}' ready! Reference: {ref_id}"
        
    except Exception as e:
        return f"❌ Error: {str(e)[:50]}"

def get_mindmap(ref_id: str) -> str:
    """Retrieve full base64 URL from reference."""
    return MINDMAP_STORE.get(ref_id, {}).get("base64", "Reference not found")

def display_mindmap(ref_id: str) -> str:
    """Create HTML file for viewing mindmap."""
    if ref_id not in MINDMAP_STORE:
        return "Reference not found"
    
    data = MINDMAP_STORE[ref_id]
    html = f"""<!DOCTYPE html>
<html><head><title>{data['topic']}</title></head>
<body style="text-align:center;font-family:Arial;">
<h2>🧠 {data['topic']}</h2>
<p>Topics: {', '.join(data['subtopics'])}</p>
<img src="{data['base64']}" style="max-width:90%;border:2px solid #ddd;border-radius:10px;"/>
<p><small>Reference: {ref_id} | Created: {data['created'][:16]}</small></p>
</body></html>"""
    
    with open(f"{ref_id}.html", "w") as f:
        f.write(html)
    return f"Saved: {ref_id}.html"

def _parse_input(input_text: str) -> tuple:
    """Parse input formats to extract topic and subtopics."""
    # JSON format
    try:
        if input_text.strip().startswith('{'):
            data = json.loads(input_text)
            return data.get('topic', 'Main Topic'), data.get('subtopics', [])
    except:
        pass
    
    # Colon format: "Topic: item1, item2"
    if ':' in input_text:
        parts = input_text.split(':', 1)
        topic = parts[0].strip()
        subtopics = [s.strip() for s in parts[1].split(',') if s.strip()]
        if subtopics:
            return topic, subtopics
    
    # Comma format: "Topic, item1, item2"
    if ',' in input_text:
        parts = [p.strip() for p in input_text.split(',')]
        if len(parts) >= 4:
            return parts[0], parts[1:]
    
    # Python-like format
    topic_match = re.search(r'topic\s*[=:]\s*["\']([^"\']+)["\']', input_text)
    if topic_match:
        topic = topic_match.group(1)
        subtopics_match = re.search(r'subtopics\s*[=:]\s*\[(.*?)\]', input_text)
        subtopics = []
        if subtopics_match:
            subtopics = [s.strip().strip('"\'') for s in re.findall(r'["\']([^"\']+)["\']', subtopics_match.group(1))]
        return topic, subtopics or _get_default_subtopics(topic)
    
    # Natural language
    return _extract_topic(input_text), _get_default_subtopics(_extract_topic(input_text), input_text)

def _extract_topic(text: str) -> str:
    """Extract main topic from natural language."""
    text_lower = text.lower()
    
    topics = {
        ["python", "programming"]: "Python Programming",
        ["machine", "learning", "ai", "ml"]: "Machine Learning", 
        ["web", "website", "html"]: "Web Development",
        ["data", "science", "analysis"]: "Data Science",
        ["javascript", "js"]: "JavaScript"
    }
    
    for keywords, topic in topics.items():
        if any(x in text_lower for x in keywords):
            return topic
    
    # Extract first meaningful word
    for word in text.split():
        if len(word) > 3 and word.isalpha() and word.lower() not in ['mind', 'map', 'create', 'about', 'with']:
            return word.title()
    
    return "Main Topic"

def _get_default_subtopics(topic: str, context: str = "") -> List[str]:
    """Generate default subtopics based on topic."""
    defaults = {
        "python": ["Variables", "Functions", "Classes", "Libraries", "Modules"],
        "javascript": ["Variables", "Functions", "Objects", "DOM", "Events"],
        "machine": ["Supervised Learning", "Unsupervised Learning", "Deep Learning", "Neural Networks"],
        "web": ["Frontend", "Backend", "Database", "APIs", "Deployment"],
        "data": ["Collection", "Cleaning", "Analysis", "Visualization", "Modeling"]
    }
    
    for key, subtopics in defaults.items():
        if key in topic.lower() or key in context.lower():
            return subtopics
    
    return ["Concept A", "Concept B", "Concept C", "Concept D"]

# Export tools and utilities
__all__ = ['create_mindmap', 'simple_mindmap', 'get_mindmap', 'display_mindmap', 'MINDMAP_STORE']
