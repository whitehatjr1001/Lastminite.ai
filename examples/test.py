from src.lastminute_api.infrastructure.llm_providers.base import get_llm_by_type
from src.lastminute_api.domain.tools.registry import create_mindmap, simple_mindmap, get_mindmap, display_mindmap
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
import re

# Clean, reference-aware tools
tools = [create_mindmap, simple_mindmap]

# Updated prompt for reference-based mindmaps
reference_mindmap_prompt = PromptTemplate.from_template("""
🧠 MINDMAP SPECIALIST - Reference-Based System

Tools: {tools}
Available: {tool_names}

You create mindmaps that return REFERENCE IDs instead of large data.

FORMAT:
Question: {input}
Thought: [plan approach]
Action: [tool name]
Action Input: [input format]
Observation: [reference result]
Final Answer: [tell user about their mindmap reference]

TOOL FORMATS:

✅ create_mindmap (recommended):
   Input: "Python: Variables, Functions, Classes, Libraries"
   Output: "✅ Mindmap 'Python' created! Reference: mm_123456_abc"

✅ simple_mindmap:
   Input: topic="Python Programming", subtopics="Variables,Functions,Classes"
   Output: "✅ Simple mindmap 'Python Programming' ready! Reference: mm_789012_def"

EXAMPLES:
- "Machine Learning: Supervised, Unsupervised, Deep Learning, Neural Networks"
- "Web Development: HTML, CSS, JavaScript, React, Node.js"

Return the reference ID to the user - they can view the actual mindmap using the reference.

Question: {input}
Thought: {agent_scratchpad}
""")

def create_reference_agent():
    """Create agent optimized for reference-based mindmaps."""
    model = get_llm_by_type("openai")
    
    agent = create_react_agent(llm=model, tools=tools, prompt=reference_mindmap_prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=2,
        early_stopping_method="generate"
    )

def extract_reference(text: str) -> str:
    """Extract reference ID from agent response."""
    match = re.search(r'Reference:\s*([a-zA-Z0-9_]+)', text)
    return match.group(1) if match else None

def mindmap_demo():
    """Complete demo: Create mindmap + Display it."""
    
    print("🧠 REFERENCE-BASED MINDMAP DEMO")
    print("="*50)
    
    # Method 1: Direct tool usage
    print("🚀 Direct Tool Usage:")
    ref_result = create_mindmap.invoke("Python: Variables, Functions, Classes, Libraries, Modules")
    print(f"Tool result: {ref_result}")
    
    ref_id = extract_reference(ref_result)
    if ref_id:
        print(f"📋 Extracted reference: {ref_id}")
        display_result = display_mindmap(ref_id)
        print(f"💾 {display_result}")
    
    print("\n" + "-"*50 + "\n")
    
    # Method 2: Agent usage
    print("🤖 Agent Usage:")
    agent = create_reference_agent()
    
    result = agent.invoke({"input": "Create a mindmap for Machine Learning concepts"})
    agent_output = result.get("output", "")
    print(f"Agent result: {agent_output}")
    
    # Extract and display
    ref_id = extract_reference(agent_output)
    if ref_id:
        print(f"📋 Agent reference: {ref_id}")
        display_result = display_mindmap(ref_id)
        print(f"💾 {display_result}")

def batch_mindmap_demo():
    """Create multiple mindmaps efficiently."""
    
    print("📊 BATCH MINDMAP CREATION")
    print("="*40)
    
    topics = [
        "Data Science: Collection, Cleaning, Analysis, Visualization, Modeling",
        "Web Development: Frontend, Backend, Database, APIs, Testing",
        "AI: Machine Learning, Deep Learning, NLP, Computer Vision, Robotics"
    ]
    
    references = []
    agent = create_reference_agent()
    
    for i, topic in enumerate(topics, 1):
        print(f"🔄 Creating mindmap {i}/3: {topic[:30]}...")
        
        try:
            # Use agent for variety
            if i % 2 == 1:
                result = agent.invoke({"input": f"Create mindmap: {topic}"})
                output = result.get("output", "")
            else:
                # Use direct tool
                output = create_mindmap.invoke(topic)
            
            ref_id = extract_reference(output)
            if ref_id:
                references.append(ref_id)
                print(f"✅ Created: {ref_id}")
            else:
                print("❌ No reference found")
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
    
    print(f"\n🎯 Created {len(references)} mindmaps!")
    
    # Create combined view
    if references:
        create_combined_view(references)

def create_combined_view(ref_ids):
    """Create HTML page showing all mindmaps."""
    from src.lastminute_api.domain.tools.registry import MINDMAP_STORE
    
    html_parts = ["""
<!DOCTYPE html>
<html><head><title>All Mindmaps</title>
<style>
body { font-family: Arial; margin: 20px; text-align: center; }
.mindmap { margin: 20px 0; padding: 20px; border: 2px solid #ddd; border-radius: 10px; }
img { max-width: 400px; height: auto; }
h3 { color: #333; }
</style></head><body>
<h1>🧠 Generated Mindmaps</h1>
"""]
    
    for ref_id in ref_ids:
        if ref_id in MINDMAP_STORE:
            data = MINDMAP_STORE[ref_id]
            html_parts.append(f"""
<div class="mindmap">
    <h3>{data['topic']}</h3>
    <p><strong>Topics:</strong> {', '.join(data['subtopics'])}</p>
    <img src="{data['base64']}" alt="Mindmap: {data['topic']}" />
    <p><small>Reference: {ref_id} | Created: {data['created'][:16]}</small></p>
</div>
""")
    
    html_parts.append("</body></html>")
    
    with open("all_mindmaps.html", "w") as f:
        f.write("".join(html_parts))
    
    print("📚 Combined view: all_mindmaps.html")

# Simple one-liner functions
def quick_create(topic_description: str) -> str:
    """One-liner mindmap creation."""
    result = create_mindmap.invoke(topic_description)
    ref_id = extract_reference(result)
    if ref_id:
        display_mindmap(ref_id)
        return f"Created and saved: {ref_id}"
    return "Failed to create mindmap"

def interactive_session():
    """Interactive mindmap creation session."""
    print("🧠 Interactive Mindmap Session")
    print("Commands: 'quit', 'view [ref_id]', 'list'")
    print("-" * 40)
    
    agent = create_reference_agent()
    created_refs = []
    
    while True:
        user_input = input("\n💭 Describe your mindmap: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'list':
            print(f"📋 Created references: {', '.join(created_refs)}")
            continue
        elif user_input.lower().startswith('view '):
            ref_id = user_input.split(' ', 1)[1]
            result = display_mindmap(ref_id)
            print(f"👁️ {result}")
            continue
        
        try:
            result = agent.invoke({"input": user_input})
            output = result.get("output", "")
            print(f"🤖 {output}")
            
            ref_id = extract_reference(output)
            if ref_id:
                created_refs.append(ref_id)
                print(f"💾 Auto-saved as {ref_id}.html")
                display_mindmap(ref_id)
                
        except Exception as e:
            print(f"❌ Error: {str(e)[:100]}")

if __name__ == "__main__":
    # Run demos
    mindmap_demo()
    
    print("\n" + "="*50 + "\n")
    batch_mindmap_demo()
    
    print("\n💡 Try these one-liners:")
    print("quick_create('JavaScript: Variables, Functions, Objects, Arrays')")
    
    # Uncomment for interactive mode
    # interactive_session()
