# STEM Revision Agent - Product Requirements Document

## Project Overview

**Product Name:** QuickSTEM Revision Agent  
**Project Type:** Personal Learning Project / Multi-Agent System  
**Target Users:** STEM students needing last-minute revision materials  
**Core Technology:** LangChain + LangGraph, MCP integrations, Mind mapping

## Problem Statement

STEM students often need quick, visual summaries of complex topics for last-minute revision, but existing tools either provide too much detail (full papers) or too little context (basic summaries). Current revision methods don't effectively synthesize multiple academic sources into digestible visual formats.

## Success Metrics (Learning Goals)

- **Technical Mastery:** Successfully implement multi-agent workflow using LangGraph
- **API Integration:** Robust handling of PubMed and arXiv APIs via MCP
- **Content Quality:** Generate coherent mind maps from academic sources
- **Personal Growth:** Understand agent orchestration, prompt engineering, and academic data processing

## Core Features

### MVP (Phase 1)
- **Single Topic Query:** Accept STEM topic input (e.g., "quantum tunneling", "CRISPR mechanism")
- **Dual Source Search:** Query both arXiv and PubMed for relevant papers
- **Content Synthesis Agent:** Extract key concepts and relationships from paper abstracts
- **Mind Map Generation:** Create visual mind map using text-to-image or structured diagram
- **Simple Web Interface:** Basic input/output interface for testing

### Phase 2 Enhancements
- **Multi-Agent Orchestration:** 
  - Research Agent (paper discovery)
  - Analysis Agent (content extraction)
  - Synthesis Agent (mind map creation)
  - Quality Agent (relevance filtering)
- **Enhanced Visuals:** Better mind map styling and hierarchical organization
- **Topic Expansion:** Related concept suggestions
- **Export Options:** PNG, PDF, or shareable formats

### Phase 3 (Stretch Goals)
- **Interactive Mind Maps:** Clickable nodes for deeper exploration
- **Multi-Topic Sessions:** Compare/contrast different concepts
- **Study Session Integration:** Timed revision modes
- **Collaborative Features:** Share mind maps with study groups

## Technical Architecture

### Agent Workflow
```
User Input → Research Agent → Analysis Agent → Synthesis Agent → Output
                ↓              ↓              ↓
            [arXiv/PubMed] → [Extract Key] → [Generate Map]
                            [Concepts]
```

### Technology Stack
- **Framework:** LangChain + LangGraph for agent orchestration
- **APIs:** arXiv API, PubMed API via MCP
- **LLM:** GPT-4 or Claude for content processing
- **Visualization:** Graphviz, D3.js, or AI-generated diagrams
- **Backend:** Python FastAPI
- **Frontend:** Streamlit or basic React interface

### Data Flow
1. **Input Processing:** Validate and expand user query
2. **Parallel Search:** Simultaneous arXiv + PubMed queries
3. **Content Filtering:** Remove irrelevant/low-quality results
4. **Concept Extraction:** Identify key terms, relationships, processes
5. **Mind Map Structure:** Create hierarchical concept organization
6. **Visual Generation:** Convert structure to visual mind map
7. **Quality Review:** Agent validates output relevance and accuracy

## User Stories

### Primary User Story
"As a STEM student cramming for an exam, I want to quickly generate a visual summary of 'mitochondrial respiration' so I can understand the key processes and relationships in under 5 minutes."

### Secondary User Stories
- "I want to compare different research perspectives on the same topic"
- "I need to understand complex biochemical pathways visually"
- "I want to save and reference multiple mind maps for different topics"

## Constraints & Assumptions

### Technical Constraints
- API rate limits (arXiv: 3 req/sec, PubMed: varies)
- LLM context window limitations for large papers
- Image generation processing time
- Local development environment (no cloud deployment initially)

### Assumptions
- Users have specific STEM topics in mind (not broad exploration)
- Abstract-level information is sufficient for revision purposes
- Visual learning is preferred over text-heavy summaries
- English language content only (MVP)

## Risk Assessment

### High Risk
- **API Reliability:** PubMed/arXiv downtime or rate limiting
- **Content Quality:** Generated summaries may be inaccurate or incomplete
- **Scope Creep:** Multi-agent systems can become complex quickly

### Medium Risk
- **Performance:** Slow response times due to multiple API calls
- **Relevance:** Search results may not match user intent
- **Visual Quality:** Generated mind maps may be cluttered or unclear

### Mitigation Strategies
- Implement robust error handling and fallbacks
- Start with single-agent MVP before adding complexity
- Cache successful results to reduce API dependency
- Test with real STEM content early and often

## Development Phases

### Phase 1: Core Pipeline (2-3 weeks)
- Set up LangChain environment
- Implement basic arXiv integration
- Create simple content extraction
- Generate text-based concept lists
- Basic testing interface

### Phase 2: Multi-Agent System (3-4 weeks)
- Implement LangGraph agent workflow
- Add PubMed integration via MCP
- Develop mind map generation
- Improve content synthesis quality
- User interface improvements

### Phase 3: Enhancement & Polish (2-3 weeks)
- Visual improvements
- Performance optimization
- Additional features based on testing
- Documentation and code cleanup

## Success Criteria

### Learning Objectives Met
- [ ] Successfully orchestrate multiple agents using LangGraph
- [ ] Integrate external APIs robustly with error handling
- [ ] Generate meaningful visual content from text sources
- [ ] Understand academic data processing challenges

### Functional Requirements
- [ ] Process user query in under 30 seconds
- [ ] Generate relevant mind maps for common STEM topics
- [ ] Handle API failures gracefully
- [ ] Provide clear visual output that aids comprehension

## Future Considerations

- Integration with note-taking apps (Notion, Obsidian)
- Mobile-friendly interface
- Offline mode with cached content
- Integration with university library systems
- Collaborative study features

---

## Notes for Implementation

**Start Simple:** Begin with single-agent, single-source (arXiv only) implementation before adding complexity.

**Focus on Learning:** Prioritize understanding LangGraph concepts over feature completeness.

**Test Early:** Use real STEM topics you're familiar with to validate concept extraction quality.

**Document Process:** Keep detailed notes on challenges and solutions for future reference.