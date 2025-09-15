# Project Agents.md Guide for STEM Revision Multi-Agent System

This Agents.md file provides comprehensive guidance for AI coding assistants working with the STEM Revision Agent codebase, built with Python, LangChain, LangGraph, and UV environment management.

## Project Structure for AI Agent Navigation

```
stem-revision-agent/
├── src/                          # Core source code for AI agents to analyze
│   ├── agents/                   # LangGraph agent implementations
│   │   ├── research_agent.py     # arXiv/PubMed search agent
│   │   ├── analysis_agent.py     # Content extraction and processing
│   │   ├── synthesis_agent.py    # Mind map generation agent
│   │   └── coordinator_agent.py  # Multi-agent orchestration
│   ├── core/                     # Core business logic
│   │   ├── models/               # Pydantic data models
│   │   ├── services/             # External API integrations
│   │   └── utils/                # Utility functions and helpers
│   ├── api/                      # FastAPI route handlers
│   ├── workflows/                # LangGraph workflow definitions
│   └── visualization/            # Mind map generation components
├── tests/                        # Test files that AI agents should maintain
├── config/                       # Configuration files
├── scripts/                      # Development and deployment scripts
└── docs/                         # Documentation
```

## Environment Setup for AI Agents

### UV Environment Management
```bash
# Initialize UV environment (AI agents should reference this)
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (AI agents should use this command)
uv pip install -r requirements.txt

# Add new dependencies (AI agents should update requirements.txt)
uv pip install package_name
uv pip freeze > requirements.txt
```

### Essential Dependencies
```python
# Core dependencies that AI agents should be familiar with
langchain>=0.1.0
langgraph>=0.0.40
pydantic>=2.0.0
fastapi>=0.104.0
httpx>=0.25.0
python-dotenv>=1.0.0
```

## Coding Conventions for AI Agents

### Python Standards for AI Code Generation
- **Python Version**: 3.11+ (AI agents should target this version)
- **Code Style**: Black formatting, isort imports, flake8 linting
- **Type Hints**: Mandatory for all function signatures and class attributes
- **Docstrings**: Google style docstrings for all public functions and classes
- **Error Handling**: Use custom exceptions, never bare `except:` clauses

### LangChain/LangGraph Patterns for AI Agents
```python
# AI agents should follow this pattern for LangChain tools
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional

class ToolInputSchema(BaseModel):
    query: str = Field(description="Search query for academic papers")
    max_results: int = Field(default=10, description="Maximum results to return")

class CustomTool(BaseTool):
    name: str = "custom_tool"
    description: str = "Tool description for AI agents"
    args_schema: Type[BaseModel] = ToolInputSchema
    
    def _run(self, query: str, max_results: int = 10) -> str:
        """AI agents should implement robust error handling here."""
        try:
            # Implementation
            return result
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            raise
```

### Agent Implementation Patterns
```python
# AI agents should follow this structure for LangGraph agents
from langgraph import StateGraph
from typing import TypedDict, Optional
import logging

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    query: str
    papers: Optional[list]
    concepts: Optional[list]
    mind_map: Optional[dict]
    error: Optional[str]

def create_research_agent() -> StateGraph:
    """AI agents should create state graphs following this pattern."""
    workflow = StateGraph(AgentState)
    
    # Add nodes with proper error handling
    workflow.add_node("search", search_papers)
    workflow.add_node("filter", filter_results)
    workflow.add_node("extract", extract_concepts)
    
    # Define edges with conditional routing
    workflow.add_conditional_edges(
        "search",
        should_continue,
        {"continue": "filter", "end": "__end__"}
    )
    
    return workflow.compile()
```

### Data Models for AI Agents
```python
# AI agents should use Pydantic models for data validation
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum

class PaperSource(str, Enum):
    ARXIV = "arxiv"
    PUBMED = "pubmed"

class AcademicPaper(BaseModel):
    """Model for academic papers - AI agents should follow this pattern."""
    title: str = Field(..., description="Paper title")
    abstract: str = Field(..., description="Paper abstract")
    authors: List[str] = Field(default_factory=list)
    source: PaperSource = Field(..., description="Paper source")
    url: Optional[str] = Field(None, description="Paper URL")
    
    @validator('title')
    def title_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Title cannot be empty')
        return v.strip()

class MindMapNode(BaseModel):
    """AI agents should use this for mind map structure."""
    id: str
    label: str
    level: int = Field(ge=0, le=5, description="Hierarchy level")
    children: List['MindMapNode'] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)

# Enable forward references
MindMapNode.model_rebuild()
```

## Testing Requirements for AI Agents

### Testing Patterns for Clean Architecture

#### Domain Layer Tests
```python
# tests/domain/test_tools.py - AI agents should test domain logic in isolation
import pytest
from unittest.mock import AsyncMock
from src.lastminute_api.domain.tools import AcademicSearchTool, ToolResult
from src.lastminute_api.domain.exceptions import DomainError

class TestAcademicSearchTool:
    """AI agents should group domain tests by aggregate root."""
    
    @pytest.fixture
    def search_tool(self):
        """AI agents should use fixtures for test setup."""
        return AcademicSearchTool()
    
    @pytest.mark.asyncio
    async def test_execute_success(self, search_tool):
        """AI agents should test happy path scenarios."""
        # Arrange
        search_tool._search_papers = AsyncMock(return_value=[
            {"title": "Test Paper", "abstract": "Test abstract"}
        ])
        
        # Act
        result = await search_tool.execute(
            query="quantum computing",
            sources=["arxiv"],
            max_results=5
        )
        
        # Assert
        assert result.success is True
        assert len(result.data) == 1
        assert result.data[0]["title"] == "Test Paper"
        assert result.metadata["query"] == "quantum computing"
    
    @pytest.mark.asyncio
    async def test_execute_empty_query_fails(self, search_tool):
        """AI agents should test domain invariants."""
        # Act
        result = await search_tool.execute(query="")
        
        # Assert
        assert result.success is False
        assert "cannot be empty" in result.error_message
    
    @pytest.mark.asyncio
    async def test_execute_handles_infrastructure_failure(self, search_tool):
        """AI agents should test error handling."""
        # Arrange
        search_tool._search_papers = AsyncMock(side_effect=Exception("API failure"))
        
        # Act
        result = await search_tool.execute(query="valid query")
        
        # Assert
        assert result.success is False
        assert "API failure" in result.error_message

#### Application Layer Tests
# tests/application/test_chat_service.py - AI agents should test use cases
import pytest
from unittest.mock import Mock, AsyncMock
from src.lastminute_api.application.chat_service import ChatService
from src.lastminute_api.domain.exceptions import ApplicationError

class TestChatService:
    """AI agents should test application service orchestration."""
    
    @pytest.fixture
    def mock_tools(self):
        """AI agents should mock dependencies at boundaries."""
        search_tool = Mock()
        search_tool.name = "academic_search"
        search_tool.execute = AsyncMock()
        return [search_tool]
    
    @pytest.fixture
    def mock_llm_provider(self):
        llm = Mock()
        llm.generate = AsyncMock()
        return llm
    
    @pytest.fixture
    def chat_service(self, mock_tools, mock_llm_provider):
        return ChatService(tools=mock_tools, llm_provider=mock_llm_provider)
    
    @pytest.mark.asyncio
    async def test_process_revision_request_success(self, chat_service, mock_tools):
        """AI agents should test successful use case execution."""
        # Arrange
        mock_tools[0].execute.return_value = Mock(
            success=True,
            data=[{"title": "Test Paper"}],
            metadata={"query": "test"}
        )
        chat_service._generate_mind_map = AsyncMock(return_value={"root": "test"})
        
        # Act
        result = await chat_service.process_revision_request("quantum computing")
        
        # Assert
        assert "mind_map" in result
        assert result["sources_count"] == 1
        mock_tools[0].execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_revision_request_short_query_fails(self, chat_service):
        """AI agents should test input validation."""
        with pytest.raises(ApplicationError) as exc_info:
            await chat_service.process_revision_request("hi")
        
        assert "at least 3 characters" in str(exc_info.value)

#### Infrastructure Layer Tests
# tests/infrastructure/test_mcp_clients.py - AI agents should test external integrations
import pytest
import httpx
from unittest.mock import Mock, patch
from src.lastminute_api.infrastructure.mcp_clients import ArxivMCPClient
from src.lastminute_api.domain.exceptions import InfrastructureError

class TestArxivMCPClient:
    """AI agents should test infrastructure adapters."""
    
    @pytest.mark.asyncio
    async def test_search_papers_success(self):
        """AI agents should test successful API interactions."""
        # Arrange
        mock_response = Mock()
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed>
            <entry>
                <title>Test Paper</title>
                <summary>Test abstract</summary>
            </entry>
        </feed>"""
        
        with patch('httpx.AsyncClient.get', return_value=mock_response):
            client = ArxivMCPClient()
            client._parse_arxiv_response = Mock(return_value=[
                {"title": "Test Paper", "abstract": "Test abstract"}
            ])
            
            # Act
            async with client:
                results = await client.search_papers("quantum computing")
            
            # Assert
            assert len(results) == 1
            assert results[0]["title"] == "Test Paper"
    
    @pytest.mark.asyncio
    async def test_search_papers_timeout_raises_infrastructure_error(self):
        """AI agents should test error scenarios."""
        with patch('httpx.AsyncClient.get', side_effect=httpx.TimeoutException("Timeout")):
            client = ArxivMCPClient()
            
            async with client:
                with pytest.raises(InfrastructureError) as exc_info:
                    await client.search_papers("query")
                
                assert "timeout" in str(exc_info.value).lower()

#### Integration Tests
# tests/integration/test_api_endpoints.py - AI agents should test full request flows
import pytest
from httpx import AsyncClient
from src.lastminute_api.infrastructure.api.main import create_app

@pytest.mark.asyncio
async def test_revision_endpoint_integration():
    """AI agents should test complete request/response cycles."""
    # Arrange
    app = create_app()
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Act
        response = await client.post("/api/v1/revision", json={
            "topic": "quantum entanglement",
            "sources": ["arxiv"],
            "max_papers": 5
        })
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "mind_map" in data
        assert "sources_used" in data
```
        
        assert len(result) == 1
        assert result[0].title == "Test Paper"
        mock_search.assert_called_once_with("quantum computing")
    
    def test_search_papers_with_invalid_query(self):
        """AI agents should test edge cases."""
        agent = ResearchAgent()
        
        with pytest.raises(ValueError):
            agent.search_papers("")
```

### Running Tests
```bash
# AI agents should use these commands for testing
pytest tests/ -v                          # Run all tests with verbose output
pytest tests/test_agents/ -k "research"   # Run specific test pattern
pytest --cov=src --cov-report=html        # Generate coverage report
```

## API Development Guidelines for AI Agents

### FastAPI Route Patterns
```python
# AI agents should follow this pattern for API routes
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["revision"])

class RevisionRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    sources: List[str] = Field(default=["arxiv", "pubmed"])
    max_papers: int = Field(default=10, ge=1, le=50)

class RevisionResponse(BaseModel):
    mind_map: dict
    sources_used: List[str]
    processing_time: float

@router.post("/generate-revision", response_model=RevisionResponse)
async def generate_revision(request: RevisionRequest):
    """AI agents should include comprehensive error handling."""
    try:
        # Process request
        result = await process_revision_request(request)
        return RevisionResponse(**result)
    
    except ValueError as e:
        logger.warning(f"Invalid request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Revision generation failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Configuration Management for AI Agents

### Environment Variables
```python
# AI agents should use this pattern for configuration
from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    """AI agents should centralize configuration here."""
    
    # API Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Service Configuration
    arxiv_base_url: str = Field(
        default="http://export.arxiv.org/api/query",
        env="ARXIV_BASE_URL"
    )
    pubmed_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        env="PUBMED_BASE_URL"
    )
    
    # Application Settings
    max_concurrent_requests: int = Field(default=5, env="MAX_CONCURRENT_REQUESTS")
    request_timeout: int = Field(default=30, env="REQUEST_TIMEOUT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

## Logging Standards for AI Agents

```python
# AI agents should use structured logging
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    """AI agents should use this for consistent log formatting."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id
        
        return json.dumps(log_data)

# AI agents should configure logging like this
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/app.log")
        ]
    )
    
    for handler in logging.root.handlers:
        handler.setFormatter(StructuredFormatter())
```

## Development Workflow for AI Agents

### Pre-commit Checks
```bash
# AI agents should run these before committing
black src/ tests/                    # Format code
isort src/ tests/                    # Sort imports
flake8 src/ tests/                   # Lint code
mypy src/                           # Type check
pytest tests/ --cov=src            # Run tests with coverage
```

### Git Commit Conventions
```bash
# AI agents should follow conventional commits
feat: add research agent for arXiv integration
fix: handle rate limiting in PubMed API calls
docs: update agents.md with new patterns
test: add integration tests for mind map generation
refactor: extract common API client patterns
```

## Performance Guidelines for AI Agents

### Async Patterns
```python
# AI agents should use async/await for I/O operations
import asyncio
import httpx
from typing import List

async def fetch_papers_concurrently(queries: List[str]) -> List[dict]:
    """AI agents should implement concurrent API calls."""
    async with httpx.AsyncClient() as client:
        tasks = [fetch_single_query(client, query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        return [r for r in results if not isinstance(r, Exception)]
```

### Memory Management
```python
# AI agents should be mindful of memory usage with large datasets
from typing import Iterator, List
import gc

def process_papers_in_batches(papers: List[dict], batch_size: int = 100) -> Iterator[List[dict]]:
    """AI agents should process large datasets in batches."""
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        yield batch
        
        # Explicit garbage collection for large batches
        if len(batch) == batch_size:
            gc.collect()
```

## Error Handling Patterns for AI Agents

```python
# AI agents should use custom exceptions for domain errors
class StemRevisionError(Exception):
    """Base exception for STEM revision system."""
    pass

class PaperNotFoundError(StemRevisionError):
    """Raised when no papers found for query."""
    pass

class APIRateLimitError(StemRevisionError):
    """Raised when API rate limit exceeded."""
    pass

# AI agents should implement retry logic
import time
from functools import wraps

def retry_with_backoff(max_retries=3, backoff_factor=2):
    """AI agents should use this decorator for retrying operations."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except APIRateLimitError:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = backoff_factor ** attempt
                    await asyncio.sleep(wait_time)
            return None
        return wrapper
    return decorator
```

## Documentation Standards for AI Agents

```python
# AI agents should document functions with comprehensive docstrings
def extract_key_concepts(paper: AcademicPaper, max_concepts: int = 10) -> List[str]:
    """Extract key concepts from an academic paper.
    
    This function analyzes the paper's title and abstract to identify
    the most important concepts for mind map generation.
    
    Args:
        paper: The academic paper to analyze
        max_concepts: Maximum number of concepts to extract (default: 10)
        
    Returns:
        List of key concepts ordered by importance
        
    Raises:
        ValueError: If paper abstract is empty or too short
        APIError: If concept extraction service fails
        
    Example:
        >>> paper = AcademicPaper(title="Quantum Computing", abstract="...")
        >>> concepts = extract_key_concepts(paper, max_concepts=5)
        >>> len(concepts) <= 5
        True
    """
```

---

## Notes for AI Development Agents

**Code Quality**: Always prioritize readable, maintainable code over clever solutions. The multi-agent system complexity requires clear, debuggable implementations.

**Testing First**: Write tests for core functionality before implementing complex agent interactions. LangGraph workflows can be difficult to debug without proper test coverage.

**Incremental Development**: Build one agent at a time, test thoroughly, then integrate. Don't attempt to build the entire multi-agent system at once.

**Resource Management**: Be mindful of API rate limits and implement proper backoff strategies. Academic APIs are often slower and less reliable than commercial ones.

**Monitoring**: Include comprehensive logging and metrics collection. Multi-agent systems can fail in unexpected ways, and good observability is crucial for debugging.