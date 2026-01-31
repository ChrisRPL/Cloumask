# Cloumask Multi-User Architecture

## Deployment Modes

Cloumask supports three deployment modes, similar to Label Studio:

| Mode | Target Users | LLM Backend | Infrastructure |
|------|-------------|-------------|----------------|
| **Local** | Individual developers | Ollama | Desktop app |
| **Team** | Companies, self-hosted | vLLM/TGI | Docker Compose |
| **Cloud** | Paying customers | vLLM + cloud APIs | Kubernetes |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLOUMASK CLOUD                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│  │   Frontend  │   │   API       │   │   Worker    │   │   LLM       │    │
│  │   (Svelte)  │◄──│   Gateway   │◄──│   Queue     │◄──│   Service   │    │
│  │   + CDN     │   │   (FastAPI) │   │   (Celery)  │   │   (vLLM)    │    │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
│         │                │                  │                  │           │
│         └────────────────┼──────────────────┼──────────────────┘           │
│                          │                  │                              │
│                    ┌─────┴─────┐      ┌─────┴─────┐                        │
│                    │ PostgreSQL│      │   Redis   │                        │
│                    │   (Users, │      │   (Queue, │                        │
│                    │   Projects)│      │   Cache)  │                        │
│                    └───────────┘      └───────────┘                        │
│                          │                                                 │
│                    ┌─────┴─────┐                                           │
│                    │    S3     │                                           │
│                    │  (Files,  │                                           │
│                    │   Models) │                                           │
│                    └───────────┘                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## LLM Backend Comparison

### Ollama (Local Mode)

**Pros:**
- Simple setup, works offline
- Privacy - data never leaves device
- Free, no API costs

**Cons:**
- ❌ Single-user only
- ❌ No request batching
- ❌ Memory-intensive (loads full model)
- ❌ Not suitable for multi-tenant

**Use for:** Development, offline work, privacy-sensitive local use

### vLLM (Team/Cloud Mode)

**Pros:**
- ✅ High throughput (10-24x faster than naive serving)
- ✅ Continuous batching (handles concurrent requests)
- ✅ PagedAttention (efficient memory)
- ✅ OpenAI-compatible API
- ✅ Supports tensor parallelism (multi-GPU)

**Cons:**
- Requires GPU server
- More complex setup

**Use for:** Self-hosted team deployments, cloud hosting

### Cloud APIs (Cloud Mode)

**Options:**
- **Together AI** - vLLM-based, open models
- **Groq** - Ultra-fast inference
- **Fireworks AI** - Good price/performance
- **Anyscale** - vLLM as a service

**Pros:**
- ✅ No GPU management
- ✅ Auto-scaling
- ✅ Pay per use

**Cons:**
- Data leaves your infrastructure
- Ongoing costs

**Use for:** Quick start cloud hosting, variable load

## Container Architecture (Team/Cloud)

```yaml
# docker-compose.team.yml
services:
  # Frontend - Static files served by nginx
  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - api

  # API Gateway - FastAPI with auth
  api:
    build: ./backend
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://redis:6379
      - LLM_MODE=team
      - LLM_URL=http://vllm:8000/v1
    depends_on:
      - postgres
      - redis
      - vllm

  # LLM Service - vLLM for high throughput
  vllm:
    image: vllm/vllm-openai:latest
    command: --model Qwen/Qwen2.5-14B-Instruct --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Worker Queue - Async job processing
  worker:
    build: ./backend
    command: celery -A backend.tasks worker
    depends_on:
      - redis
      - vllm

  # Databases
  postgres:
    image: postgres:16
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

## Multi-User Data Model

```python
# backend/models/organization.py
from sqlalchemy import Column, ForeignKey, String, Table
from sqlalchemy.orm import relationship

class Organization(Base):
    """Company/team that owns projects."""
    __tablename__ = "organizations"

    id = Column(UUID, primary_key=True)
    name = Column(String, nullable=False)
    slug = Column(String, unique=True)
    plan = Column(String, default="free")  # free, team, enterprise

    members = relationship("OrganizationMember", back_populates="org")
    projects = relationship("Project", back_populates="org")


class OrganizationMember(Base):
    """User membership in organization."""
    __tablename__ = "organization_members"

    user_id = Column(UUID, ForeignKey("users.id"))
    org_id = Column(UUID, ForeignKey("organizations.id"))
    role = Column(String, default="member")  # owner, admin, member, viewer

    user = relationship("User")
    org = relationship("Organization")


class Project(Base):
    """CV annotation project."""
    __tablename__ = "projects"

    id = Column(UUID, primary_key=True)
    org_id = Column(UUID, ForeignKey("organizations.id"))
    name = Column(String, nullable=False)

    # Project settings
    config = Column(JSONB)  # Pipeline config, model settings

    org = relationship("Organization")
    tasks = relationship("Task")


class Task(Base):
    """Single annotation task (image/video/pointcloud)."""
    __tablename__ = "tasks"

    id = Column(UUID, primary_key=True)
    project_id = Column(UUID, ForeignKey("projects.id"))
    data = Column(JSONB)  # File paths, metadata
    annotations = Column(JSONB)  # User annotations

    # Assignment
    assigned_to = Column(UUID, ForeignKey("users.id"), nullable=True)
    status = Column(String, default="pending")  # pending, in_progress, complete
```

## LLM Provider Abstraction

To support all modes, abstract the LLM provider:

```python
# backend/agent/llm/base.py
from abc import ABC, abstractmethod
from enum import Enum

class DeploymentMode(Enum):
    LOCAL = "local"    # Ollama
    TEAM = "team"      # vLLM self-hosted
    CLOUD = "cloud"    # vLLM cloud / API providers

class LLMProvider(ABC):
    """Abstract LLM provider for all deployment modes."""

    @abstractmethod
    async def chat(self, messages: list) -> str:
        """Send messages, get response."""
        pass

    @abstractmethod
    async def stream(self, messages: list) -> AsyncIterator[str]:
        """Stream response tokens."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check provider availability."""
        pass


def create_provider(mode: DeploymentMode, config: dict) -> LLMProvider:
    """Factory to create appropriate provider."""
    match mode:
        case DeploymentMode.LOCAL:
            from backend.agent.llm.ollama import OllamaProvider
            return OllamaProvider(config)
        case DeploymentMode.TEAM | DeploymentMode.CLOUD:
            from backend.agent.llm.vllm import VLLMProvider
            return VLLMProvider(config)
```

## Migration Path

### Phase 1: Current (Local Only)
- ✅ Ollama for LLM
- ✅ SQLite/local files
- ✅ Single user

### Phase 2: Team Mode (Next)
- Add PostgreSQL support
- Add vLLM container option
- Add basic auth (organization + users)
- Docker Compose deployment

### Phase 3: Cloud Mode (Future)
- Kubernetes manifests
- S3/GCS storage
- Stripe billing integration
- Multi-region support

## Recommendation

**For your immediate needs:**

1. **Keep Ollama for local mode** - It works well for single-user
2. **Add vLLM option for team/cloud** - Use OpenAI-compatible API
3. **Abstract the provider interface** - So same agent works with both
4. **Add PostgreSQL for multi-user** - Replace SQLite when needed

**Don't:**
- ❌ Use Ollama for cloud hosting (not efficient)
- ❌ Build multi-user features until you need them
- ❌ Over-engineer before validating product-market fit

## Related: Label Studio Architecture

Label Studio uses:
- Django backend (PostgreSQL)
- React frontend
- ML backend as separate service (pluggable)
- Redis for task queue
- S3 for storage

Key lesson: They keep ML backends **pluggable** - users can connect their own models. Consider similar for Cloumask.
