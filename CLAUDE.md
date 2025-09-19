# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the OpenAI Agents SDK repository - a Python framework for building multi-agent workflows. The SDK supports OpenAI Responses and Chat Completions APIs, as well as 100+ other LLMs through LiteLLM.

## Development Commands

All commands use `uv` as the package manager. Run commands via `uv run python ...` for Python scripts.

### Core Development Commands
- **Build/Setup**: `make sync` - Install all dependencies and sync the environment
- **Format code**: `make format` - Auto-format code with ruff 
- **Lint**: `make lint` - Run linting checks
- **Type check**: `make mypy` - Run MyPy type checking
- **Run tests**: `make tests` - Run the full test suite
- **Run single test**: `uv run pytest -s -k <test_name>` 
- **Full check**: `make check` - Runs format-check, lint, mypy, and tests

### Snapshot Testing
- **Fix snapshots**: `make snapshots-fix` - Update existing inline snapshots
- **Create snapshots**: `make snapshots-create` - Create new inline snapshots
- **Coverage**: `make coverage` - Generate test coverage reports

### Documentation
- **Build docs**: `make build-docs` - Build documentation locally
- **Serve docs**: `make serve-docs` - Serve docs locally for development

## Architecture Overview

### Core Components

**Main Package Structure**: `src/agents/` contains the implementation

1. **Agent (`agent.py`)**: Central class representing an AI agent with instructions, tools, guardrails, and handoffs
2. **Runner (`run.py`)**: Orchestrates agent execution with `Runner.run()` and `Runner.run_sync()`
3. **Models (`models/`)**: Abstractions for different LLM providers
   - `interface.py`: Base Model interface  
   - `openai_chatcompletions.py`: OpenAI Chat Completions API
   - `openai_responses.py`: OpenAI Responses API
   - `manual.py`: Manual model for testing without API keys
   - `multi_provider.py`: Multi-provider support
4. **Tools (`tool.py`)**: Function tools, computer tools, file search, code interpreter, etc.
5. **Handoffs (`handoffs.py`)**: Agent-to-agent control transfer mechanism
6. **Guardrails (`guardrail.py`)**: Input/output validation and safety checks
7. **Sessions (`memory/`)**: Conversation history management
8. **Tracing (`tracing/`)**: Built-in run tracking and debugging

### Key Concepts

- **Agent Loop**: Runs until final output, handling tool calls and handoffs
- **Handoffs**: Specialized tool calls for transferring control between agents
- **Sessions**: Automatic conversation history across agent runs
- **Tracing**: Built-in tracking for debugging and optimization
- **Model Providers**: Pluggable LLM backends (OpenAI, LiteLLM, etc.)

### Examples Directory Structure
- `examples/basic/`: Simple agent examples and tools
- `examples/agent_patterns/`: Advanced patterns (routing, parallelization, etc.)
- `examples/customer_service/`, `examples/research_bot/`: Complete applications
- `examples/mcp/`: Model Context Protocol integration examples
- `examples/realtime/`: Real-time voice/audio examples

## Code Style & Conventions

- **Formatting**: Uses `ruff format` with 100 character line length
- **Imports**: Use `ruff` for import sorting, `agents` is first-party
- **Type hints**: Strict typing with MyPy
- **Comments**: Write as full sentences ending with periods
- **Python version**: Supports 3.9+, target version is 3.9

## Testing

- **Test location**: `tests/` directory with comprehensive coverage
- **Snapshot tests**: Uses `inline-snapshot` for test validation
- **Async testing**: Uses pytest-asyncio for async test support
- **Running**: Always run full test suite before committing

## Development Setup Notes

- **Virtual environment**: Use `uv` for dependency management
- **API Keys**: Set `OPENAI_API_KEY` environment variable for testing
- **Manual testing**: Use `ManualModel()` for testing without API keys
- **Dependencies**: See `pyproject.toml` for full dependency list

## Important Files to Read

- `AGENTS.md`: Main contributor guidelines and workflow
- `tests/README.md`: Testing guidelines and snapshot instructions
- `examples/`: Comprehensive examples for all major features
- `docs/`: Full documentation in markdown format

Read the AGENTS.md file for detailed contributor instructions.