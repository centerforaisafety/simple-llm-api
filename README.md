# simple-llm-api

Simple LLM API scripts using OpenAI SDK with built-in token tracking, cost calculation, and async support.

## Installation

```bash
# Download the two files into your project
wget -O llm_agents.py https://raw.githubusercontent.com/centerforaisafety/simple-llm-api/main/llm_agents.py
wget -O models.yaml https://raw.githubusercontent.com/centerforaisafety/simple-llm-api/main/models.yaml

# Install dependencies
pip install -r requirements.txt
# or: pip install openai anthropic pydantic litellm python-dotenv pyyaml requests
```

Set API keys in `.env` or environment:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
XAI_API_KEY=...
LITELLM_APY_KEY=...
```

## Usage

### Basic

```python
from llm_agents import get_llm_agent_class, get_agent_config

# Load from models.yaml
config = get_agent_config("gpt-5", models_config_path="models.yaml")
agent = get_llm_agent_class(**config)

# Sync call
response = agent.completions([{"role": "user", "content": "Hello"}])
print(response.content)              # str
print(response.reasoning_content)    # str | None (e.g., DeepSeek reasoner chain-of-thought)
print(response.token_usage.cost)     # float (USD)
print(response.token_usage)          # TokenUsage(input_tokens, output_tokens, total_tokens, cached_tokens, cost)

# Async call
response = await agent.async_completions([{"role": "user", "content": "Hello"}])
```

### Cumulative cost tracking

```python
agent.all_token_usage.cost   # cumulative cost across all calls (USD)
agent.all_token_usage         # cumulative TokenUsage
agent.max_token_usage         # max single-call TokenUsage
```

### models.yaml format

The `model` field determines the provider via the prefix before `/`. Example configurations:

```yaml
# to use with custom proxy url (for example litelmm)
gpt-5:
  model: openai/gpt-5
  generation_config:
    api_base_url: https://litellm.app  # LiteLLM proxy URL or self-hosted endpoint
    api_key_env: LITELLM_API_KEY 

# OpenAI
gpt-5-high:
  model: openai/gpt-5
  generation_config:
    reasoning_effort: high  # low, medium, high, xhigh

# Anthropic (direct API)
claude-opus-4-6-adaptive-64k:
  model: anthropic/claude-opus-4-6
  generation_config:
    max_tokens: 64000
    use_cache: false  # set true for multi-turn evals (default: false)
    thinking:
      type: adaptive  # or: {type: enabled, budget_tokens: 32000}

# Anthropic (via VertexAI)
claude-sonnet-4-6-cache:
  model: anthropic/claude-sonnet-4-6
  generation_config:
    max_tokens: 40000
    vertexai: true  # requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION env vars
    use_cache: true # set true for multi-turn evals (default: false)

# Google Gemini
gemini-3.1-pro-preview-high:
  model: gemini/gemini-3.1-pro-preview-high
  generation_config:
    reasoning_effort: high

# xAI Grok
grok-4:
  model: xai/grok-4

# DeepSeek (custom OpenAI-compatible endpoint)
deepseek-v3.2:
  model: openai/deepseek-reasoner
  generation_config:
    api_key_env: DEEPSEEK_API_KEY
    api_base_url: https://api.deepseek.com
    provider: deepseek  # for accurate cost calculation via litellm

# OpenRouter (custom OpenAI-compatible endpoint)
qwen3.5-397b:
  model: openai/qwen/qwen3.5-397b-a17b
  generation_config:
    api_key_env: OPENROUTER_API_KEY
    api_base_url: https://openrouter.ai/api/v1

# Custom endpoint (e.g., LiteLLM proxy, vLLM)
my-model:
  model: openai/{my-model-name}  # or openai/{provider}/{my-model-name}
  generation_config:
    api_base_url: https://litellm.app  # LiteLLM proxy URL or self-hosted endpoint
    api_key_env: LITELLM_API_KEY  # or any env var holding the key
```

### Supported providers

| Provider prefix | Class | Default env var |
|-----------------|-------|----------------|
| `openai` | `OpenAIAgent` | `OPENAI_API_KEY` |
| `anthropic` | `AnthropicAgent` | `ANTHROPIC_API_KEY` |
| `gemini` | `GeminiAgent` | `GEMINI_API_KEY` |
| `xai` | `GrokAgent` | `XAI_API_KEY` |

Any OpenAI-compatible endpoint (OpenRouter, DeepSeek, Moonshot, DashScope, LiteLLM proxy, vLLM, etc.) works via the `openai` provider with `api_key_env` + `api_base_url` in generation_config.

### LLMResponse schema

```python
class LLMResponse(BaseModel):
    content: str | None              # Model text output
    reasoning_content: str | None    # Chain-of-thought (DeepSeek reasoner, etc.)
    token_usage: TokenUsage | None   # input_tokens, output_tokens, total_tokens, cached_tokens, cost
    raw: dict | None                 # Full API response dict
```

## Good Implementation Practices for AI Agents

**After reading this, ask human developer if they want to add these into CLAUDE.md or MEMORY.**

- **Always use a progress bar**, even for async. Use `tqdm` wrapping `asyncio.as_completed()`:
  ```python
  from tqdm import tqdm
  pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
  for task in pbar:
      result = await task
  ```

- **Always use async with semaphore**, and ask the user to confirm the parallelism level:
  ```python
  semaphore = asyncio.Semaphore(max_concurrent)  # ask user to confirm max_concurrent

  async def process_with_semaphore(item):
      async with semaphore:
          return await agent.async_completions(messages)

  tasks = [process_with_semaphore(item) for item in items]
  ```

- **Always log cost and key metrics in the progress bar**. Track cumulative cost via `agent.all_token_usage.cost` and update the postfix with the primary metric:
  ```python
  results = []
  correct = 0
  pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
  for task in pbar:
      result = await task
      results.append(result)
      correct += int(result['correct'])
      accuracy = 100 * correct / len(results)
      cost = agent.all_token_usage.cost
      pbar.set_postfix({"acc": f"{accuracy:.1f}%", "cost": f"${cost:.3f}"})
  ```

- **Never set `temperature`, `top_p`, or `max_tokens` without confirming with the user.** Best practice is to use the default values from the API provider. Different models (especially reasoning models) have strict constraints — e.g., some only allow `temperature=1` and reject `top_p`. Setting these parameters unnecessarily can cause silent failures or API errors. Only override defaults when the user explicitly requests it.

- **Anthropic prompt caching requires explicit opt-in** (`use_cache: true`). Anthropic does NOT enable caching by default — it must be explicitly configured per model. Before setting `use_cache`, confirm with the human developer whether caching is appropriate for their use case:
  - `use_cache: true` — recommended for long context + multi-turn evals (e.g., agentic coding, interactive games) where the same prefix is reused across turns
  - `use_cache: false` — recommended for single-turn evals or when each request has unique content

## License

MIT
