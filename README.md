# Seed Creative Swarm

**Ensemble of 3 Seed-mini processes + Seed-pro philosophical judge for divergent creative generation.**

```
                         ┌──────────────────────────────────────────────┐
                         │           CREATIVE SWARM ARCHITECTURE        │
                         └──────────────────────────────────────────────┘
                         
                                    ┌─────────────┐
                                    │   PROMPT    │
                                    └──────┬──────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
              ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
              │Seed-mini  │          │Seed-mini  │          │Seed-mini  │
              │   T=0.7   │          │   T=0.85   │          │   T=1.0   │
              │(cautious) │          │(balanced)  │          │ (wild)    │
              └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
                    │                      │                      │
                    └──────────────────────┼──────────────────────┘
                                           │
                         ┌─────────────────▼─────────────────┐
                         │           SEED-PRO JUDGE           │
                         │  "Profound or Clever?" scoring     │
                         │  Score: 0.0–1.0                    │
                         └─────────────────┬───────────────────┘
                                           │
                                    ┌──────▼──────┐
                                    │   WINNER    │
                                    │  (highest   │
                                    │  pro_score) │
                                    └─────────────┘
```

## Why Temperature Variance?

Temperature controls randomness in LLM generation:

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.7 | Conservative, focused | Reliable, safe outputs |
| 0.85 | Balanced | Default creative work |
| 1.0 | Exploratory, diverse | Maximum divergence |

By running three temperatures in parallel, you get **three different angles on the same prompt** — safe and wild, focused and exploratory — in a single API call round.

## Why Seed-pro as Judge?

Seed-pro acts as a **philosophical quality filter**. It evaluates each generation for:
- **Depth** — Does it reveal something meaningful?
- **Originality** — Is it genuinely novel, not just competent?
- **Utility** — Does it actually solve the problem well?

The `pro_score` weights the ensemble vote, so you don't just get the most coherent output — you get the most **profound** one.

## Installation

```bash
pip install seed-creative-swarm
```

## Quick Start

```python
from seed_swarm import CreativeSwarm

swarm = CreativeSwarm(deepinfra_key="your_deepinfra_key")

# Single generation cycle
result = swarm.generate(
    prompt="Design a fleet-wide failure recovery protocol",
    num_mini=3,
    include_pro=True
)

print(f"Winner: {result['winner']['text'][:100]}")
print(f"Profound score: {result['pro_score']:.2f}")
print(f"Votes: {result['votes']}")
```

## API Reference

### `CreativeSwarm(deepinfra_key: str)`

Initialize the swarm with your DeepInfra API key.

### `swarm.generate(prompt, temperatures, num_mini, include_pro)`

Run one creative swarm cycle.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Input prompt |
| `temperatures` | List[float] | [0.7, 0.85, 1.0] | Temp for each mini |
| `num_mini` | int | 3 | Number of mini processes |
| `include_pro` | bool | True | Run Seed-pro judge |

Returns:
```python
{
    "prompt": str,
    "generations": [{"temperature", "text", "pro_score", "verdict", "pro_reasoning"}, ...],
    "winner": {...},        # Generation with highest pro_score
    "pro_score": float,     # 0.0–1.0
    "votes": {temp: score},
    "timestamp": float
}
```

### `swarm.generate_with_variance(prompt, num_generations, include_pro)`

Run multiple swarm cycles, return the most profound result.

```python
result = swarm.generate_with_variance(
    prompt="Write a haiku about distributed systems",
    num_generations=5
)

print(f"Most profound: {result['most_profound']}")
```

## Examples

### Creative Writing

```python
swarm = CreativeSwarm(deepinfra_key="...")

result = swarm.generate(
    prompt="Write an opening line for a sci-fi novel about time travel",
    temperatures=[0.7, 0.85, 1.0],
    num_mini=3
)
print(result['winner']['text'])
```

### System Design

```python
result = swarm.generate(
    prompt="Design a consensus protocol for a fleet of autonomous vessels",
    temperatures=[0.8, 0.9, 1.0],  # Higher temps for architectural creativity
    num_mini=3
)
```

## License

MIT