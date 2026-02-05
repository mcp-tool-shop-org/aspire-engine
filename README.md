# ScalarScope

**Evaluative Internalization Training Framework** - A Python engine for training models to internalize scalar evaluations.

## Overview

ScalarScope is a training framework that explores whether language models can learn to internalize evaluative criteria - developing an internal "conscience" that predicts how external evaluators would rate outputs, without needing those evaluators at inference time.

The framework implements the core training loop, token-level scalar feedback, and geometry export for visualization.

## Installation

```bash
# Basic installation
pip install -e .

# With PyTorch support
pip install -e ".[torch]"

# With ONNX runtime (for inference)
pip install -e ".[onnx]"

# Development
pip install -e ".[dev]"
```

## Quick Start

```python
from scalarscope.engine import ScalarScopeEngine, TrainingMetrics
from scalarscope.governor import TokenPool, GovernorConfig

# Configure the token governor
governor_config = GovernorConfig(
    max_tokens_per_cycle=1000,
    budget_strategy="adaptive"
)
pool = TokenPool(governor_config)

# Create the training engine
engine = ScalarScopeEngine(
    model=your_model,
    evaluators=your_evaluators,
    token_pool=pool
)

# Run a training cycle
result = engine.run_cycle(prompt="Your training prompt")
print(f"Metrics: {result.metrics}")
```

## Architecture

```
src/scalarscope/
├── engine/
│   ├── loop.py              # ScalarScopeEngine - core training loop
│   ├── revision.py          # Revision decision logic
│   └── revision_engine.py   # RevisionScalarScopeEngine - with self-correction
├── governor/
│   ├── token_pool.py        # Token budget management
│   └── config.py            # Governor configuration
├── evaluators/
│   ├── base.py              # Evaluator protocol
│   └── scalar_head.py       # Learned scalar predictor
├── export/
│   └── geometry_export.py   # Export trajectory data for visualization
└── critics/
    └── learned_critic.py    # LearnedCritic V1 with logit-derived features
```

## Key Components

### ScalarScopeEngine

The core training loop that:
1. Generates responses from the model
2. Collects scalar feedback from evaluators
3. Updates model parameters based on evaluative signals
4. Tracks geometry for visualization

### RevisionScalarScopeEngine

Extended engine with self-correction capabilities:
- Detects when outputs need revision
- Applies targeted corrections
- Learns from revision patterns

### TokenPool & Governor

Budget management for training:
- Controls token expenditure per cycle
- Adaptive budgeting based on task difficulty
- Prevents runaway token usage

### Geometry Export

Exports training dynamics for visualization in [ScalarScope-Desktop](https://github.com/mcp-tool-shop-org/ScalarScope-Desktop):
- State vector trajectories
- Eigenvalue spectra
- Evaluator geometry

## Scientific Background

ScalarScope explores a key question in AI alignment:

> **Can models internalize evaluative criteria, developing genuine judgment rather than just predicting rewards?**

The framework tests this through:
- **Multiple evaluators** with potentially different criteria
- **Token-level feedback** for fine-grained learning signals
- **Geometry analysis** to detect shared evaluative structure

Key finding: Internalization succeeds when evaluators share a latent evaluative manifold (Path B), and fails when evaluators are orthogonal (Path A).

## Examples

See the `examples/` directory for:
- `basic_training.py` - Simple training loop
- `revision_demo.py` - Self-correction capabilities
- `geometry_export_demo.py` - Exporting for visualization

## Related

- [ScalarScope-Desktop](https://github.com/mcp-tool-shop-org/ScalarScope-Desktop) - .NET MAUI visualization app
- `docs/RESULTS_AND_LIMITATIONS.md` - Experimental results and known limitations

## License

MIT License - See LICENSE file for details.
