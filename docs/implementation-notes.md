# Implementation Notes

## Project Status

**Phase**: Scaffolding (initial setup)

The project is in its early stages with basic directory structure and configuration in place:
- Package structure defined in `pyproject.toml`
- Source modules created (`data/`, `features/`, `models/`, `liquidity/`, `pipeline/`, `config/`)
- Testing infrastructure configured (pytest, ruff, mypy)
- Dependencies declared

## Planned Implementation Approach

### Phase 1: Data Layer (Week 1-2)
- Implement WRDS API client
- Build data loaders for CRSP, Compustat, IBES, TAQ
- Define schema validation
- Create data caching mechanism

### Phase 2: Feature Engineering (Week 3-4)
- Implement G1-G4 fundamental features
- Implement G5 temporal features from OHLCV
- Implement G6 risk factor calculations
- Build feature pipeline with transforms

### Phase 3: Model Development (Week 5-7)
- Build Temporal Encoder (TCN/Transformer)
- Build Tabular Encoder (FT-Transformer)
- Implement RankSCL similarity ranking
- Train dual-encoder contrastive model

### Phase 4: Liquidity Filters (Week 8)
- Implement Amihud illiquidity measure
- Implement DArLiQ (depth-adjusted)
- Add bid-ask spread constraints
- Build volume-based filters

### Phase 5: Pipeline Integration (Week 9-10)
- End-to-end inference pipeline
- Query preprocessing
- Candidate ranking and selection
- API endpoint for recommendations

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Polars for data processing | Performance with large datasets (>100k rows), lazy evaluation |
| Dual-encoder architecture | Separate handling of temporal (OHLCV) and tabular (fundamentals) data |
| RankSCL for similarity | Ordinal ranking preserves relative similarity better than distance metrics |
| 7 liquidity filters | Hard constraints ensure recommended stocks meet liquidity thresholds |
| Russell 2000 + S&P 400 universe | Mid/small-cap focus where liquidity risk is most relevant |

## Dependencies

Core:
- `polars`, `pandas` - Data processing
- `numpy`, `scipy` - Numerical operations
- `scikit-learn` - ML utilities

Dev:
- `pytest` - Testing
- `ruff` - Linting/formatting
- `mypy` - Type checking

## Technical Constraints

- Python 3.11+
- WRDS subscription required for data access
- ML fallback for spread estimation if TAQ unavailable
- Data stored in `data/` (not committed to repo)