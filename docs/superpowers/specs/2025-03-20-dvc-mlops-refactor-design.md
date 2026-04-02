# LiquidSearcher DVC MLOps Refactor Design Specification

**Date**: March 20, 2026  
**Version**: 1.0  
**Status**: Approved for Implementation

## Project Overview

Refactor the existing LiquidSearcher dual-encoder contrastive learning project to implement modern MLOps practices with DVC experiment tracking while preserving all existing functionality. The project demonstrates professional ML engineering standards suitable for portfolio showcase and future production deployment.

## Design Goals

### Primary Objectives
1. **Self-Understanding**: Clean, well-organized structure for better project comprehension
2. **Production Deployment**: Industry-standard MLOps organization ready for deployment
3. **Reproducibility**: Full experiment tracking and versioning for portfolio publication
4. **DVC Integration**: Local-first experiment tracking with optional DagsHub integration

### Design Principles
- **Single Source of Truth**: All parameters in `params.yaml` (DVC standard)
- **Behavior Preservation**: No changes to existing model functionality
- **Modular Pipeline**: Independent, reusable DVC stages
- **Comprehensive Evaluation**: Extensive multi-dimensional model evaluation (prepared for later)
- **Portfolio-Ready**: Professional structure demonstrating ML engineering competency

## System Architecture

### Core Components

#### 1. Configuration Management
- **Single `params.yaml`**: Central configuration following DVC best practices
- **Hierarchical structure**: Organized sections for data, model, training, evaluation
- **DVC parameter tracking**: Automatic versioning of all parameter changes
- **Environment variables**: Support for credentials and environment-specific configs

#### 2. DVC Pipeline Architecture
```
Data Ingestion → Feature Engineering → Model Training → [Evaluation Stages] → Reporting
```

**Core Pipeline Stages**:
- `data_ingestion`: WRDS data collection and validation
- `feature_engineering`: Financial feature computation and preprocessing  
- `training`: Dual-encoder model training with DVCLive integration
- `evaluation_*`: Multiple evaluation stages (prepared for later implementation)

#### 3. Experiment Tracking
- **DVCLive Integration**: Automatic metric logging during training
- **Real-time Monitoring**: Training progress, loss curves, custom metrics
- **Experiment Comparison**: Tabular comparison of hyperparameter variations
- **Model Versioning**: Automatic checkpoint tracking and versioning

#### 4. Data Management
- **DVC Data Versioning**: Track large datasets and model files
- **Structured Storage**: Organized raw/processed/embeddings hierarchy
- **Cache Management**: Efficient data pipeline caching
- **Remote Storage**: Optional S3/DagsHub integration for sharing

## Detailed Design

### Project Structure
```
LiquidSearcher/
├── params.yaml                    # Single configuration source
├── dvc.yaml                      # Pipeline definition
├── dvclive/                      # Experiment tracking results
├── src/                          # Production-organized source code
│   ├── data/                     # Data pipeline (existing code + minor changes)
│   ├── models/                   # Model architectures (existing code)
│   ├── training/                 # Training pipeline (add DVCLive integration)
│   ├── evaluation/               # Evaluation framework (prepared structure)
│   ├── inference/                # Production inference pipeline
│   └── utils/                    # Utilities (existing code)
├── pipelines/                    # DVC pipeline implementations
├── data/                         # DVC-tracked data
├── models/                       # DVC-tracked models  
├── results/                      # Evaluation results (prepared structure)
├── docs/                         # Comprehensive documentation
├── notebooks/                    # Analysis notebooks
└── reports/                      # Generated reports
```

### Configuration Architecture (`params.yaml`)
```yaml
data:
  wrds:                          # WRDS connection and queries
  preprocessing:                 # Feature engineering parameters
  splits:                        # Train/validation/test splits

model:
  dual_encoder:                  # Model architecture parameters
    temporal_encoder:            # BiMT-TCN configuration
    tabular_encoder:             # TabMixer configuration  
    projection_dim: 128
  loss:                          # InfoNCE loss parameters

training:
  optimizer:                     # AdamW configuration
  scheduler:                     # Learning rate scheduling  
  batch_size: 64
  max_epochs: 100
  precision: "16-mixed"

evaluation:                      # Evaluation configuration (prepared)
  stages: [...]                  # Multiple evaluation dimensions
  metrics: [...]                 # Evaluation metrics
  visualization: [...]           # Plotting configurations
```

### DVC Pipeline Design
```yaml
# dvc.yaml
stages:
  data_ingestion:
    cmd: python pipelines/data_ingestion.py
    params: [data.wrds, data.preprocessing]
    outs: [data/raw/]

  feature_engineering:  
    cmd: python pipelines/feature_engineering.py
    params: [data.preprocessing.features]
    deps: [data/raw/]
    outs: [data/processed/features/]

  training:
    cmd: python pipelines/training.py
    params: [model, training, data.splits]
    deps: [data/processed/features/]
    outs: [models/dual_encoder/, dvclive/]
    metrics: [dvclive/metrics.json]

  # Evaluation stages (prepared for later)
  evaluate_embedding_quality: {prepared}
  analyze_similarity: {prepared}
  # ... additional evaluation stages
```

### DVCLive Integration
```python
# Enhanced Lightning module with DVCLive
class DualEncoderModule(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        # Existing training logic
        loss = self.compute_loss(batch)
        
        # DVCLive automatic logging
        self.log('train_loss', loss)
        self.log('alignment_score', alignment_metric)
        self.log('hard_neg_similarity', hard_neg_sim)
        
        return loss

# Training script with DVCLive logger
trainer = pl.Trainer(
    logger=DVCLiveLogger(save_dvc_exp=True),
    max_epochs=params['training']['max_epochs'],
    callbacks=[...]
)
```

## Implementation Strategy

### Phase 1: Core DVC Integration (Immediate)
1. **Parameter Centralization**: Migrate all configurations to `params.yaml`
2. **Basic Pipeline**: Implement core data → training pipeline  
3. **DVCLive Integration**: Add experiment tracking to existing training
4. **Data Versioning**: Add DVC tracking for datasets and models

### Phase 2: Evaluation Framework (Later)
1. **Evaluation Structure**: Implement prepared evaluation modules
2. **Comprehensive Metrics**: Multi-dimensional model evaluation
3. **Visualization Suite**: Extensive plotting and analysis
4. **Reporting Pipeline**: Automated report generation

### Code Migration Approach
- **Preserve Functionality**: No changes to core model behavior
- **Minimal Disruption**: Keep existing code structure where possible
- **Parameter Reading**: Replace argparse with params.yaml loading
- **DVCLive Integration**: Add logging to existing Lightning module

## Technical Specifications

### Dependencies
```yaml
# Additional dependencies for DVC integration
dvc>=3.0.0
dvcive>=3.0.0
pytorch-lightning>=2.5.0
```

### File Organization
- **Source Code**: Production-ready modular organization
- **Configuration**: Single `params.yaml` with hierarchical structure
- **Data**: DVC-tracked with .dvc pointer files
- **Models**: Versioned checkpoints with model registry integration
- **Results**: Structured evaluation output (prepared for implementation)

### Experiment Tracking Features
- **Real-time Metrics**: Training loss, validation metrics, custom scores
- **Hyperparameter Tracking**: Automatic parameter versioning
- **Model Versioning**: Checkpoint tracking and comparison
- **Visualization**: Training curves, metric evolution plots
- **Experiment Comparison**: Tabular comparison of runs

## Quality Assurance

### Testing Strategy
- **Functionality Preservation**: Ensure existing model behavior unchanged
- **Pipeline Testing**: Validate DVC pipeline execution
- **Configuration Validation**: Test parameter loading and validation
- **Integration Testing**: End-to-end pipeline testing

### Validation Criteria
- ✅ Existing model produces identical results with new structure
- ✅ DVCLive captures all training metrics correctly
- ✅ DVC pipeline executes without errors
- ✅ Parameter changes trigger appropriate pipeline re-execution
- ✅ Model versioning works correctly

## Risk Assessment

### Potential Issues
1. **Configuration Migration**: Risk of missing parameters during migration
2. **DVCLive Integration**: Potential conflicts with existing Lightning setup
3. **Data Pipeline**: Changes to data loading could affect training
4. **Dependency Conflicts**: New dependencies might conflict with existing ones

### Mitigation Strategies
1. **Incremental Migration**: Implement changes step-by-step with testing
2. **Backup Strategy**: Maintain working branch during refactoring
3. **Validation Testing**: Comprehensive testing at each migration step
4. **Rollback Plan**: Clear rollback procedures if issues arise

## Success Metrics

### Technical Metrics
- **Pipeline Execution**: All DVC stages execute successfully
- **Experiment Tracking**: DVCLive captures complete experiment data
- **Model Performance**: Training results identical to original implementation
- **Configuration Management**: All parameters managed through single file

### User Experience Metrics  
- **Development Workflow**: Improved experiment management and comparison
- **Documentation Quality**: Clear, comprehensive project documentation
- **Code Organization**: Cleaner, more maintainable code structure
- **Portfolio Readiness**: Professional ML engineering demonstration

## Future Considerations

### Potential Extensions
1. **Model Registry**: Full DVC 3.0 model registry implementation
2. **Deployment Pipeline**: Production deployment automation
3. **Advanced Evaluation**: Complete evaluation framework implementation
4. **Remote Storage**: Enhanced DagsHub/S3 integration
5. **CI/CD Integration**: Automated testing and validation

### Scalability Considerations
- **Distributed Training**: Lightning distributed training compatibility
- **Large Datasets**: Efficient data pipeline for larger WRDS datasets  
- **Model Serving**: Production inference pipeline readiness
- **Monitoring**: Production model monitoring integration

## Conclusion

This design provides a comprehensive plan for refactoring LiquidSearcher into a modern MLOps project while preserving all existing functionality. The approach balances immediate experiment tracking needs with future scalability and production requirements, resulting in a portfolio-worthy demonstration of professional ML engineering practices.

The phased implementation strategy minimizes risk while delivering immediate value through improved experiment tracking and project organization. The prepared evaluation framework ensures future extensibility without requiring additional architectural changes.