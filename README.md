# Sindarin Tensor

A tensor computation and graph neural network library for the [Sindarin](https://github.com/SindarinSDK/sindarin-compiler) programming language. Provides GPU-accelerated tensor operations, GNN architectures (GCN, GraphSAGE, GAT), and model persistence — powered by [ggml](https://github.com/ggerganov/ggml).

## Installation

Add the package as a dependency in your `sn.yaml`:

```yaml
dependencies:
- name: sindarin-pkg-tensor
  git: git@github.com:SindarinSDK/sindarin-pkg-tensor.git
  branch: main
```

Then run `sn --install` to fetch the package.

## Quick Start

### Tensor Operations

```sindarin
import "sindarin-pkg-tensor/src/tensor"

fn main(): void =>
  var a: Tensor = Tensor.zeros(3, 4)
  var w: Tensor = Tensor.zeros(4, 2)
  w.initKaiming()

  # Chainable method-style API
  var out: Tensor = a.matmul(w).relu().softmax(1)
  var data: double[] = out.toDoubles()
  println($"output: {len(data)} elements")
```

### Graph Neural Network

```sindarin
import "sindarin-pkg-tensor/src/tensor"
import "sindarin-pkg-tensor/src/gnn"

fn main(): void =>
  # Configure and create a GNN
  var config: GnnConfig = GnnConfig {
    inputDim: 5,
    hiddenDim: 64,
    numActions: 3,
    numLayers: 2,
    arch: "gat",
    dropoutRate: 0.1
  }
  var model: Gnn = Gnn.create(config)

  # Prepare graph data
  var graph: GraphTensors = GraphTensors {
    nodeFeatures: Tensor.fromDoubles(featureData, numNodes, 5),
    edgeIndex: Tensor.fromDoubles(edgeData, 2, numEdges),
    edgeWeight: Tensor.fromDoubles(weightData, 1, numEdges),
    batchIndex: Tensor.fromDoubles(batchData, numNodes, 1),
    numNodes: numNodes,
    numEdges: numEdges,
    featureDim: 5
  }

  # Forward pass: graph -> action probabilities + embedding
  var output: GnnOutput = model.forward(graph, false)
  var probs: double[] = output.probs.toDoubles()

  # Save/load model weights
  model.save("model.bin")
  var loaded: Tensor[] = sn_model_load("model.bin")
```

## Modules

| Module | Import                                    | Description                                        |
|--------|-------------------------------------------|----------------------------------------------------|
| Tensor | `import "sindarin-pkg-tensor/src/tensor"` | Tensor operations, optimizer config, graph tensors |
| GNN    | `import "sindarin-pkg-tensor/src/gnn"`    | Graph neural network layers and models             |

## API Reference

### Tensor

Opaque handle to a tensor in the ggml computation pool.

**Creation:**

| Method                                 | Description                      |
|----------------------------------------|----------------------------------|
| `Tensor.zeros(rows, cols)`             | Create a zero-initialized tensor |
| `Tensor.fromDoubles(data, rows, cols)` | Create from a double array       |

**Arithmetic (chainable):**

| Method           | Description           |
|------------------|-----------------------|
| `t.matmul(other)` | Matrix multiplication |
| `t.add(other)`   | Element-wise addition |
| `t.scale(s)`     | Scalar multiplication |

**Activations (chainable):**

| Method                      | Description             |
|-----------------------------|-------------------------|
| `t.relu()`                  | ReLU activation         |
| `t.softmax(dim)`            | Softmax along dimension |
| `t.dropout(rate, training)` | Dropout regularisation  |

**Normalization:**

| Method                                            | Description         |
|---------------------------------------------------|---------------------|
| `t.batchNorm(weight, bias, mean, var, training)` | Batch normalization |

**GNN Aggregation:**

| Method                                                     | Description                                                    |
|------------------------------------------------------------|----------------------------------------------------------------|
| `t.aggregate(edgeIndex, edgeWeight, mode)`                 | Sparse message passing (`"sum"`, `"mean"`, `"sum_normalized"`) |
| `t.attentionAggregate(edgeIndex, edgeWeight, attWeight)` | Attention-based aggregation (GAT)                              |
| `t.meanPool(batchIndex)`                                   | Mean pooling over graph batch                                  |

**Reduction:**

| Method                     | Description            |
|----------------------------|------------------------|
| `t.argmax(dim)`            | Index of maximum value |
| `t.crossEntropy(targets)` | Cross-entropy loss     |

**Data access:**

| Method         | Description             |
|----------------|-------------------------|
| `t.toDoubles()` | Convert to double array |
| `t.shape()`    | Get shape as int array  |
| `t.dispose()`  | Free tensor memory      |

**Initialization:**

| Method             | Description                     |
|--------------------|---------------------------------|
| `t.initKaiming()` | Kaiming uniform initialization |

**Persistence:**

| Function                      | Description                        |
|-------------------------------|------------------------------------|
| `sn_model_save(params, path)` | Save tensor array to binary file   |
| `sn_model_load(path)`         | Load tensor array from binary file |

### Optimizer

Configuration for training optimizers.

```sindarin
var opt: Optimizer = Optimizer.adamw(0.001, 0.01)   # lr, weight_decay
var sgd: Optimizer = Optimizer.sgd(0.01)             # lr
```

### GnnConfig

Configuration for GNN model architecture.

| Field         | Type   | Description                                  |
|---------------|--------|----------------------------------------------|
| `inputDim`    | int    | Number of input features per node            |
| `hiddenDim`   | int    | Hidden layer dimension                       |
| `numActions`  | int    | Number of output classes/actions              |
| `numLayers`   | int    | Number of message-passing layers             |
| `arch`        | str    | Architecture: `"gcn"`, `"sage"`, or `"gat"` |
| `dropoutRate` | double | Dropout probability (0.0 to 1.0)             |

### GnnLayer

Single message-passing layer supporting three architectures:

- **GCN** — Graph Convolutional Network: transform, aggregate (normalized sum), batch norm, ReLU
- **GraphSAGE** — Sample and Aggregate: aggregate (mean), transform, batch norm, ReLU
- **GAT** — Graph Attention Network: transform, attention aggregate, batch norm, ReLU

### Gnn

Full GNN model: message-passing layers + mean-pool readout + 2-layer classification head.

| Method                           | Description                                             |
|----------------------------------|---------------------------------------------------------|
| `Gnn.create(config)`            | Create a new model with random weights                  |
| `model.forward(graph, training)` | Run inference, returns `GnnOutput` (probs + embedding) |
| `model.parameters()`            | Collect all trainable tensors                           |
| `model.save(path)`              | Save model weights to disk                              |

## Backend

Tensor operations run on the [ggml](https://github.com/ggerganov/ggml) backend:

- **CPU**: SIMD-accelerated (NEON on ARM, AVX on x86)
- **GPU**: Auto-detected via `ggml_backend_init_best()` (CUDA, Metal, Vulkan)
- **Format**: Host-side float32 arrays in a global tensor pool
- **Persistence**: Binary format with magic `SNTN`, stores shape + data per tensor

## Dependencies

- `sindarin-pkg-sdk` (transitive, for core types)
- `ggml`, `ggml-base`, `ggml-cpu` (bundled in `libs/`)
