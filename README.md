# Sindarin Tensor

A tensor computation and graph neural network library for the [Sindarin](https://github.com/SindarinSDK/sindarin-compiler) programming language. Provides GPU-accelerated tensor operations, GNN architectures (GCN, GraphSAGE, GAT), end-to-end training via ggml-opt, and model persistence — powered by [ggml](https://github.com/ggerganov/ggml).

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

  # Forward pass: graph -> probs, logits, embedding
  var output: GnnOutput = model.forward(graph, false)
  var probs: double[] = output.probs.toDoubles()

  # Save/load model weights
  model.save("model.bin")
  model.load("model.bin")
```

### Training

```sindarin
import "sindarin-pkg-tensor/src/tensor"
import "sindarin-pkg-tensor/src/gnn"

fn main(): void =>
  var model: Gnn = Gnn.create(GnnConfig {
    inputDim: 5, hiddenDim: 64, numActions: 3,
    numLayers: 2, arch: "gat", dropoutRate: 0.1
  })

  # Caller supplies a list of real GraphTensors, one per training sample.
  # All graphs must share numNodes/featureDim; multi-batch runs also
  # require uniform numEdges. See `Gnn.train` docstring.
  var graphs: GraphTensors[] = buildTrainingGraphs()
  var labels: double[] = buildOneHotLabels()     # len == numGraphs * numActions
  var weights: double[] = buildSampleWeights()   # len == numGraphs (use 1.0s for unweighted CE)

  var result: TrainResult = model.train(
    graphs, labels, weights,
    Optimizer.adamw(0.001, 0.01),
    epochs=200, batchSize=16,
    valSplit=0.0, seed=42l)

  println($"final loss: {result.loss}")
```

## Modules

| Module | Import                                    | Description                                        |
|--------|-------------------------------------------|----------------------------------------------------|
| Tensor | `import "sindarin-pkg-tensor/src/tensor"` | Tensor operations, optimizer config, graph tensors, `batchGraphs` |
| GNN    | `import "sindarin-pkg-tensor/src/gnn"`    | Graph neural network layers, full model, training driver |

## API Reference

### Tensor

Opaque handle to a tensor in the ggml computation pool.

| Method | Description |
|---|---|
| **Creation** | |
| `Tensor.zeros(rows, cols)` | Create a zero-initialized tensor |
| `Tensor.fromDoubles(data, rows, cols)` | Create from a row-major double array |
| **Arithmetic (chainable)** | |
| `t.matmul(other)` | Generic matrix multiplication |
| `t.gnnMatmul(weight)` | Weight-layout-aware matmul used by GNN layers (weight stored as `(outputDim, inputDim)`) |
| `t.add(other)` | Element-wise addition |
| `t.scale(s)` | Scalar multiplication |
| **Activations (chainable)** | |
| `t.relu()` | ReLU activation |
| `t.softmax(dim)` | Softmax along dimension |
| `t.dropout(rate, training)` | Dropout regularisation |
| **Normalization** | |
| `t.batchNorm(weight, bias, mean, var, training)` | Batch normalization |
| `t.layerNorm(weight, bias)` | Layer normalization |
| **GNN aggregation** | |
| `t.aggregate(edgeIndex, edgeWeight, mode)` | Sparse message passing (`"sum"`, `"mean"`, `"sum_normalized"`) |
| `t.attentionAggregate(edgeIndex, edgeWeight, attWeight)` | Attention-based aggregation (GAT) |
| `t.meanPool(batchIndex)` | Mean pooling over graph batch |
| **Reduction** | |
| `t.argmax(dim)` | Index of maximum value |
| `t.norm()` | Scalar L2 norm |
| `t.crossEntropy(targets)` | Cross-entropy loss |
| **Data access** | |
| `t.toDoubles()` | Convert to double array |
| `t.shape()` | Get shape as int array |
| `t.dispose()` | Free tensor memory |
| **Initialization** | |
| `t.initKaiming()` | Kaiming uniform initialization |
| `t.initKaimingSeeded(seed)` | Deterministic Kaiming init (same seed → bit-identical weights) |
| **Persistence** | |
| `sn_model_save(params, path)` | Save tensor array to binary file |
| `sn_model_load(path)` | Load tensor array from binary file |

### Optimizer

Configuration for training optimizers.

```sindarin
var opt: Optimizer = Optimizer.adamw(0.001, 0.01)   # lr, weight_decay
var sgd: Optimizer = Optimizer.sgd(0.01)             # lr
```

### GraphTensors

Graph data in tensor form, consumed by `Gnn.forward` and `Gnn.train`.

| Field          | Type     | Description                                               |
|----------------|----------|-----------------------------------------------------------|
| `nodeFeatures` | `Tensor` | `(numNodes, featureDim)` row-major feature matrix         |
| `edgeIndex`    | `Tensor` | `(2, numEdges)` — row 0 sources, row 1 destinations       |
| `edgeWeight`   | `Tensor` | `(1, numEdges)` per-edge weights                          |
| `batchIndex`   | `Tensor` | `(numNodes, 1)` — maps each node to its source graph      |
| `numNodes`     | `int`    | Node count                                                |
| `numEdges`     | `int`    | Edge count                                                |
| `featureDim`   | `int`    | Features per node                                         |

### `batchGraphs(graphs)`

Free function that concatenates a list of `GraphTensors` into one
batched graph. Edge indices are offset so each input graph's edges
still point at the right rows of the concatenated `nodeFeatures`
tensor; `batchIndex` maps every row back to its source graph index.
All input graphs must share `featureDim`; per-graph `numNodes` and
`numEdges` may vary.

```sindarin
var batched: GraphTensors = batchGraphs({graphA, graphB, graphC})
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

`GnnConfig.defaults(inputDim, numActions)` returns a config with
`hiddenDim=128`, `numLayers=3`, `arch="gat"`, `dropoutRate=0.1`.

### GnnLayer

Single message-passing layer supporting three architectures:

- **GCN** — Graph Convolutional Network: transform, aggregate (normalized sum), ReLU, dropout, residual
- **GraphSAGE** — Sample and Aggregate: aggregate (mean), transform, ReLU, dropout, residual
- **GAT** — Graph Attention Network: transform, attention aggregate, ReLU, dropout, residual

### GnnOutput

Returned by `Gnn.forward`.

| Field       | Type     | Description                                     |
|-------------|----------|-------------------------------------------------|
| `probs`     | `Tensor` | Softmax probabilities over actions              |
| `logits`    | `Tensor` | Raw pre-softmax classifier output               |
| `embedding` | `Tensor` | Mean-pooled graph embedding                     |

### TrainResult

Returned by `Gnn.train`. Carries top-line scalars plus the Phase C
diagnostics consumers need to detect Bug B3-style regressions,
diverging loss, optimizer stalls, and garbage-in/garbage-out
conditions without tailing stderr.

| Field              | Type       | Description |
|--------------------|------------|-------------|
| `loss`             | `double`   | **Final-epoch** training loss (tail of `lossCurve`). Semantics changed from "across-epoch average" in v1.0.0. |
| `accuracy`         | `double`   | Training-set accuracy computed by a post-train forward pass over the training graphs |
| `epochs`           | `int`      | Epochs actually executed |
| `lossCurve`        | `double[]` | Per-epoch average loss — length == epochs |
| `gradNormCurve`    | `double[]` | Per-epoch L2 of the parameter delta vector (proxy for optimizer effective step) |
| `paramNormBefore`  | `double[]` | Per-parameter L2 at `train()` entry, in `model.parameters()` order |
| `paramNormAfter`   | `double[]` | Per-parameter L2 at `train()` exit |
| `paramMaxAbsDelta` | `double[]` | Per-parameter max absolute elementwise change from entry to exit |
| `weightSumIn`      | `double`   | Sum of the caller-supplied `weights` array (sanity check: detects silently-dropped weights) |
| `weightVarianceIn` | `double`   | Population variance of the caller-supplied `weights` (zero ⇒ unweighted) |
| `inputMean`        | `double`   | Mean of the padded feature host buffer across the whole training set |
| `inputStd`         | `double`   | Std of the padded feature host buffer across the whole training set |

### Gnn

Full GNN model: message-passing layers + mean-pool readout + 2-layer classification head.

| Method | Description |
|---|---|
| `Gnn.create(config)` | Create a new model with random weights |
| `Gnn.createWithSeed(config, seed)` | Create a deterministic model: two runs with the same seed produce bit-identical initial weights |
| `model.forward(graph, training)` | Run inference, returns `GnnOutput` (probs, logits, embedding) |
| `model.predictBatch(graphs)` | Run inference on a list of graphs in input order; returns `GnnOutput[]`. Naive loop over `forward(g, false)` — callers holding the returned outputs should not `reset()` the pool until finished reading |
| `model.train(graphs, labels, weights, optimizer, epochs, batchSize, valSplit, seed)` | End-to-end training on a list of `GraphTensors`. See topology precondition below and the docstring in `gnn.sn`. |
| `model.parameters()` | Collect all trainable tensors |
| `model.save(path)` | Save model weights to disk |
| `model.load(path)` | Restore model weights from disk in-place |
| `model.reset()` | Free all tensor pool slots |
| `model.checkpoint()` | Snapshot the current pool high-water mark |
| `model.restore(cp)` | Release pool slots allocated since a checkpoint |

#### `Gnn.train` shape precondition

The only shape requirement is that every graph in the training set
shares the same `featureDim`. Per-graph `numNodes` and `numEdges`
can vary freely within one `train()` call. Internally each sample
is padded to `maxNodes = max(graphs[i].numNodes)` and the dense
adjacency / pool matrix is rebuilt per batch via the per-batch upload
registry. See `docs/issues/heterogeneous-graph-batching.md` for the
long-form architectural write-up.

`weights[i]` is the per-sample loss multiplier. Pass all-1.0s for
standard unweighted cross-entropy, or per-sample rewards / advantages
for policy-gradient style training (see `tests/test_reward_weighted_policy.sn`
for the existence proof).

Training is deterministic for a given `seed`: the across-batch shuffle
is seeded from `Random.createWithSeed(seed)`, and the underlying ggml
optimizer is driven with the caller-supplied AdamW / SGD hyperparameters.
Two `Gnn.createWithSeed(config, seed)` + `train(..., seed)` runs on
identical data produce bit-identical forward outputs to 1e-6 tolerance
(regression-guarded by `tests/test_crusher_policy_e2e.sn` and
`tests/test_save_load_roundtrip_under_train.sn`).

### `distributionDivergence(a, b, kind)`

Free function in `src/tensor.sn` that compares two discrete probability
distributions. Supports `kind ∈ {"l1", "l2", "kl", "js"}`. `kl` uses
natural log with an epsilon floor on `b[i]`; `js` is the symmetric
Jensen-Shannon divergence (natural log, so bounded above by `ln(2)`).

```sindarin
var jsd: double = distributionDivergence(
  model.forward(stateA, false).probs.toDoubles(),
  model.forward(stateB, false).probs.toDoubles(),
  "js")
```

The canonical consumer is a periodic canonical-pair probe that reports
the JSD between two fixed graphs' predictions as a learning-progress
signal. Covered by `tests/test_predict_batch_and_divergence.sn`.

### Training metric callback

For continuous metric streaming into a consumer-side metrics store
(e.g. skynet's Postgres-backed dashboard), register a callback via
`sn_graph_set_train_metric_callback`. The package emits structured
`(name, value, labels)` tuples per epoch and once more at
end-of-train. `labels` is a `StringField[]` where each entry is a
`{key, value}` pair — labels follow the Prometheus-style convention
so consumers can filter and group metrics by dimension.

```sindarin
var cb: fn(str, double, StringField[]): void =
  fn(name: str, value: double, labels: StringField[]): void =>
    # forward into your MetricsClient, log, etc.
    var labelStr: str = ""
    for var i: int = 0; i < len(labels); i++ =>
      if i > 0 =>
        labelStr = labelStr + ","
      labelStr = labelStr + labels[i].key + "=" + labels[i].value
    println($"{name}[{labelStr}] = {value}")
sn_graph_set_train_metric_callback(cb)

var result: TrainResult = model.train(graphs, labels, weights, opt, ...)
# cb was invoked:
#   - per epoch with ("train_loss", v, [{epoch, "E"}])
#     and ("grad_norm_l2", v, [{epoch, "E"}])
#   - once at end with ("weight_sum_in", v, []), ("weight_variance_in", v, []),
#     ("input_mean", v, []), ("input_std", v, []), ("accuracy", v, [])
#   - per layer param with ("param_norm_before", v, [{layer, "L"}, {kind, "K"}])
#     where K ∈ {weight, bias, attSrc, attDst}
#   - per classifier param with ("param_norm_before", v, [{kind, "classW1|classB1|classW2|classB2"}])
#   - same set for "param_norm_after" and "param_max_abs_delta"

sn_graph_clear_train_metric_callback()
```

`StringField` is a small local struct in `src/tensor.sn` with two
fields: `key: str` and `value: str`. The callback is deep-copied on
registration and survives across multiple `train()` calls until
explicitly cleared. Covered by `tests/test_train_metric_callback.sn`.

## Backend

Tensor operations run on the [ggml](https://github.com/ggerganov/ggml) backend:

- **CPU**: SIMD-accelerated (NEON on ARM, AVX on x86)
- **GPU**: Auto-detected via `ggml_backend_init_best()` (CUDA, Metal, Vulkan)
- **Format**: Host-side float32 arrays in a global tensor pool
- **Persistence**: Binary format with magic `SNTN`, stores shape + data per tensor
- **Fork**: Pinned to `RealOrko/ggml@master` via a vcpkg overlay port; see `docs/issues/ggml-issue.md` for the backward-pass contiguity patches that the training path depends on.

## Dependencies

- `sindarin-pkg-sdk` (transitive, for core types and `Random`)
- `ggml`, `ggml-base`, `ggml-cpu` (built from the overlay port under `vcpkg-overlay/ggml/`)

## License

This package is licensed under the [MIT License](LICENSE).

The vendored [ggml](https://github.com/ggerganov/ggml) library is also MIT-licensed. See `vendor/ggml/LICENSE` for details.
