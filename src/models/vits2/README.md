# Monotonic Alignのコンパイル
```
cd src/models/core/monotonic_align/

python setup.py build_ext --inplace
```

# plmodule to pytorch module

学習したpytorch lightning moduleを、pytorch module, onnxに変換

```
python ./src/models/vits/plmodule_to_pytorch_module.py \
 --checkpoint input_checkpoint.ckpt \
 --output ./checkpoints/vits_best_model.pth
```

onnxモデルの最適化

```
onnxsim ./checkpoints/vits_best_model.onnx ./checkpoints/vits_best_model.sim.onnx

Simplifying...
Finish! Here is the difference:
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃                  ┃ Original Model ┃ Simplified Model ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ Add              │ 334            │ 310              │
│ And              │ 3              │ 3                │
│ Cast             │ 96             │ 10               │
│ Ceil             │ 1              │ 1                │
│ Clip             │ 1              │ 1                │
│ Concat           │ 119            │ 44               │
│ Constant         │ 2171           │ 528              │
│ ConstantOfShape  │ 102            │ 22               │
│ Conv             │ 164            │ 164              │
│ ConvTranspose    │ 4              │ 4                │
│ CumSum           │ 7              │ 7                │
│ Div              │ 201            │ 89               │
│ Equal            │ 60             │ 18               │
│ Erf              │ 24             │ 24               │
│ Exp              │ 3              │ 2                │
│ Expand           │ 90             │ 54               │
│ Gather           │ 155            │ 93               │
│ GatherElements   │ 21             │ 21               │
│ GatherND         │ 15             │ 15               │
│ GreaterOrEqual   │ 6              │ 6                │
│ LeakyRelu        │ 69             │ 69               │
│ Less             │ 2              │ 2                │
│ LessOrEqual      │ 3              │ 3                │
│ MatMul           │ 50             │ 38               │
│ Mul              │ 402            │ 180              │
│ Neg              │ 7              │ 6                │
│ NonZero          │ 21             │ 6                │
│ Not              │ 3              │ 3                │
│ Pad              │ 10             │ 10               │
│ Pow              │ 40             │ 40               │
│ RandomNormalLike │ 2              │ 2                │
│ Range            │ 38             │ 20               │
│ ReduceL2         │ 112            │ 0                │
│ ReduceMax        │ 1              │ 1                │
│ ReduceMean       │ 74             │ 74               │
│ ReduceSum        │ 4              │ 4                │
│ Relu             │ 6              │ 6                │
│ Reshape          │ 133            │ 83               │
│ ScatterND        │ 30             │ 30               │
│ Shape            │ 282            │ 73               │
│ Sigmoid          │ 16             │ 16               │
│ Slice            │ 164            │ 130              │
│ Softmax          │ 12             │ 12               │
│ Softplus         │ 3              │ 3                │
│ Split            │ 9              │ 15               │
│ Sqrt             │ 40             │ 40               │
│ Squeeze          │ 7              │ 7                │
│ Sub              │ 84             │ 71               │
│ Tanh             │ 17             │ 17               │
│ Transpose        │ 168            │ 113              │
│ Unsqueeze        │ 294            │ 89               │
│ Where            │ 72             │ 30               │
│ Model Size       │ 120.3MiB       │ 119.8MiB         │
└──────────────────┴────────────────┴──────────────────┘

```

# Onnxの推論

```
python ./vits/models/onnx_infer.py -h
usage: onnx_infer.py [-h] [--checkopint CHECKOPINT] [--text TEXT] [--output OUTPUT] [--sr SR]

options:
  --checkopint CHECKOPINT
  --text TEXT
  --output OUTPUT
  --sr SR
```