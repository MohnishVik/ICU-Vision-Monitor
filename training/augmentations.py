"""
augmentations.py — training-time augmentations
  Time-warp: ±10% temporal stretch/compress
  Amplitude scale: random gain in [0.8, 1.2]
  Additive noise: Gaussian sigma in [0, 0.02]
  RR channel dropout: zero one of the two input channels (flow or depth)
  Pain AU dropout: randomly mask individual AU channels for occlusion robustness
"""
# TODO: implement
