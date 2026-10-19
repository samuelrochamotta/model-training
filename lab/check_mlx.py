"""Confere que o MLX enxerga a GPU do Mac e faz uma conta nela."""

import time

import mlx.core as mx

print("dispositivo:", mx.default_device())
a = mx.random.normal((2048, 2048))
t0 = time.time()
c = a @ a
mx.eval(c)
print(f"matmul 2048x2048 em {(time.time() - t0) * 1000:.1f} ms, soma = {c.sum().item():.2f}")
