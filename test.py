import jax
import jax.numpy as jnp
import numpy as np

key = jax.random.PRNGKey(0)

print(jax.random.normal(key))
print(jax.random.normal(key))

key = jax.random.PRNGKey(0)

key, k1 = jax.random.split(key)
x = jax.random.normal(k1, (3,))

key, k2 = jax.random.split(key)
y = jax.random.normal(k2, (3,))

print(x)  # random numbers
print(y)  # different random numbers

set_seed(42)
print(np.random.randn())
print(np.random.randn())


'''561.64
687.16
887.77
236.79
288.92
385.65
180.67
267.85
250.02
349.21
395.26
267.48
274.95
281.06
339.90
254.36
336.60
299.31
148.35
137.09
190.18
194.96
163.18
147.76
981.06
1047.67
1009.62'''
#=========
#10,564.47
#=========


