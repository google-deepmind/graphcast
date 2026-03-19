import jax
import jax.numpy as jnp
import xarray
from graphcast import samplers_utils
from graphcast import xarray_jax

def generate_noise_levels(key, batch_size, min_noise, max_noise, rho):
    # Sample a uniform random value for each item in the batch.
    uniform_samples = jax.random.uniform(key, shape=(batch_size,), dtype=jnp.float32)
    # Transform the uniform samples using the rho inverse CDF function.
    noise = samplers_utils.rho_inverse_cdf(
        min_value=min_noise,
        max_value=max_noise,
        rho=rho,
        cdf=uniform_samples)
    # Wrap the noise values in an xarray DataArray.
    noise_levels = xarray_jax.DataArray(data=noise, dims=('batch',))
    return noise_levels

def main():
    key = jax.random.PRNGKey(42)  # Initial random key.
    batch_size = 1                # Number of samples per iteration.
    min_noise = 0.02              # Training minimum noise level.
    max_noise = 88              # Training maximum noise level.
    rho = 7.0                     # Training noise schedule parameter.

    # Loop to generate and print noise levels repeatedly.
    for i in range(10):
        key, subkey = jax.random.split(key)
        noise_levels = generate_noise_levels(subkey, batch_size, min_noise, max_noise, rho)
        print(f"Iteration {i+1}: Noise Levels = {noise_levels.data}")

if __name__ == "__main__":
    main()


'''

Iteration 1: Noise Levels = xarray_jax.JaxArrayWrapper(Array([20.09941], dtype=float32))
Iteration 2: Noise Levels = xarray_jax.JaxArrayWrapper(Array([2.2559268], dtype=float32))
Iteration 3: Noise Levels = xarray_jax.JaxArrayWrapper(Array([1.2343705], dtype=float32))
Iteration 4: Noise Levels = xarray_jax.JaxArrayWrapper(Array([22.923538], dtype=float32))
Iteration 5: Noise Levels = xarray_jax.JaxArrayWrapper(Array([49.872597], dtype=float32))
Iteration 6: Noise Levels = xarray_jax.JaxArrayWrapper(Array([29.471828], dtype=float32))
Iteration 7: Noise Levels = xarray_jax.JaxArrayWrapper(Array([0.16446371], dtype=float32))
Iteration 8: Noise Levels = xarray_jax.JaxArrayWrapper(Array([0.99823976], dtype=float32))
Iteration 9: Noise Levels = xarray_jax.JaxArrayWrapper(Array([20.825964], dtype=float32))
Iteration 10: Noise Levels = xarray_jax.JaxArrayWrapper(Array([6.820768], dtype=float32))

'''