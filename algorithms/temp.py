import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

class Foo(nn.Module):
    @nn.compact
    def __call__(self, x, train): # type: ignore
        var = self.variable("batch_stats", "test", jax.nn.initializers.ones, key=jax.random.PRNGKey(0), shape=(1,)).value
        var = var + 1 if train else var

        x = nn.Dense(3)(x)
        # only spectral normalize the params of the second Dense layer
        x = nn.SpectralNorm(nn.Dense(4))(x, update_stats=train)
        x = nn.Dense(5)(x)
        return x + var

# init
x = jnp.ones((1, 2))
y = jnp.ones((1, 5))
model = Foo()
variables = model.init(jax.random.PRNGKey(0), x, train=False)
flax.core.freeze(jax.tree_util.tree_map(jnp.shape, variables)) # type: ignore

# train
def train_step(variables, x, y):
    def loss_fn(params):
        logits, updates = model.apply({"params": params, "batch_stats": variables["batch_stats"]}, x, train=True, mutable=['batch_stats'])
        loss = jnp.mean(optax.l2_loss(predictions=logits, targets=y))
        return loss, updates

    (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(variables['params']) 

    return { 
            'params': 
                jax.tree_util.tree_map(
                    lambda p, g: p - 0.1 * g, 
                    variables['params'], 
                    grads
                ), 
            'batch_stats': updates['batch_stats']
            }, loss

for _ in range(10):
    variables, loss = train_step(variables, x, y)
    # print("\n\n",variables,"\n", loss, "\n\n")

# inference / eval
out = model.apply(variables, x, train=False, mutable=["batch_stats"])

print("\n\n", out)
