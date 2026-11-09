import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
c1,c2,x1,x2=6.,2.,-2.,2.
def u_exact(x,t):
    z1=x-c1*t-x1; z2=x-c2*t-x2
    s1,s2=jnp.sqrt(c1),jnp.sqrt(c2)
    num=2*(c1-c2)*(c1*jnp.cosh(s2*z2/2)**2+c2*jnp.sinh(s1*z1/2)**2)
    den=((s1-s2)*jnp.cosh((s1*z1+s2*z2)/2)+(s1+s2)*jnp.cosh((s1*z1-s2*z2)/2))**2
    return num/den
