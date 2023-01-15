---
title: "Probability and Statistics" 

subtitle: ""
date: 2020-01-04
author: "Nadhir Hassen"
draft: false
tags:
  - Probability
  - Statistics
  - Python
categories:
  - Probability
  - Statistics
  - Python
layout: single
links:
- icon: box
  icon_pack: fas
  name: MTH2303
  url: https://www.polymtl.ca/programmes/cours/probabilite-et-statistique
# - icon: comment
#   icon_pack: fas
#   name: related talk
#   url: "https://github.com/SciML/DiffEqFlux.jl"
# - icon: chart-bar
#   icon_pack: fas
#   name: tidytuesday databases on notion
#   url: tiny.cc/notion-dataviz
---


## Program overview

### Part I
- Axioms, conditional probability, Bayes rule, combinatorial analysis.
- Random variables: functions of distribution, mass and density, expectation and variance. 
- Discrete and continuous probability laws. 
- Reliability. 
- Random vectors: correlation, multinormal distribution, central limit theorem
- Stochastic processes: Markov chains, Poisson process, Brownian motion.
- Descriptive statistics: diagrams, calculation of characteristics. 
- Sampling distributions: estimate, mean squared error, confidence intervals. - Hypothesis tests: parametric tests and fit test. Simple linear regression.

### Part II

- Discrete time and continuous time Markov chains. 
- Birth and death process. 
- Brownian movement. 
-Dissemination process. 
-Poisson process. 
- Non-homogeneous Poisson processes and other generalizations. 
- Renewal process. 
- Queues with one and with multiple servers.




<!-- ## Rabbit population 
Imagine that some rabbits make their way onto an island that doesn't have any predators. We intially have N rabbits and after a month they make K more. After another month, those N+K rabbits make L rabbits and we observe that $\\frac{N+k}{N} = \\frac{L}{N+K}$, that is, the number of new-born rabbits is proportional to the number of rabbits currently on the island. If we denote time with the variable t, we've observed the following relationship,
    $$
\frac{\\partial N(t)}{\\partial t} = k N(t),
    $$
that is, the rate of change of the population is proportional to the population. You may recognise this as the continuous version of the gemoetric progression $x_n = q x_{n-1}$. This equation is simple enough such that we can solve it [analytically](https://www.mathsisfun.com/calculus/differential-equations.html) and obtain an explicit representation of $N(t)=Ce^{kt}$ for some value $C$. Most commonly, though, it is either very difficult or impossible to find an explicit solution for an equation of this kind, for example it is unclear how to solve (if it is possible at all),
    $$
    \left(\frac{\partial N(t)}{\partial t}\right)^3 + y^2 = N(t) y,
    $$
without using some advanced methods. Even if we do not knowing the exact representation, we can still do interesting things with these equations. For example, we can re-arrange and start from some initial value N(0) and (approximately) simulate how these change in time by iteratively applying the below equation for some small time difference $\\Delta t$,
    $$
    N(t  \Delta t) = N(t) + \sqrt[3]{N(t) y - y^2}\Delta t.
    $$
This is known as the [Euler method](https://en.wikipedia.org/wiki/Euler_method) and while it doesn't give great results due to the accumulation of errors, it shows how we can avoid requiring an explicit representation of N(t).


## Making the ODE Neural

Looking at the previous section, we are inspired to ask ourselves the question what happens if we tried to model the derivative (with respect to time $t$) of the function $z(x)$ taking our inputs $x$ into our outputs $y$ with a neural network?. That is, we imagine that our function $z$ is some continuous transformation that starts at time $t=0$ at $x$ and arrives at $y$ at time $t=1$ and are interested in how it changes as we vary $t$ from 0 to 1. If we're fitting to data anyway, we'll learn some very complex and inscrutable function, so does it provide any advantages over trying to fit the function $z$ itself? The answer, as you may expect, is yes and we will spend the rest of this tutorial looking at various ways in which this is hepful. 

Firstly, though, let's briefly talk about exactly how we can learn the parameters $\\theta$ of our network $f_{\\theta}$ under this new setting. We will still employ gradient-based optimisation, which means that we need to find the quantity\n",
$$
  \\frac{\\partial L(z(1), y)}{\\partial \\theta}
$$
where $L$ is the loss function (e.g. least squares), and $z(t)$ is the aforementioned continuous process, with $z(0) = x$ and $z(1) = \\hat{y}$, that is, our prediction. Now, we know that $z(T) = z(0) + \\int_0^T f_{\\theta}(z(t), t) dt$, for some $0 <= T <= 1$, this is exactly us using our learnt derivative to find the value at time $t=T$ and is the analogue of running our \"network\" $G$ forward. Notice how we can set $T$ to be any real value, this is why we interpret Neural ODEs as having infinitely many hidden layers. As you may guess at this point, in order to fit our weights, we will need to do the equivalent of back-propagation through these infinite layers as well. This is where the concept of the adjoint state $a_z(t) = \\frac{\\partial L}{\\partial z(t)}$ comes in - this is similar to the error signal $\\delta$ in the normal neural network case. From here on out, with a bit of maths, we find the derivative of this adjoint state\n",
$$
\\frac{\\partial a_z(t)}{\\partial t} = -a_z(t)\\frac{\\partial f_{\\theta}(z(t),t)}{\\partial z(t)}.
$$.
Just like having the derivative of $z(t)$ allowed us to calculate $z(T)$ for any $T$, we can now calculate $a_z(T)$ as well. Note that this computation is \"backwards in time\" - we start from the known quantity $a(1)$ and go back towards $a(T)$. Finally, by similar argument to the above, we can define other adjoints $a_{\\theta}(t)$ and $a_t(t)$ to find each of $\\frac{\\partial L}{\\partial \\theta}$ and $\\frac{\\partial L}{\\partial t}$. Unsuprisingly, we get,
$$
\\frac{\\partial a_{\\theta}}{\\partial t} = -a_z(t)\\frac{\\partial f_{\\theta}(z(t),t)}{\\partial \\theta}, \\\\,
\\frac{\\partial a_t}{\\partial t} = -a_z(t)\\frac{\\partial f_{\\theta}(z(t),t)}{\\partial t},
$$
where again, the first line is reminiscent to how we compute the gradient of $\\theta$ given the error signal $\\delta$ and the current hidden state $h_t = f_{\\theta}(z(t), t)$, and the last line follows the functional form of the other two. One final note is that we know $\\frac{\\partial L}{\\partial t}$ at time $t=1$ exactly (it is $a_z(1)f_{\\theta}(z(1), 1)$).
With the gradients of $L$ with respect to its input parameters known, we can now minimise the function given some data.

More detail on the maths can be found in this [blog post](https://ml.berkeley.edu/blog/posts/neural-odes/#training-odenets)

## Python Implementation
We use PyTorch to define the ODENet. We will go over the implementation from this [repo](https://github.com/msurtsukov/neural-ode) as it is slightly more brief than the one in the original paper. First we modify the code as the following

```python
def ode_solve(z0, t0, t1, f):
    """
    Simplest Euler ODE initial value solver
    """
    h_max = 0.05
    n_steps = math.ceil((abs(t1 - t0)/h_max).max().item())

    h = (t1 - t0)/n_steps
    t = t0
    z = z0

    for i_step in range(n_steps):
        z = z + h * f(z, t)
        t = t + h
    return z
```

We will use the following trick several times from here on. If we want to solve several ODEs (in our case one for $a_z, a_{\theta}, a_t$ each) at the same time, we can concatenate the states of each separate ODE into a single augmented state (let's call that $a_{aug}$), and taking into account the Jacobian matrix, we can find $\frac{\partial a_{aug}(t)}{\partial t}$. This allows us to run an ODE solver on the augmented state and solve for the three variables at the same time. We define a function that performs the computation of the forward pass and the adjoint derivatives first

```python
class ODEF(nn.Module):
    def forward_with_grad(self, z, t, grad_outputs):
        """Compute f and a df/dz, a df/dp, a df/dt"""
        batch_size = z.shape[0]

        out = self.forward(z, t)

        # a_z in the description
        a = grad_outputs
        # Computes a_z [df/dz, df/dt, df/theta] using the augmented adjoint state [a_z, a_t, a_theta]
        adfdz, adfdt, *adfdp = torch.autograd.grad(
            (out,), (z, t) + tuple(self.parameters()), grad_outputs=(a),
            allow_unused=True, retain_graph=True
        )
        # grad method automatically sums gradients for batch items, we have to expand them back 
        if adfdp is not None:
            adfdp = torch.cat([p_grad.flatten() for p_grad in adfdp]).unsqueeze(0)
            adfdp = adfdp.expand(batch_size, -1) / batch_size
        if adfdt is not None:
            adfdt = adfdt.expand(batch_size, 1) / batch_size
        return out, adfdz, adfdt, adfdp

    def flatten_parameters(self):
        p_shapes = []
        flat_parameters = []
        for p in self.parameters():
            p_shapes.append(p.size())
            flat_parameters.append(p.flatten())
        return torch.cat(flat_parameters)
```
Next, we define a function that allows us to repeat the process described above for a series of times $[t_0, t_1, ..., t_N]$. This will come in useful in the next section, where we do sequence modelling.

```python
class ODEAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z0, t, flat_parameters, func):
        assert isinstance(func, ODEF)
        bs, *z_shape = z0.size()
        time_len = t.size(0)

        with torch.no_grad():
            z = torch.zeros(time_len, bs, *z_shape).to(z0)
            z[0] = z0
            for i_t in range(time_len - 1):
                z0 = ode_solve(z0, t[i_t], t[i_t+1], func)
                z[i_t+1] = z0

        ctx.func = func
        ctx.save_for_backward(t, z.clone(), flat_parameters)
        return z

    @staticmethod
    def backward(ctx, dLdz):
        """
        dLdz shape: time_len, batch_size, *z_shape
        """
        func = ctx.func
        t, z, flat_parameters = ctx.saved_tensors
        time_len, bs, *z_shape = z.size()
        n_dim = np.prod(z_shape)
        n_params = flat_parameters.size(0)

        # Dynamics of augmented system to be calculated backwards in time
        def augmented_dynamics(aug_z_i, t_i):
            """
            tensors here are temporal slices
            t_i - is tensor with size: bs, 1
            aug_z_i - is tensor with size: bs, n_dim*2 + n_params + 1
            """
            z_i, a = aug_z_i[:, :n_dim], aug_z_i[:, n_dim:2*n_dim]  # ignore parameters and time

            # Unflatten z and a
            z_i = z_i.view(bs, *z_shape)
            a = a.view(bs, *z_shape)
            with torch.set_grad_enabled(True):
                t_i = t_i.detach().requires_grad_(True)
                z_i = z_i.detach().requires_grad_(True)
                func_eval, adfdz, adfdt, adfdp = func.forward_with_grad(z_i, t_i, grad_outputs=a)  # bs, *z_shape
                adfdz = adfdz.to(z_i) if adfdz is not None else torch.zeros(bs, *z_shape).to(z_i)
                adfdp = adfdp.to(z_i) if adfdp is not None else torch.zeros(bs, n_params).to(z_i)
                adfdt = adfdt.to(z_i) if adfdt is not None else torch.zeros(bs, 1).to(z_i)

            # Flatten f and adfdz
            func_eval = func_eval.view(bs, n_dim)
            adfdz = adfdz.view(bs, n_dim) 
            return torch.cat((func_eval, -adfdz, -adfdp, -adfdt), dim=1)

        dLdz = dLdz.view(time_len, bs, n_dim)  # flatten dLdz for convenience
        with torch.no_grad():
            ## Create placeholders for output gradients
            # Prev computed backwards adjoints to be adjusted by direct gradients
            adj_z = torch.zeros(bs, n_dim).to(dLdz)
            adj_p = torch.zeros(bs, n_params).to(dLdz)
            # In contrast to z and p we need to return gradients for all times
            adj_t = torch.zeros(time_len, bs, 1).to(dLdz)

            for i_t in range(time_len-1, 0, -1):
                z_i = z[i_t]
                t_i = t[i_t]
                f_i = func(z_i, t_i).view(bs, n_dim)

                # Compute direct gradients
                dLdz_i = dLdz[i_t]
                dLdt_i = torch.bmm(torch.transpose(dLdz_i.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

                # Adjusting adjoints with direct gradients
                adj_z += dLdz_i
                adj_t[i_t] = adj_t[i_t] - dLdt_i

                # Pack augmented variable
                aug_z = torch.cat((z_i.view(bs, n_dim), adj_z, torch.zeros(bs, n_params).to(z), adj_t[i_t]), dim=-1)

                # Solve augmented system backwards
                aug_ans = ode_solve(aug_z, t_i, t[i_t-1], augmented_dynamics)

                # Unpack solved backwards augmented system
                adj_z[:] = aug_ans[:, n_dim:2*n_dim]
                adj_p[:] += aug_ans[:, 2*n_dim:2*n_dim + n_params]
                adj_t[i_t-1] = aug_ans[:, 2*n_dim + n_params:]

                del aug_z, aug_ans

            ## Adjust 0 time adjoint with direct gradients
            # Compute direct gradients 
            dLdz_0 = dLdz[0]
            dLdt_0 = torch.bmm(torch.transpose(dLdz_0.unsqueeze(-1), 1, 2), f_i.unsqueeze(-1))[:, 0]

            # Adjust adjoints
            adj_z += dLdz_0
            adj_t[0] = adj_t[0] - dLdt_0
        return adj_z.view(bs, *z_shape), adj_t, adj_p, None
```
Finally, we define an neural network module wrapper of the function for more convenient use
<!-- <span style="background-color: #FFFF00">Marked text</span> -->
