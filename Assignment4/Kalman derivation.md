# Derivation of matrices for Kalman filter, based on motion model

As Kalman step is already implemented for us in this assignment, all we need to do is choose the motion model (RW, NCV, NCA), and define the following:
- State $\vec{x}$
- Matrix $F$, or the state transition matrix, used for calculating system matrix $\Phi$
- Matrix $L$, used for calculating system covariance matrix $Q$
- Matrix $H$, or the observation matrix
- Matrix $R$, or the observation covariance matrix

Lecture slides (Recursive Bayes Filters, part I) give a taste of what the derivations are like for 1D versions of NCV and RW motion models. There is also one slide for 2D NCV. Assignment slides are incredibly helpful in their own regard, but they don't show us how to derive/define the aforementioned state and matrices for 2D case.


## Notes

Some notations used here are a bit different to lecture slides:
- $T$ corresponds to $\Delta t$ from lecture slides (time between frames/timesteps)
- $\vec{x}$ and $\vec{x_k}$ correspond to $\mathbf{x}$ and $\mathbf{x}_k$ respectively from lecture slides (state)
- $\vec{y_k}$ corresponds to $\mathbf{y}_k$ (observation)
- $\dot{\vec{x}}$ corresponds to $\dot{\mathbf{x}}$ (first derivative of state)
- $\vec{w}$ and $\vec{v}$ corresond to $w$ and $\mathbf{v}_k$ respectively (noise)



# Random Walk

Random walk motion model states that velocity is modeled by a white noise sequence, while acceleration is 0. In other words:
- $\dot{\vec{x}} = \vec{w}$
- $\ddot{\vec{x}} = 0$

We want to define matrices $F$ and $L$ based on the following equation:
```math
\dot{\vec{x}} = F\vec{x} + L\vec{w}
```

While it may seem redundant to start with this equation (since we already have  $\dot{\vec{x}} = \vec{w}$), it will make the derivations more consistent across later motion models. 

We first need to define our state $\vec{x}$. In a random walk model, we only need to keep track of the position ($x, y$ coordinates), so our state $\vec{x}$ should be $\vec{x} = [x,\space y]^T$. The above equation would thus transform to:
```math
\dot{\vec{x}} = 
\begin{bmatrix} 
    \dot{x} \\
    \dot{y} 
\end{bmatrix} = 
F \begin{bmatrix} 
    x \\
    y 
\end{bmatrix} + 
L \begin{bmatrix} 
    w_1 \\
    w_2 
\end{bmatrix}
```

We need our $\dot{x}$ and $\dot{y}$ to be equal to noise $w_1$ and $w_2$ respectively. So, we set matrices $F$ and $L$ "manually" so that it all works out:
```math
\dot{\vec{x}} = 
\begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = 
\underbrace{\begin{bmatrix}
    0 & 0 \\
    0 & 0
\end{bmatrix}}_{F}
\begin{bmatrix} x \\ y \end{bmatrix} 
+ 
\underbrace{\begin{bmatrix}
    1 & 0 \\
    0 & 1
\end{bmatrix}}_{L}
\begin{bmatrix} w_1 \\ w_2 \end{bmatrix}
```

If we multiply the last equation out, we would get exactly what we need:
```math
\dot{\vec{x}} = 
\begin{bmatrix} \dot{x} \\ \dot{y} \end{bmatrix} = 
\begin{bmatrix}
    w_1 \\
    w_2
\end{bmatrix}, \text{i.e.} \space \dot{\vec{x}} = \vec{w}
```

We see that matrices $F$ and $L$ are in a way "trivial", but again, starting off with $\dot{\vec{x}} = F\vec{x} + L\vec{w}$ will make derivations for NCV and NCA motion models more consistent and easy to follow. Starting off with that equation will also make more sense for those two models.

---

In a similar fashion, observation matrix $H$ is defined based on equation
```math
\vec{y_k} = H\vec{x_k} + \vec{v}
```

$\vec{x_k}$ is just the discrete version of our state, while $\vec{v}$ is the noise. $\vec{y_k}$ is our observation, and for all motion models, we want to observe only the position:
```math
\vec{y_k} = \begin{bmatrix}
    x_k^{(measured)} \\
    y_k^{(measured)}
\end{bmatrix} = \begin{bmatrix}
    x_k^{(m)} \\
    y_k^{(m)}
\end{bmatrix}
```

Our state remains the same as for the RW motion model (just discretized), so the equation for $\vec{y_k}$ becomes:
```math
\vec{y_k} = \begin{bmatrix}
    x_k^{(m)} \\
    y_k^{(m)}
\end{bmatrix} =
H \begin{bmatrix}
    x_k \\
    y_k
\end{bmatrix} +
\begin{bmatrix}
    v_1 \\
    v_2
\end{bmatrix}
```

We want to get $x_k^{(m)} = x_k + v_1$, and $y_k^{(m)} = y_k + v_2$, so we set $H$ to be:
```math
H = \begin{bmatrix}
    1 & 0 \\
    0 & 1
\end{bmatrix}
```
which gives us what we want.



# Nearly Constant Velocity

Nearly constant velocity motion model states that acceleration (derivative of velocity) is modeled by a white noise sequence, meaning our velocity is not constant. In other words:
- $\ddot{\vec{x}} = \vec{w}$

We start off with the same equation as before:
```math
\dot{\vec{x}} = F\vec{x} + L\vec{w}
```

Again, we need to define our state. Apart from position ($x, y$ coordinates), we also need to keep track of the velocity, as it will change (since $\ddot{\vec{x}} \neq 0$). In other words, our state is $\vec{x} = [x,\space \dot{x},\space y,\space \dot{y}]^T$. Taking that into account, above equation transforms to:
```math
\dot{\vec{x}} = \begin{bmatrix}
    \dot{x} \\
    \ddot{x} \\
    \dot{y} \\
    \ddot{y}
\end{bmatrix} = 
F \begin{bmatrix}
    x \\
    \dot{x} \\
    y \\
    \dot{y}
\end{bmatrix} + 
L \begin{bmatrix}
    w_1 \\
    w_2
\end{bmatrix}
```

Again, we want to manually set $F$ and $L$, so that
- $\ddot{\vec{x}} = \vec{w}$, i.e.
    - $\ddot{x} = w_1$, and
    - $\ddot{y} = w_2$,

while also making sure that 
- $\dot{x} = \dot{x}$ and 
- $\dot{y} = \dot{y}$ (as silly as it sounds):

```math
\dot{\vec{x}} = \begin{bmatrix}
    \dot{x} \\
    \ddot{x} \\
    \dot{y} \\
    \ddot{y}
\end{bmatrix} = 
\underbrace{\begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
\end{bmatrix}}_{F}
\begin{bmatrix}
    x \\
    \dot{x} \\
    y \\
    \dot{y}
\end{bmatrix} +  
\underbrace{\begin{bmatrix}
    0 & 0 \\
    1 & 0 \\
    0 & 0 \\
    0 & 1
\end{bmatrix}}_{L}
\begin{bmatrix}
    w_1 \\
    w_2
\end{bmatrix}
```

If we multiplied everything out, we would get what we want, i.e. $\dot{x} = \dot{x}$, $\dot{y} = \dot{y}$, and $\ddot{\vec{x}} = \vec{w}$:
```math
\begin{bmatrix}
    \dot{x} \\
    \ddot{x} \\
    \dot{y} \\
    \ddot{y}
\end{bmatrix} = 
\begin{bmatrix}
    \dot{x} \\
    0 \\
    \dot{y} \\
    0
\end{bmatrix} +  
\begin{bmatrix}
    0 \\
    w_1 \\
    0 \\
    w_2
\end{bmatrix}
```

---

Again, we also want to define the observation matrix $H$ based on equation:
```math
\vec{y_k} = H\vec{x_k} + \vec{v}
```

As stated, for all motion models, we want to observe only the position:
```math
\vec{y_k} = \begin{bmatrix}
    x_k^{(m)} \\
    y_k^{(m)}
\end{bmatrix}
```

Our state remains the same as for the NCV motion model (just discretized), so the equation for $\vec{y_k}$ becomes:
```math
\vec{y_k} = \begin{bmatrix}
    x_k^{(m)} \\
    y_k^{(m)}
\end{bmatrix} =
H \begin{bmatrix}
    x_k \\
    \dot{x_k} \\
    y_k \\
    \dot{y_k}
\end{bmatrix} +
\begin{bmatrix}
    v_1 \\
    v_2
\end{bmatrix}
```

We again want to get $x_k^{(m)} = x_k + v_1$, and $y_k^{(m)} = y_k + v_2$, so we set $H$ to be:
```math
H = \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0
\end{bmatrix}
```
which gives us what we want.



# Nearly Constant Acceleration

Nearly constant acceleration motion model states that jerk (derivative of acceleration) is modeled by a white noise sequence, meaning our acceleration is not constant. In other words:
- $\overset{\textbf{...}}{\vec{x}} = \vec{w}$

We start off with the same equation as with RW and NCV motion models:
```math
\dot{\vec{x}} = F\vec{x} + L\vec{w}
```

Again, we need to define our state. Apart from position ($x, y$ coordinates) and velocity ($\dot{x}, \dot{y}$), we also need to keep track of the acceleration, as it will change (since $\overset{\textbf{...}}{\vec{x}} \neq 0$). In other words, our state is $\vec{x} = [x, \space\dot{x}, \space\ddot{x}, \space y, \space\dot{y}, \space\ddot{y}]^T$. Taking that into account, above equation transforms to:
```math
\dot{\vec{x}} = \begin{bmatrix}
    \dot{x} \\
    \ddot{x} \\
    \overset{\textbf{...}}{x} \\
    \dot{y} \\
    \ddot{y} \\
    \overset{\textbf{...}}{y}
\end{bmatrix} = 
F \begin{bmatrix}
    x \\
    \dot{x} \\
    \ddot{x} \\
    y \\
    \dot{y} \\
    \ddot{y}
\end{bmatrix} + 
L \begin{bmatrix}
    w_1 \\
    w_2
\end{bmatrix}
```

Again, we want to manually set $F$ and $L$, so that
- $\overset{\textbf{...}}{\vec{x}} = \vec{w}$, i.e.
    - $\overset{\textbf{...}}{x} = w_1$, and
    - $\overset{\textbf{...}}{y} = w_2$,

while also making sure that 
- $\dot{x} = \dot{x}$ and $\dot{y} = \dot{y}$,
- $\ddot{x} = \ddot{x}$ and $\ddot{y} = \ddot{y}$:

```math
\dot{\vec{x}} = \begin{bmatrix}
    \dot{x} \\
    \ddot{x} \\
    \overset{\textbf{...}}{x} \\
    \dot{y} \\
    \ddot{y} \\
    \overset{\textbf{...}}{y}
\end{bmatrix} =
\underbrace{\begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix}}_{F}
\begin{bmatrix}
    x \\
    \dot{x} \\
    \ddot{x} \\
    y \\
    \dot{y} \\
    \ddot{y}
\end{bmatrix} + 
\underbrace{\begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    1 & 0 \\
    0 & 0 \\
    0 & 0 \\
    0 & 1
\end{bmatrix}}_{L}
\begin{bmatrix}
    w_1 \\
    w_2
\end{bmatrix}
```

If we multiplied everything out, we would get what we want:
```math
\begin{bmatrix}
    \dot{x} \\
    \ddot{x} \\
    \overset{\textbf{...}}{x} \\
    \dot{y} \\
    \ddot{y} \\
    \overset{\textbf{...}}{y}
\end{bmatrix} = 
\begin{bmatrix}
    \dot{x} \\
    \ddot{x} \\
    0 \\
    \dot{y} \\
    \ddot{y} \\
    0
\end{bmatrix} +  
\begin{bmatrix}
    0 \\
    0 \\
    w_1 \\
    0 \\
    0 \\
    w_2
\end{bmatrix}
```

---

Again, we also want to define the observation matrix $H$ based on equation:
```math
\vec{y_k} = H\vec{x_k} + \vec{v}
```

As stated, for all motion models, we want to observe only the position:
```math
\vec{y_k} = \begin{bmatrix}
    x_k^{(m)} \\
    y_k^{(m)}
\end{bmatrix}
```

Our state remains the same as for the NCA motion model (just discretized), so the equation for $\vec{y_k}$ becomes:
```math
\vec{y_k} = \begin{bmatrix}
    x_k^{(m)} \\
    y_k^{(m)}
\end{bmatrix} =
H \begin{bmatrix}
    x_k \\
    \dot{x_k} \\
    \ddot{x_k} \\
    y_k \\
    \dot{y_k} \\
    \ddot{y_k}
\end{bmatrix} +
\begin{bmatrix}
    v_1 \\
    v_2
\end{bmatrix}
```

We again want to get $x_k^{(m)} = x_k + v_1$, and $y_k^{(m)} = y_k + v_2$, so we set $H$ to be:
```math
H = \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}
```
which gives us what we want.



# Summary

We will quickly summarize what we got from previous derivations.

---

For **Random Walk (RW)** motion model:
- Conditions: $\dot{\vec{x}} = \vec{w}$, and $\ddot{\vec{x}} = 0$
- State: $\vec{x} = [x, \space y]^T$
```math
F = \begin{bmatrix}
    0 & 0 \\
    0 & 0
\end{bmatrix},
L = \begin{bmatrix}
    1 & 0 \\
    0 & 1
\end{bmatrix}, 
H = \begin{bmatrix}
    1 & 0 \\
    0 & 1
\end{bmatrix}
```

---

For **Nearly Constant Velocity (NCV)** motion model:
- Condition: $\ddot{\vec{x}} = \vec{w}$
- State: $\vec{x} = [x,\space \dot{x},\space y,\space \dot{y}]^T$
```math
F = \begin{bmatrix}
    0 & 1 & 0 & 0 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0
\end{bmatrix}, 
L = \begin{bmatrix}
    0 & 0 \\
    1 & 0 \\
    0 & 0 \\
    0 & 1
\end{bmatrix},
H = \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0
\end{bmatrix}
```

---

For **Nearly Constant Acceleration (NCA)** motion model:
- Condition: $\overset{\textbf{...}}{\vec{x}} = \vec{w}$
- State: $\vec{x} = [x, \space\dot{x}, \space\ddot{x}, \space y, \space\dot{y}, \space\ddot{y}]^T$
```math
F = \begin{bmatrix}
    0 & 1 & 0 & 0 & 0 & 0 \\
    0 & 0 & 1 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 & 0 & 0
\end{bmatrix},
L = \begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    1 & 0 \\
    0 & 0 \\
    0 & 0 \\
    0 & 1
\end{bmatrix},
H = \begin{bmatrix}
    1 & 0 & 0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}
```

---

Based on state transition matrix $F$, matrix $L$ and observation matrix $H$, we can obtain the system matrix $\Phi$ and the system covariance matrix $Q$. The system covariance matrix $Q$ arises when the motion model equations we used are discretized, but it's not necessary to go into those details (you can check the lecture slides Recursive Bayes Filter part I). It's enough to know that $Q$ is the covariance matrix of noise $\vec{w}$, with power spectral density $q$. We calculate $\Phi$ and $Q$ in the following manner:
```math
\begin{align*}
    \Phi &= \Phi(T) = e^{FT} \\
    Q &= \int_{0}^{T} (\Phi L) q (\Phi L)^T dT
\end{align*}
```

where $T$ is the time between frames/timesteps. For this assignment, we can safely take $T = 1$. When implementing, as we will be using `sympy` (Python's symbolic toolbox) for calculations, we should keep using the symbol $T$ until just the very last step, where we can replace it with the value 1.

The only thing that remains is the observation covariance matrix $R$ of noise $\vec{v}$, with power spectral density $r$. According to assignment slides, for all motion models, we set: 
```math
R = r \begin{bmatrix}
    1 & 0 \\
    0 & 1
\end{bmatrix}
```

When implementing, power spectral densities $q$ and $r$ for covariance matrices $Q$ and $R$ are just some real numbers.
 