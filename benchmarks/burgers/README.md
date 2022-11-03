## Guide to Parthenon-VIBE: A benchmark that solves the Vector Inviscid Burgers' Equation on a block-AMR mesh

### Description

This benchmark solves the inviscid Burgers' equation

$$
\begin{equation*}
\partial_t \mathbf{u} + \nabla\cdot\left(\frac{1}{2}\mathbf{u} \mathbf{u}\right) = 0 \\[1ex]
\end{equation*}
$$

and evolves one or more passive scalar quantities $q^i$ according to

$$
\begin{equation*}
\partial_t q^i + \nabla \cdot \left( q^i \mathbf{u} \right) = 0
\end{equation*}
$$

as well as computing an auxiliary quantity $d$ that resemebles a kinetic energy

$$
\begin{equation*}
d = \frac{1}{2} q^0 \mathbf{u}\cdot\mathbf{u}\;.
\end{equation*}
$$


