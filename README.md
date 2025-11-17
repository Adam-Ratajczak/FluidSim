# Navier-Stokes equations solver for incompressible fluids using implicit diffusion method

This is a brief summary covering the general idea behind eulerian fluid simulations.

## 1. Dependencies:
- numpy (for working with large datasets)
- scipy (for scientific calculations)
- pyamg (for efficient linear systems solvers using multigrid algorithm)

## 2. Navier-Stokes equations:

### 2.1. General form:

$$
\rho\left(\frac{\partial u}{\partial t} + (u\cdot\nabla)u\right)
= -\nabla p + \nu\nabla^2 u + f
$$

$$
\nabla\cdot u = 0
$$

### 2.2. Left-hand side:

- $\rho$, Fluid density (mass per unit volume). Scales the momentum of the fluid.

- $\frac{\partial {u}}{\partial {t}}$, Local or unsteady acceleration. Represents how the velocity field changes with time at a fixed point.

- $\rho ( \frac{\partial {u}}{\partial t} + ({u} \cdot \nabla){u} )$, Total (material) acceleration multiplied by density, which gives the inertial forces per unit volume.

### 2.3. Right-hand side:

- $- \nabla {p} $, Pressure gradient force. Fluid accelerates from high pressure toward low pressure.

- $\nu \nabla^2 {u}$, Viscous diffusion term:

    - $\nu$ is the kinematic viscosity (resistance to shear).

    - $\nabla^2 {u}$ is the Laplacian of the velocity field; it represents how momentum diffuses due to internal friction.

- ${f}$, External forces influencing the system e.g. gravity, electromagnetic forces

### 2.4. Continuity Equation (Incompressibility Constraint):

- $\nabla \cdot {u} = 0$, Enforces constant density. States that the velocity field has zero divergence, meaning fluid volume stays the same.

### 2.5. Term explanation:

- ${u}$, Velocity field, how fast fluid particles are moving

- ${p}$, Pressure field

- $\nu$, Kinematic viscosity, a ratio of dynamic viscosity and gradient:

$$
    \nu = \frac{\mu}{\rho}
$$

- $\nabla$, Gradient, first derivative of the vector field. For 2D, it can be expanded as follows:

$$
    \nabla = (\frac{\partial}{\partial {x}}, \frac{\partial}{\partial {y}} )
$$

- $\nabla^2 u$, Laplacian, second derivative of the vector field. For 2D, it can be expanded as follows:

$$
    \nabla^2 u = (\frac{\partial^2 u}{\partial {x}^2}, \frac{\partial^2 u}{\partial {y}^2} )
$$

- $\nabla \cdot {u}$, Divergence, It is basically a dot product of vector field and its gradient. Expanded form:
$$
    \nabla \cdot {u} 
    = (\frac{\partial}{\partial {x}}, \frac{\partial}{\partial {y}} ) \cdot ({u_x}, {u_y} )
    = \frac{\partial {u_x}}{\partial {x}} + \frac{\partial {u_y}}{\partial {y}}
$$

Therefore, divergence means that amount of fluid entering the cell must be equal to the amount of fluid leaving the cell.

## 3. About the algorithm:

### 3.1. Algorithm assumes that:
- fluid is incompressible (the amount of fluid entering and leaving individual cell is same i.e. divergence for each cell is equal to zero)
- density and viscosity in the entire system is constant
- structure of the system is as follows:
    ```
    O ---- V ---- O ---- V ---- O ---- V ---- O
    |             |             |             |
    |             |             |             |
    U      P      U      P      U      P      U
    |             |             |             |
    |             |             |             |
    O ---- V ---- O ---- V ---- O ---- V ---- O
    |             |             |             |
    |             |             |             |
    U      P      U      P      U      P      U
    |             |             |             |
    |             |             |             |
    O ---- V ---- O ---- V ---- O ---- V ---- O
    ```

    Where:
    
    - U, velocity in X axis
    
    - V, velocity in Y axis
    
    - P, uniform pressure of the cell

    Worth noting is that, U and V represents inflow and outflow for each cell, that's why It is measured at the cell faces. 

    Dimensions of U matrix are: Nx+1, Ny

    Dimensions of V matrix are: Nx, Ny+1

    P is pressure of the cell, thus its dimensions are: Nx,Ny

    Where Nx and Ny are dimensions of the grid

### 3.2. Steps of the algorithm:

- Diffusion (local velocities needs to be averaged)

- advection (velocities need to propagate)

- pressure solve (divergence needs to be equal to zero at every cell in the system)

- projection (pressure difference needs to be accounted for the velocity change)

- apply external forces / boundary conditions

#### 3.2.1. Diffusion:

##### 3.2.1.1. Overview:

Viscous diffusion makes the velocity field more even. It irons out sharp gradients so the flow becomes smoother.
It is governed by this part of the equation:

$$
    \nu \nabla^2 {u}
$$

Why is there a laplacian?
First derivative measures the change of the slope.
A second derivative measures the curvature of the field — how sharply it bends.
The Laplacian $\nabla^2$ measures how much a value differs from its surroundings. This syntax is used in many equations in physics e.g. heat propagation.
A large Laplacian means the point is very different from its neighbors, so diffusion acts strongly.

##### 3.2.1.2. Algorithm overview:

```py
def diffuse_velocity(U, V):
    def solve_component(F, solver):
        b = F.flatten()
        x = solver.solve(b, tol=1e-8)
        return x.reshape(F.shape)

    U = solve_component(U, M_diffusionU)
    V = solve_component(V, M_diffusionV)

    return U, V
```

**diffuse_velocity** function applies diffusion to U and V vector fields. Algorithm description:
1. Each term is flattened to solver appropriate format.
2. Linear equation is solved using multigrid algorithm
3. The result is reshaped to its original form and returned

But which equation is given to the solver? Let's look at the linear equation matrix:
```py
def build_diffusion_matrix(width, height, a):
    N = width * height
    A = lil_matrix((N, N))

    def idx(x, y):
        return y + x * height

    for x in range(width):
        for y in range(height):
            i = idx(x, y)

            if is_solid(x, y):
                A[i, i] = 1.0
                continue

            A[i, i] = 1.0 + 4.0 * a

            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if 0 <= nx < width and 0 <= ny < height and not is_solid(nx, ny):
                    j = idx(nx, ny)
                    A[i, j] = -a

    return A.tocsr()
```
Where:
- width, height are dimensions of the field
- a is some coefficient: ${a} = \frac{d {t} \nu}{d {x} d {y}}$

Its purpose is to construct a sparse matrix representing the discrete diffusion operator on a 2-D grid using a 5-point Laplacian stencil.

In continuous form, it solves:

$$
    {u}^{n + 1} - a \nabla^2 {u}^{n + 1} = {u}^{n}
$$

But computers cannot solve the continuous form, we need to discretize it into a 2D grid:

$$
    \nabla^2 {u}[i,j] \approx {u}[i, j+1] + {u}[i, j-1] + {u}[i+1, j] + {u}[i-1, j] - 4{u}[i, j]
$$

When we plug it to the actual equation, we derive:


$$
    {u}^{n + 1}[i,j] - {a} ( {u}^{n + 1}[i, j+1] + {u}^{n + 1}[i, j-1] + {u}^{n + 1}[i+1, j] + {u}^{n + 1}[i-1, j] - 4{u}[i, j] ) = {u}^{n}
$$

$$
    (1 + 4{a} ){u}^{n}[i, j] - {a}{u}^{n + 1}[i, j+1] - {a}{u}^{n + 1}[i, j-1] - {a}{u}^{n + 1}[i+1, j] - {a}{u}^{n + 1}[i-1, j] = {u}^{n}
$$

Which is exactly what our stencil solves:
- Center coefficient: $1 + 4{a}$
- Four neighbors: $-{a}$

#### 3.2.2. Advection:

##### 3.2.2.1. Overview:

Advection is a step when velocities are updated before pressure can be resolved.

Its purpose is to solve this part of the equation:

$$
\begin{aligned}
    \frac{\partial {u}}{\partial t} + ({u} \cdot \nabla){u} = 0
\end{aligned}
$$

##### 3.2.2.2. Algorithm overview:

```py
def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def lerp(a, b, t):
    return a + (b - a) * t

def sample_bilinear(eV, eCX, eCY, worldPos):
    width = (eCX - 1) * CELL_SIZE
    height = (eCY - 1) * CELL_SIZE

    px = (worldPos[0] + width  / 2.0) / CELL_SIZE
    py = (worldPos[1] + height / 2.0) / CELL_SIZE

    left   = int(m.floor(clamp(px, 0.0, eCX - 2.0)))
    bottom = int(m.floor(clamp(py, 0.0, eCY - 2.0)))
    right  = left + 1
    top    = bottom + 1

    xFrac = clamp(px - left,   0.0, 1.0)
    yFrac = clamp(py - bottom, 0.0, 1.0)

    vT = lerp(eV[left,  top],    eV[right, top],    xFrac)
    vB = lerp(eV[left,  bottom], eV[right, bottom], xFrac)

    return lerp(vB, vT, yFrac)

def get_vel_at_world_pos(worldPos):
    velX = sample_bilinear(U, FIELD_WIDTH + 1, FIELD_HEIGHT, worldPos)
    velY = sample_bilinear(V, FIELD_WIDTH, FIELD_HEIGHT + 1, worldPos)
    
    return (velX, velY)

def advect(U, V, dt):
    newU = U.copy()
    newV = V.copy()

    for x in range(FIELD_WIDTH + 1):
        for y in range(FIELD_HEIGHT):

            if is_solid(x, y) or is_solid(x-1, y):
                newU[x][y] = 0
                continue

            wx = (x - FIELD_WIDTH/2) * CELL_SIZE
            wy = (y + 0.5 - FIELD_HEIGHT/2) * CELL_SIZE

            vel = get_vel_at_world_pos((wx, wy))

            back = (wx - vel[0] * dt,
                    wy - vel[1] * dt)

            newU[x][y] = sample_bilinear(U, FIELD_WIDTH+1, FIELD_HEIGHT, back)

    for x in range(FIELD_WIDTH):
        for y in range(FIELD_HEIGHT + 1):

            if is_solid(x, y) or is_solid(x, y-1):
                newV[x][y] = 0
                continue

            wx = (x + 0.5 - FIELD_WIDTH/2) * CELL_SIZE
            wy = (y - FIELD_HEIGHT/2) * CELL_SIZE

            vel = get_vel_at_world_pos((wx, wy))

            back = (wx - vel[0] * dt,
                    wy - vel[1] * dt)

            newV[x][y] = sample_bilinear(V, FIELD_WIDTH, FIELD_HEIGHT+1, back)

    return newU, newV
```

**advect** function implements a Semi-Lagrangian method for fluid advection. What does Semi-Lagrangian mean? We call Lagrangian anything that bases on some kind of particles.

Semi-Lagrangian advection follows the idea:

$$
    {u}^{n+1}(x) = {u}^{n} ({x} - {u}^{n}(x) d {t})
$$

i.e., “go backward in time along the flow, pick up the old value.”

In order to reliably fetch the desired value, we need to sample the field based on the discrete values we already have.
Functions **get_vel_at_world_pos** and **sample_bilinear** does exactly that.


#### 3.2.3. Pressure solve:

##### 3.2.3.1. Overview:

The constraint of incompressible fluid simulation is that, divergence of every cell needs to be equal to zero. This step is supposed to update the pressure in order to keep the "zero divergence principle".
It is represented by the pressure Poisson operator:

$$
    \nabla^2 p = \frac {1} {d t} \nabla \cdot {u}
$$

##### 3.2.3.2. Algorithm overview:

```py
def calculate_velocity_divergence_at_cell(x, y):
    uR = U[x+1, y]
    uL = U[x,   y]
    vT = V[x,   y+1]
    vB = V[x,   y]

    div = ((uR - uL) + (vT - vB)) / CELL_SIZE
    return div

def get_divergence_matrix():
    divergence = np.zeros([FIELD_WIDTH, FIELD_HEIGHT])
    for x in range(FIELD_WIDTH - 1):
        for y in range(FIELD_HEIGHT - 1):
            divergence[x, y] = calculate_velocity_divergence_at_cell(x, y)
    return divergence

def pressure_poisson_solve(P):
    divergence = get_divergence_matrix()
    b = divergence.flatten()
    
    new_p = M_Pressure.solve(b, tol=1e-8)

    return new_p.reshape(P.shape)
```

**pressure_poisson_solve** is to ensure that divergence of every cell will be zero. 
**get_divergence_matrix** is the helper function fetching calculated divergence matrix.
**calculate_velocity_divergence_at_cell** is the helper function getting divergence at a given cell. It realizes the equation: 

$$
    \nabla \cdot {u} 
    = \frac{\partial {u_x}}{\partial {x}} + \frac{\partial {u_y}}{\partial {y}}
    = \frac{d {u_x} + d {u_y}}{h}
$$
Where h is cell size. In our case, $h = d {x} = d {y}$

In **pressure_poisson_solve** function:
1. Pressure term is flattened to solver appropriate format.
2. Linear equation is solved using multigrid algorithm
3. The result is reshaped to its original form and returned

Once again, we need to construct the matrix for solver, as in diffusion step:

```py
def build_pressure_matrix(width, height):
    N = width * height
    A = lil_matrix((N, N))

    def idx(x, y):
        return y + x * height

    for x in range(width):
        for y in range(height):
            i = idx(x, y)

            if is_solid(x, y):
                A[i, i] = 1.0
                continue

            neighbors = 0

            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if 0 <= nx < width and 0 <= ny < height and not is_solid(nx, ny):
                    neighbors += 1

            A[i, i] = neighbors

            for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
                if 0 <= nx < width and 0 <= ny < height and not is_solid(nx, ny):
                    j = idx(nx, ny)
                    A[i, j] = -1.0

    return A.tocsr()
```

Let's start with the Poisson operator for a pressure correction step:

$$
    \nabla^2 p = \frac {1} {d t} \nabla \cdot {u}
$$

It is obtained by substituting the velocity update
$$
    u^{n+1} = u^{*} - \frac{dt}{\rho}\nabla p
$$
into the incompressibility condition
$$
    \nabla \cdot u^{n+1} = 0
$$
and taking the divergence.

Let a be the number of valid neighbors. (Valid - not solid)

Let A be the list of valid neighbors.

Then the discrete form of our equation shall be:

$$
    \nabla^2 p \approx a{p}[i,j] - \sum_{i,j}^{A} p[i,j]
$$

Our matrix does exactly that:
- Center coefficient: number of neighbors
- Valid neighbors: -1
- Solid cells: 1

#### 3.2.4. Velocities projection:

##### 3.2.4.1. Overview:

After the pressure grid is updated, velocities need to follow pressure change, therefore this step is written as pressure gradient.

$$
-\nabla p
$$

##### 3.2.4.2. Algorithm overview:

```py
def project_velocities(U, V, rho, dt):
    k = dt / (rho * CELL_SIZE)
    for x in range(0, FIELD_WIDTH + 1):
        for y in range(0, FIELD_HEIGHT):
            if is_solid(x - 1, y) or is_solid(x, y):
                U[x, y] = 0.0
                continue
            
            pL = get_pressure(x - 1, y + 0)
            pR = get_pressure(x + 0, y + 0)
            U[x, y] -= k * (pR - pL)
            
    for x in range(0, FIELD_WIDTH):
        for y in range(0, FIELD_HEIGHT + 1):
            if is_solid(x, y - 1) or is_solid(x, y):
                V[x, y] = 0.0
                continue
            
            pT = get_pressure(x + 0, y + 0)
            pB = get_pressure(x + 0, y - 1)
            V[x, y] -= k * (pT - pB)
    
    return U, V
```

Algorithm is rather straightforward.
Velocities need to be updated by the current acceleration.
Acceleration in terms of fluid dynamics is pressure difference between two cells.

In **project_velocities** function:
1. Fetch pressure values for neighboring cells
2. Calculate the acceleration and subtract from current velocities

But where does **k** coefficient come from, i.e. how do we know how quickly does the fluid accelerate between two cells?

Let's start by writing and rearranging continuous form of the Newton's second law of fluids:

$$
\rho \frac{\partial {u}}{\partial {t}}= -\nabla p + f \\[6pt]
\frac{\partial {u}}{\partial {t}} = - \frac{1}{\rho} \nabla p
$$

For discrete field, equation needs to be rewritten as follows:

$$
{u}^{n+1} = {u}^{n} - \frac{d {t}}{\rho} \nabla p
$$

For U component: 

$$
{u}^{n+1}[i,j] = {u}^{n}[i,j] - \frac{dt (p[i-1,j] - p[i,j])}{\rho dx} = {u}^{n}[i,j] - k (p[i-1,j] - p[i,j])
$$

For V component: 

$$
{v}^{n+1}[i,j] = {v}^{n}[i,j] - \frac{dt (p[i,j-1] - p[i,j])}{\rho dy} = {v}^{n}[i,j] - k (p[i,j-1] - p[i,j])
$$

h = dx = dy, therefore $k = \frac{dt}{\rho h}$

## 4. License:

This software is published under BeerWare license:
```
"THE BEER-WARE LICENSE" (Revision 42):

<a3.ratajczak@gmail.com> wrote this file. As long as you retain this notice you can do whatever you want with this stuff. If we meet some day, and you think this stuff is worth it, you can buy me a beer in return.
Adam Ratajczak
```