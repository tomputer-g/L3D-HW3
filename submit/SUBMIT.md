# 0. Transmittance Calculation

TODO Submit result

$$AAA$$

# 1. Differentiable Volume Rendering

## 1.3 Ray Sampling

Here are the resulting XY Grid and Rays outputs:

<image src="q1/xy_grid.png" width=256> <image src="q1/rays.png" width=256>

## 1.4 Point Sampling

Here is the render points output that shows the point samples:

<image src="q1/sampled_pts.png" width=256>

## 1.5 Volume Rendering

Here is the trained spinning box:

<image src="q1/part_1.gif" width=256>

Here is the visualized depth of the box from one angle:

<image src="q1/depth.png" width=256>

# 2. Optimizing a basic implicit volume

## 2.1 Random ray sampling

TODO verify what we are submitting, this was just code

## 2.2 Loss and training

The optimized box center is at $(0.25, 0.25, 0.00)$ and the side lengths are $(2.00, 1.50, 1.50)$.

## 2.3 Visualization

<image src="q2/part_2.gif" width=256>

# 3. Optimizing a Neural Radiance Field

TODO take out direction (view dependence). Train and show resulting gif

# 4. NeRF extras

## 4.1 View Dependance

## 4.2 Coarse/Fine Sampling

# 5. Sphere Tracing

Here is the rendered torus:

<image src="q5/part_5.gif" width=256>

Here is the code snippet for my implementation:

```python
EPSILON = 1e-5
points = origins
mask = torch.zeros((directions.shape[0], 1)).bool().to(origins.device)
t = torch.zeros((directions.shape[0], 1)).to(origins.device)
for i in range(self.max_iters):
    fp = implicit_fn(points)
    mask = mask | (fp < EPSILON)
    t += fp
    points = origins + t * directions

return points, mask
```

Overall, we run sphere tracing for max_iters iterations. At each iteration, we take the implicit function value at the point and check if it is smaller than epsilon (effectively on the surface). If so, we set the mask to be true for that point, since it has intersected the surface. The t and points values are updated according to the lecture to increase by the radius of the sphere and take a step towards the surface. After reaching max_iters, any points with mask = false means the ray never found an intersecting surface, whereas all points with mask = true have converged at a surface.

# 6. Optimizing a Neural SDF

Here is the predicted NeuralSDF for the bunny, given the input point cloud:

<image src="q6/part_6_input.gif" width=256>
<image src="q6/part_6.gif" width=256>

The MLP adopts a similar architecture and uses a single MLP with input skips at the fourth layer. One major difference compared to the NeRF task was that the distance output head no longer contains an activation for nonlinearity. Since the distance is an unbounded linear output, the output head simply has a Linear layer.

The eikonal loss enforces the constraint that the gradient norms are equal to one, and satisfies the eikonal constraint. To enforce this, the eikonal loss function ensures that the MSE Loss between the norm of the gradients and the value 1 is minimized, forcing the norm of the gradients to approach 1 as the loss decreases.

# 7. VolSDF

TODO

# 8. Neural Surface Extras

## 8.1 Render Large Scene with Sphere Tracing

## 8.2 Fewer Training Views

## 8.3 Alternate SDF to Density Conversions