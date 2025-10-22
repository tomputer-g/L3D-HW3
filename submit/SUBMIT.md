# 0. Transmittance Calculation

The base case for transmittance at y1 is T(x, y1) = 1.

The inductive case is $$T(x, x_{t_i}) = T(x, x_{t_{i-1}}) e^{- \sigma_{t_{i-1}} \cdot \Delta t}$$

$$T(y1, y2) = 1 \cdot  e^{-1 \cdot 2} =  e^{-2}$$

$$T(y1, y3) =  e^{-2} \cdot  e^{-0.5 \cdot 1} =  e^{-2.5}$$

$$T(y1, y4) = e^{-2.5}\cdot e^{-10 * 3} = e^{-32.5} $$


The transmittance from the source to the observer is $e^{-32.5}$.

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

## 2.2 Loss and training

The optimized box center is at $(0.25, 0.25, 0.00)$ and the side lengths are $(2.00, 1.50, 1.50)$.

## 2.3 Visualization

<image src="q2/part_2.gif" width=256>

# 3. Optimizing a Neural Radiance Field

<image src="q3/part_3.gif" width=256>

# 4. NeRF extras

## 4.1 View Dependance

Including view dependance allows for shinier surfaces to be modeled more accurately (such as shiny polished metal), and thus represent materials in greater fidelity compared to just modeling based on the XYZ point coordinates. However, high view dependance results in worse generalization quality. If not regularized properly, as shown in the [VDN-NeRF paper](https://arxiv.org/abs/2303.17968), naive neural nets tend to lose geometric details when there is high view dependance fo rthe scene. 

Here is the same Lego scene rendered with view dependence:

<image src="q4/part_3.gif" width=256>

Here are the Materials and High-res materials scenes:

<image src="q4/materials.gif" width=256>

<image src="q4/materials_highres.gif" width=512>

## 4.2 Coarse/Fine Sampling

Not Attempted.

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

Alpha controls the maximum density inside the surface (since it is directly multiplied with the CDF). Larger alpha values make the interior of the object appear more opaque, and vice versa. Beta is inversely related to the exponent in the definition of the CDF controlling the behavior of the density at the edges of the surface. A smaller beta results in a larger exponentiation value, and creates a sharper transition discussed below.

Beta controls the sharpness of the density transition at the edge of the SDF. A higher beta results in a softer transition which creates a blurrier edge, while a lower beta creates a sharper transition and a thinner surface. Beta in this way biases the learned SDF in creating a smoother surface when using high beta, and vice versa.

A SDF is easier to train with a high beta, because the gradient at the edges are smaller and thus easier to learn and optimize against compared to a step-function-like transition that a lower beta configuration creates.

However, an accurate surface requires lower beta values. The lower beta is, the stepper the transition is at the object's edge, which creates a more well-defined boundary that better approximates the sharp boundary that a true SDF would create.

I wanted to experiment with beta so that the output is reasonably sharp but the edges of the object are not jagged or unstable.

With a very low beta of 0.025, while the surface appears sharper, there are increasing amounts of fake detections along the side of the base platform.

<image src="q7/part_7_geometry_beta0025.gif" width=256>
<image src="q7/part_7_beta0025.gif" width=256>

I chose to use a higher beta of 0.07, the geometry is a lot more consistent, at a slight cost of the model appearing somewhat rounded.

<image src="q7/part_7_geometry_beta007.gif" width=256>
<image src="q7/part_7_beta007.gif" width=256>


# 8. Neural Surface Extras

## 8.1 Render Large Scene with Sphere Tracing

I created a 3x3x3 grid (27 objects) of SDFs, then randomly picked each one to be either a cube, sphere, or torus. The SDFs are composed by taking the  minimum distance for all SDFs whenever the distance of a ray is requested, so that it returns the union of the objects.

<image src="q8/part_8_1.gif" width=256>

## 8.2 Fewer Training Views

The following VolSDF scene was trained using only 20 views:

<image src="q8/part_7_geometry_20view_volsdf.gif" width=256>
<image src="q8/part_7_20view_volsdf.gif" width=256>

NeRF was not able to converge with 20 views with the default settings. I experimented with different settings and found that with 192 points per ray, NeRF does train to the below scene successfully:

<image src="q8/part_3_sparse_192ppr.gif" width=256>

We observe that the NeRF scene has significant blurring around the model, suggesting that the provided views were not enough to totally converge the learned scene.

After some experimentation, the minimal amount of images for NeRF to consistently converge upon (without modifying points per ray) was around 70 images, which results in the following scene:

<image src="q8/part8_2_70views.gif" width=256>

In contrast, here is the VolSDF trained on the same 70 views:

<image src="q8/part_7_geometry_70view_volsdf.gif" width=256>
<image src="q8/part_7_70view_volsdf.gif" width=256>


## 8.3 Alternate SDF to Density Conversions

Not Attempted.
