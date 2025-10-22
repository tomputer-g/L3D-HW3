# How to run this code to recreate results

## Q1

### Q1.3

Run `python3 volume_rendering_main.py --config-name=box`.

### Q1.4

Run `python3 volume_rendering_main.py --config-name=box`.

### Q1.5

Run `python3 volume_rendering_main.py --config-name=box`.

## Q2

### Q2.2

Run `python3 volume_rendering_main.py --config-name=train_box`.

## Q3

Go to implicit.py and change the network so that it uses the (256, 128) linear layer, and change rgb_output to call self.final_net with second_block_output directly (instead of concatenated features).

Run `python3 volume_rendering_main.py --config-name=nerf_lego`.

## Q4

### Q4.1

Run `python3 volume_rendering_main.py --config-name=nerf_materials` and `python3 volume_rendering_main.py --config-name=nerf_materials_highres`.

## Q5

Run `python3 surface_rendering_main.py --config-name=torus_surface`.

## Q6

Run `python3 surface_rendering_main.py --config-name=points_surface`.

## Q7
Run `python3 -m surface_rendering_main --config-name=volsdf_surface`.

## Q8

### 8.1
Run `python3 -m surface_rendering_main --config-name=scene_surface_q8_1`.

### 8.2
Modify the code in dataset.py (around line 125) to the desired amount of downsampling depending on part of the question. Run `python3 -m surface_rendering_main --config-name=volsdf_surface`. Run `python3 -m volume_rendering_main --config-name=nerf_lego`.
