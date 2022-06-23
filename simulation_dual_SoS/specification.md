```pseudocode
/* Size = Depth * width */
Define temporary_grid:
	/* For guiding the defining of initial_pressure_map*/
	temp_grid_size = 1024 x 768 /* The center with 384px width of the upper half will be the real grid*/
Define initial_pressure_map:
	grid_size = 512 x 384
	pixel_size = 100 um
Define medium_SoS and tissue_SoS:
	medium_SoS = c = waterSoundSpeed(20)
	tissue_SoS = Gaussian_random(mean=1538, std_div=39.4)
Define 'water-tissue boundary':
	y = uniform_random(512,1024)
	x = uniform_random(192 - floor((x - 512) / 512 * 192), 576 + ceil((x + 512) / 512 * 192))	
	center = (y, x)
	radius = uniform_random(y - 192, y - 64)
	/* The boundary is defined by a circle with upper parameters*/
	medium[r][c] = tissue_SoS if sqrt((r-y).^2 + (c-x).^2) <= radium else medium_SoS
Define 'other tissues':
	tissus_number = uniform_random(30,100)
	for i in vessels_number:
		width = exp_random(2)
		elliptic_ratio = uniform_random(0.8, 1)
		initial_pressure = uniform_random(0.6, 1)
```

The blue region will be the tissue with different SoS compared with medium (water).

<img src="C:\Users\luolu\AppData\Roaming\Typora\typora-user-images\image-20220614143847500.png" alt="image-20220614143847500" style="zoom:25%;" />