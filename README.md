# MyGPURaytracer
A high performance monte-carlo raytracer programmed with CUDA. There are various improval features such as AI denoising, cache first bounce, sort materials, arbitrary mesh loading. etc in it.  

![obj](.\imgs\obj.png)




## Features

* Monte-Carlo ray tracer on GPU;

* BSDF Evaluation: pure diffuse, reflection and refraction, diffuse+specular;

  |         Purely Diffuse         |                Reflective                |
  | :----------------------------: | :--------------------------------------: |
  | ![diffuse](.\imgs\diffuse.png) |   ![reflection](.\imgs\reflection.png)   |
  |         **Refractive**         |         **Diffuse & Reflective**         |
  | ![refr](.\imgs\refraction.png) | ![diffuse&spec](.\imgs\diffuse&spec.png) |

  

* AI denoise implemented with [Intel Open Image Denoise](https://github.com/OpenImageDenoise/oidn);

  Use albedo image as an auxiliary to reduce noise (oidnRayTraced.exe file output denoised result in real-time). 

  |               Ray-traced noisy result                |                         Albedo                         |             Denoised Result              |
  | :--------------------------------------------------: | :----------------------------------------------------: | :--------------------------------------: |
  |         ![diffuse](.\imgs\diffuse_input.png)         |     ![diffuse_albedo ](.\imgs\diffuse_albedo .png)     |      ![diffuse](.\imgs\diffuse.png)      |
  |      ![reflect_input](.\imgs\reflect_input.png)      |      ![reflect_albedo](.\imgs\reflect_albedo.png)      |   ![reflection](.\imgs\reflection.png)   |
  |      ![refract_input](.\imgs\refract_input.png)      |      ![refract_albedo](.\imgs\refract_albedo.png)      |   ![refraction](.\imgs\refraction.png)   |
  | ![diffuse&spec_input](.\imgs\diffuse&spec_input.png) | ![diffuse&spec_albedo](.\imgs\diffuse&spec_albedo.png) | ![diffuse&spec](.\imgs\diffuse&spec.png) |

  

* Use [tinyobj](https://github.com/syoyo/tinyobjloader) to load arbitrary .obj model with its specular,diffuse, emission and bumping textures;

  

* Depth of field

* GPU improvement: stream compaction

* Cache first intersection

* Stochastic Sampled Antialiasing

## TODO

For massive scenes:

* [Wavefront pathtracing](https://research.nvidia.com/publication/megakernels-considered-harmful-wavefront-path-tracing-gpus) --group rays by material
* Hierarchical spatial data structures(BVH)

## Usage

Only support win32：

oidnRaytracer.exe directory/to/scene.txt

executable directory example:

```
--ProjectFolder
	|--build
	    |--bin
		|--Release
		    |--oidnRaytracer.exe
		|--models
		    |--materials
			|--a.mtl
		    |--a.obj 
		|--textures
		    |--a_kd.jpg
		    |--a_ks.jpg
		    |--a_emi.jpg
```

