# MyGPURaytracer
A high performance raytracer programmed with CUDA. There are various improval features such as AI denoising, depth of field, arbitrary mesh loading. etc in it.  

**Spaceship in cornell box**

![cornell.2021-11-17_12-17-06z.43output](F:\2021b\MyGPURaytracer\build\Release\cornell.2021-11-17_12-17-06z.43output.png)

![cornell.2021-11-17_12-17-06z.54output](F:\2021b\MyGPURaytracer\build\Release\cornell.2021-11-17_12-17-06z.54output.png)

## Features

* Monte Carlo based ray tracer on GPU;

* BSDF Evaluation: pure diffuse, reflection and refraction, diffuse+specular;

* AI denoise implemented with [Intel Open Image Denoise](https://github.com/OpenImageDenoise/oidn);

* Arbitrary .obj model file loading with its specular,diffuse,bumping textures;

* Depth of field

* GPU improvement: stream compaction

* Cache first intersection

   

## TODO

* [Wavefront pathtracing](https://research.nvidia.com/publication/megakernels-considered-harmful-wavefront-path-tracing-gpus) --group rays by material
* Hierarchical spatial data structurs(BVH)--for massive scenes

## Usage

Only support win32：

Raytracer.exe directory/to/scene.txt

executable directory example:

```
--ProjectFolder
	|--build
	    |--bin
		|--Release
		    |--Raytracer.exe
		|--models
		    |--materials
			|--a.mtl
		    |-- a.obj 
		|--textures
		    |--a_kd.jpg
```



Note: The project framework is referenced from CIS565 course lab.

