# MyGPURaytracer
A high performance raytracer programmed with CUDA.  The implementation of  the paper "[Edge-Avoiding A-Trous Wavelet Transform for fast Global Illumination Filtering](https://jo.dreggn.org/home/2010_atrous.pdf)".




## Features

* Monte Carlo based ray tracer on GPU;

* BSDF Evaluation: pure diffuse, reflection and refraction, diffuse+specular;

* Arbitrary .obj model file loading with its emissive, specular,diffuse,bumping textures;

* Depth of field

* GPU improvement: stream compaction

* Cache first intersection

  

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

