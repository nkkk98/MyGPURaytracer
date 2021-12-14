# MyGPURaytracer
A high performance monte-carlo raytracer programmed with CUDA. There are various improval features such as AI denoising, cache first bounce, sort materials, arbitrary mesh loading. etc in it.  

![obj](https://user-images.githubusercontent.com/28896013/144418444-ea196f8b-9dda-4874-b55d-6a8c6a2a4b94.png)





## Features

* Monte-Carlo ray tracer on GPU;

* BSDF Evaluation: pure diffuse, reflection and refraction, diffuse+specular;

  |         Purely Diffuse         |                Reflective                |
  | :----------------------------: | :--------------------------------------: |
  | ![diffuse](https://user-images.githubusercontent.com/28896013/144418475-ad6a5ae4-2ad5-4b88-b5a9-a4f16257962e.png) |   ![reflection](https://user-images.githubusercontent.com/28896013/144418516-0c0c6ace-1a11-46d0-863b-e3eb9c7ef99a.png)   |
  |         **Refractive**         |         **Diffuse & Reflective**         |
  | ![refraction](https://user-images.githubusercontent.com/28896013/144418547-c75cf9f6-258a-4347-83fb-818fb905b548.png) | ![diffuse spec](https://user-images.githubusercontent.com/28896013/144418575-7dd252b9-bb65-4427-a88c-df29e9fa3bf1.png) |

  

* AI denoise implemented with [Intel Open Image Denoise](https://github.com/OpenImageDenoise/oidn);

  Use albedo image as an auxiliary to reduce noise (oidnRayTraced.exe file output denoised result in real-time). 

  |               Ray-traced noisy result                |                         Albedo                         |             Denoised Result              |
  | :--------------------------------------------------: | :----------------------------------------------------: | :--------------------------------------: |
  |         ![diffuse_input](https://user-images.githubusercontent.com/28896013/144418642-5f46a6db-1f4f-4f32-880f-f3f835cc6015.png)         |     ![diffuse_albedo ](https://user-images.githubusercontent.com/28896013/144418667-a5907ccb-3998-4b9b-8e29-31d97625051c.png)    |      ![diffuse](https://user-images.githubusercontent.com/28896013/144418692-8c2e7eec-97e2-4688-b5ea-c4df73bd4b98.png)     |
  |      ![reflect_input](https://user-images.githubusercontent.com/28896013/144418718-7cdf90e5-825c-442d-952e-30da065367c5.png)       |      ![reflect_albedo](https://user-images.githubusercontent.com/28896013/144418744-ab4b4cee-e029-4e67-ad06-4f2f0899b553.png)      |   ![reflection](https://user-images.githubusercontent.com/28896013/144419367-6fef6ab9-2bb8-4208-acb0-45ff8738aeaa.png)   |
  |      ![refract_input](https://user-images.githubusercontent.com/28896013/144419400-49bbb9a4-9f86-4174-8d47-6143260ee2cf.png)      |      ![refract_albedo](https://user-images.githubusercontent.com/28896013/144419424-522aafd1-d3be-4ef5-a744-3e5b22f25eaf.png)     |   ![refraction](https://user-images.githubusercontent.com/28896013/144418783-7cd5e33b-bf48-46d1-8702-471f33c845b4.png)   |
  | ![diffuse spec_input](https://user-images.githubusercontent.com/28896013/144418825-ed416553-28a0-4c54-80b6-9db4ddbddb5a.png) | ![diffuse spec_albedo](https://user-images.githubusercontent.com/28896013/144418856-afe83160-312a-40e4-be0a-3c2d41043cfa.png) | ![diffuse spec](https://user-images.githubusercontent.com/28896013/144418867-7814ddd8-9c15-4fb8-869b-9497a8227f62.png) |

  

* Use [tinyobj](https://github.com/syoyo/tinyobjloader) to load arbitrary .obj model with its specular,diffuse, emission and bumping textures;

* Depth of field

  Focus on

  | With Depth-of-Field | Without Depth-of-Field |
  | :-----------------: | :--------------------: |
  | ![DOF](https://user-images.githubusercontent.com/28896013/146015776-922907b2-fe2c-4d04-b9fe-ae777caa5a23.png)|        |

  

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

