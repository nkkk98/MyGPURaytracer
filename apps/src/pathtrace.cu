#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <stream_compaction/cpu.h>
#include <stream_compaction/naive.h>
#include <stream_compaction/efficient.h>
#include <stream_compaction/thrust.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include "timer.h"
PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}
#define ERRORCHECK 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define DEPTH_OF_FIELD 0
#define CACHE_FIRST_BOUNCE 1
#define SORT_BY_MATERIAL 1
#define ANTIALIASING 1
#define BOUNDING_BOX 0

#define AI_DENOISE 0

#define WEIGHTS 1

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the denoised image to the OpenGL PBO directly.
__global__ void sendDenosiedImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x  * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y  * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z  * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}


//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
    int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        // visualize t
        /*int index = x + (y * resolution.x);
        float timeToIntersect = gBuffer[index].t * 256.0;
        pbo[index].w = 0;
        pbo[index].x = timeToIntersect;
        pbo[index].y = timeToIntersect;
        pbo[index].z = timeToIntersect;*/

        // visualize position
        /*int index = x + (y * resolution.x);
        glm::vec3 color = glm::clamp(glm::abs(gBuffer[index].position * 25.f), 0.f, 255.f);
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;*/

        // visualize normal
        int index = x + (y * resolution.x);
        glm::vec3 color = glm::clamp(glm::abs(gBuffer[index].normal * 255.f), 0.f, 255.f);

        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void denoiseIteration(int step, float c_weight, float p_weight, float n_weight, glm::ivec2 resolution, GBufferPixel* gBuffer) {

    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x >= resolution.x && y >= resolution.y) {
        return;
    }

    //kernel
    float kernel[25] = { 1.f / 16.f, 1.f / 16.f , 1.f / 16.f , 1.f / 16.f , 1.f / 16.f,
                        1.f / 16.f, 1.f / 4.f , 1.f / 4.f , 1.f / 4.f , 1.f / 16.f,
                        1.f / 16.f, 1.f / 4.f , 3.f / 8.f , 1.f / 4.f , 1.f / 16.f,
                        1.f / 16.f, 1.f / 4.f , 1.f / 4.f , 1.f / 4.f , 1.f / 16.f,
                        1.f / 16.f, 1.f / 16.f , 1.f / 16.f , 1.f / 16.f , 1.f / 16.f };

    //offset
    glm::ivec2 offset[25] = { glm::ivec2(-2, 2) ,glm::ivec2(-1, 2), glm::ivec2(0, 2) , glm::ivec2(1, 2) ,glm::ivec2(2, 2),
                        glm::ivec2(-2, 1) ,glm::ivec2(-1, 1), glm::ivec2(0, 1) , glm::ivec2(1, 1) ,glm::ivec2(2, 1),
                         glm::ivec2(-2, 0) , glm::ivec2(-1, 0) , glm::ivec2(0, 0) , glm::ivec2(1, 0) , glm::ivec2(2, 0),
                        glm::ivec2(-2, -1) ,glm::ivec2(-1, -1), glm::ivec2(0, -1) , glm::ivec2(1, -1) ,glm::ivec2(2, -1),
                        glm::ivec2(-2, -2) ,glm::ivec2(-1, -2), glm::ivec2(0, -2) , glm::ivec2(1, -2) ,glm::ivec2(2, -2) };

    glm::vec3 sum = glm::vec3(0.f);
    glm::vec3 curr_pos = gBuffer[index].position;
    glm::vec3 curr_nor = gBuffer[index].normal;
    glm::vec3 curr_color = gBuffer[index].denoise_color;

    float cum_w = 0.f;
    float weight;
    for (int i = 0; i < 25; i++) {
        glm::ivec2 temp_cords = glm::ivec2(x, y);
        temp_cords += offset[i] * step;
        temp_cords.x = glm::clamp(temp_cords.x, 0, resolution.x - 1);
        temp_cords.y = glm::clamp(temp_cords.y, 0, resolution.y - 1);
        if (temp_cords.x < resolution.x && temp_cords.y < resolution.y) {
            int temp_index = temp_cords.x + (temp_cords.y * resolution.x);

            glm::vec3 temp_color = gBuffer[temp_index].denoise_color;
            glm::vec3 t = curr_color - temp_color;
            float dist2 = glm::dot(t, t);
            float color_weight = glm::min(glm::exp(-dist2 / c_weight), 1.f);

            glm::vec3 temp_nor = gBuffer[temp_index].normal;
            t = curr_nor - temp_nor;
            dist2 = glm::dot(t, t);
            float nor_weight = glm::min(glm::exp(-dist2 / n_weight), 1.f);

            glm::vec3 temp_pos = gBuffer[temp_index].position;
            t = curr_pos - temp_pos;
            dist2 = glm::dot(t, t);
            float pos_weight = glm::min(glm::exp(-dist2 / p_weight), 1.f);


#if WEIGHTS
            weight =color_weight;
            sum += temp_color * weight * kernel[i];
            cum_w += weight * kernel[i];
#else
            sum += temp_color * kernel[i];
            cum_w += kernel[i];
#endif
        }
    }
    gBuffer[index].updated_denoise_color = sum / cum_w;

}

__global__ void pingPongGbuffer(glm::ivec2 resolution, GBufferPixel* gBuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);

    if (x < resolution.x && y < resolution.y) {
        gBuffer[index].denoise_color = gBuffer[index].updated_denoise_color;
    }

}

void denoiseGbuffer(const dim3 blocksPerGrid2d, const dim3 blockSize2d, glm::ivec2 resolution, float c_weight, float p_weight, float n_weight, int logFilterSize, GBufferPixel* gBuffer) {
    int step = 1;
    for (int i = 0; i < logFilterSize; i++) {
        denoiseIteration << <blocksPerGrid2d, blockSize2d >> > (step, c_weight, p_weight, n_weight, resolution, gBuffer);
        pingPongGbuffer << <blocksPerGrid2d, blockSize2d >> > (resolution, gBuffer);
        step *= 2;
    }
}


static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
static ShadeableIntersection* dev_first_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static glm::vec3 * dev_albedo = NULL;
static glm::vec3 * dev_denoised_output = NULL;


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    for (int i = 0; i < scene->geoms.size(); i++)
    {
        Geom& geom = scene->geoms[i];
        cudaMalloc(&geom.dev_faces, geom.faceSize * sizeof(Face));
        cudaMemcpy(geom.dev_faces, (scene->allFaces[i]).data(), geom.faceSize * sizeof(Face), cudaMemcpyHostToDevice);

        geom.kd.channels = scene->kdTextures[i].channels;
        geom.kd.width = scene->kdTextures[i].width;
        geom.kd.height = scene->kdTextures[i].height;
        cudaMalloc(&geom.kd.image, scene->kdTextures[i].width * scene->kdTextures[i].height * scene->kdTextures[i].channels * sizeof(unsigned char));
        cudaMemcpy(geom.kd.image, scene->kdTextures[i].image, scene->kdTextures[i].width * scene->kdTextures[i].height * scene->kdTextures[i].channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

        geom.ks.channels = scene->ksTextures[i].channels;
        geom.ks.width = scene->ksTextures[i].width;
        geom.ks.height = scene->ksTextures[i].height;
        cudaMalloc(&geom.ks.image, scene->ksTextures[i].width * scene->ksTextures[i].height * scene->ksTextures[i].channels * sizeof(unsigned char));
        cudaMemcpy(geom.ks.image, scene->ksTextures[i].image, scene->ksTextures[i].width * scene->ksTextures[i].height * scene->ksTextures[i].channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

        geom.ke.channels = scene->keTextures[i].channels;
        geom.ke.width = scene->keTextures[i].width;
        geom.ke.height = scene->keTextures[i].height;
        cudaMalloc(&geom.ke.image, scene->keTextures[i].width * scene->keTextures[i].height * scene->keTextures[i].channels * sizeof(unsigned char));
        cudaMemcpy(geom.ke.image, scene->keTextures[i].image, scene->keTextures[i].width * scene->keTextures[i].height * scene->keTextures[i].channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

        geom.bump.channels = scene->bumpTextures[i].channels;
        geom.bump.width = scene->bumpTextures[i].width;
        geom.bump.height = scene->bumpTextures[i].height;
        cudaMalloc(&geom.bump.image, scene->bumpTextures[i].width * scene->bumpTextures[i].height * scene->bumpTextures[i].channels * sizeof(unsigned char));
        cudaMemcpy(geom.bump.image, scene->bumpTextures[i].image, scene->bumpTextures[i].width * scene->bumpTextures[i].height * scene->bumpTextures[i].channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

#if CACHE_FIRST_BOUNCE
        cudaMalloc(&dev_first_intersections, pixelcount* sizeof(ShadeableIntersection));
        cudaMemset(dev_first_intersections, 0, pixelcount* sizeof(ShadeableIntersection));
#endif

#if AI_DENOISE
        cudaMalloc(&dev_albedo, pixelcount * sizeof(glm::vec3));
        cudaMemset(dev_albedo, 0, pixelcount * sizeof(glm::vec3));
        cudaMalloc(&dev_denoised_output, pixelcount * sizeof(glm::vec3));
        cudaMemset(dev_denoised_output, 0, pixelcount * sizeof(glm::vec3));
#endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    if (hst_scene != NULL) {
        for (int i = 0; i < hst_scene->geoms.size(); i++)
        {
            Geom& geom = hst_scene->geoms[i];
            cudaFree(geom.dev_faces);
            cudaFree(geom.kd.image);
            cudaFree(geom.ks.image);
            cudaFree(geom.ke.image);
            cudaFree(geom.bump.image);
        }
    }
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_gBuffer);
#if CACHE_FIRST_BOUNCE
        cudaFree(dev_first_intersections);
#endif

#if AI_DENOISE
        cudaFree(dev_albedo);
        cudaFree(dev_denoised_output);
#endif
    checkCUDAError("pathtraceFree");
}

__global__ void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    GBufferPixel* gBuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        float t = shadeableIntersections[idx].t;
        gBuffer[idx].t = t;
        gBuffer[idx].normal = shadeableIntersections[idx].surfaceNormal;
        gBuffer[idx].position = pathSegments[idx].ray.origin + pathSegments[idx].ray.direction * t;
    }
}


__host__ __device__ glm::vec2 ConcentricSampleDisk(const glm::vec2 &point) {
    glm::vec2 uOffset = 2.f * point - glm::vec2(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
        return glm::vec2(0, 0);
    float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = 0.785398f * (uOffset.y / uOffset.x);
    }
    else {
        r = uOffset.y;
        theta = 1.570796f - 0.785398f * (uOffset.x / uOffset.y);
    }
    return r * glm::vec2(std::cos(theta), std::sin(theta));
}
/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;


	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        float antia_x = x;
        float antia_y = y;
#if ANTIALIASING
        thrust::default_random_engine rngANTIA = makeSeededRandomEngine(iter, index, traceDepth);
        thrust::uniform_real_distribution<float> uANTIA(-0.5, 0.5);

        antia_x += uANTIA(rngANTIA);
        antia_y += uANTIA(rngANTIA);

#endif
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)antia_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)antia_y - (float)cam.resolution.y * 0.5f)
			);
#if DEPTH_OF_FIELD
        float lensRadius = 0.8f;
        float focalDistance = 11.0f;

        thrust::uniform_real_distribution<float> uDOF(0,1);

        if (lensRadius > 0) {
            glm::vec2 pLens = lensRadius * ConcentricSampleDisk(glm::vec2(uDOF(rng), uDOF(rng)));

            float ft = glm::abs(focalDistance / segment.ray.direction.z);
            glm::vec3 pFocus = segment.ray.origin+segment.ray.direction*ft;

            segment.ray.origin += glm::vec3(pLens.x, pLens.y, 0);
            segment.ray.direction = normalize(pFocus- segment.ray.origin);
        }
#endif
		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;
        glm::vec2 uv = glm::vec2(0.0f,0.0f);

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
        glm::vec2 tmp_uv;

		// naive parse through global geoms
        //TODO BVH
		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];
            int a = geom.faceSize;
            int b = geom.type;
			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?
            else if (geom.type == OBJ)
            {
#if BOUNDING_BOX
                if (boudingBoxIntersectionTest(geom, pathSegment.ray)) {
                    t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                }
                else t = -1;
#endif
                //Although self-writing triangle intersect can also be used to cal t, it's slower than glm::intersectRayTriangle
                //t = objTriIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
                t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, outside);
            }
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
                uv = tmp_uv;
                int x = uv.x;
                int y = uv.y;
			}
		}

		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
            intersections[path_index].geomId = hit_geom_index;
            intersections[path_index].texcoord = uv;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
    , Geom* geoms
    , int depth
    , glm::vec3* albedo
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
#if AI_DENOISE
    if (iter == 1 && depth==1) {
        if (intersection.t > 0.0f) {
            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;
            albedo[pathSegments[idx].pixelIndex] = materialColor;
            
            Geom geom = geoms[intersection.geomId];
            if (geom.type == OBJ) {
                glm::vec3 emission(0.0f);
                if (geom.ke.channels) {
                    int coordU = (int)(intersection.texcoord.x * geom.ke.width);
                    int coordV = (int)(intersection.texcoord.y * geom.ke.height);
                    int pixelID = coordV * geom.ke.width + coordU;

                    unsigned int colR = (unsigned int)geom.ke.image[pixelID * geom.ke.channels];
                    unsigned int colG = (unsigned int)geom.ke.image[pixelID * geom.ke.channels + 1];
                    unsigned int colB = (unsigned int)geom.ke.image[pixelID * geom.ke.channels + 2];
                    emission = glm::vec3(colR / 255.f, colG / 255.f, colB / 255.f);
                }
                //if emittance load emission, else load diffuse color
                if (emission.x > FLT_EPSILON || emission.y > FLT_EPSILON || emission.z > FLT_EPSILON) {
                    albedo[pathSegments[idx].pixelIndex]= (emission * 5.0f);
                }else if (geom.kd.channels) {
                    int coordU = (int)(intersection.texcoord.x * geom.kd.width);
                    int coordV = (int)(intersection.texcoord.y * geom.kd.height);
                    int pixelID = coordV * geom.kd.width + coordU;
                    //diffuse color
                    coordU = (int)(intersection.texcoord.x * geom.kd.width);
                    coordV = (int)(intersection.texcoord.y * geom.kd.height);
                    pixelID = coordV * geom.kd.width + coordU;
                    glm::vec3 diffuseColor;
                    unsigned int colR = (unsigned int)geom.kd.image[pixelID * geom.kd.channels];
                    unsigned int colG = (unsigned int)geom.kd.image[pixelID * geom.kd.channels + 1];
                    unsigned int colB = (unsigned int)geom.kd.image[pixelID * geom.kd.channels + 2];
                    diffuseColor = glm::vec3(colR / 255.f, colG / 255.f, colB / 255.f);
                    albedo[pathSegments[idx].pixelIndex] = diffuseColor;
                }
            }else if (material.emittance > 0.0f) {
                albedo[pathSegments[idx].pixelIndex] = materialColor * material.emittance;
            }
            else if (material.hasRefractive > 0.0f) {
                albedo[pathSegments[idx].pixelIndex] = material.specular.color;
            }
        }
        else {
            albedo[pathSegments[idx].pixelIndex] = glm::vec3(0.0f);
        }
         
    }
#endif
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        pathSegments[idx].remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else if (pathSegments[idx].remainingBounces == 1) {
          pathSegments[idx].color = glm::vec3(0.0);
          pathSegments[idx].remainingBounces = 0;
      }
      else {
          scatterRay(pathSegments[idx], pathSegments[idx].ray.origin+intersection.t* pathSegments[idx].ray.direction, intersection, material, rng, geoms, iter, depth);
          pathSegments[idx].remainingBounces -= 1;
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = 0;
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

__global__ void pathToBuffer(GBufferPixel* gBuffer, int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        gBuffer[iterationPath.pixelIndex].denoise_color = iterationPath.color;
    }
}

__global__ void finalGatherDeniose(GBufferPixel* gBuffer, int nPaths, glm::vec3* image, PathSegment* iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += gBuffer[iterationPath.pixelIndex].denoise_color;
    }
}


struct sortByMaterial {
    __host__ __device__ bool operator() (const ShadeableIntersection& a, const ShadeableIntersection& b){
        return a.materialId > b.materialId;
    }
};

struct isTerminate {
    __host__ __device__ bool operator()(const PathSegment& p) {
        return p.remainingBounces > 0;
    }
};
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, bool denoise, int filterSize, float c_weight, float p_weight, float n_weight) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
    int num_paths_origin = num_paths;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    
    timer().startGpuTimer();
    // Empty gbuffer
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));
#if CACHE_FIRST_BOUNCE && !ANTIALIASING && !DEPTH_OF_FIELD
        if (depth==0 && iter!=1) {
            thrust::copy(thrust::device, dev_first_intersections, dev_first_intersections+ num_paths_origin, dev_intersections);
#if SORT_BY_MATERIAL
                thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths_origin, dev_paths, sortByMaterial());
#endif
        }
#endif
            // clean shading chunks
     cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
     bool iterationComplete = false;
     while (!iterationComplete) {
            dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
            // tracing
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
#if CACHE_FIRST_BOUNCE && !ANTIALIASING && !DEPTH_OF_FIELD
            if (iter == 1 && depth == 0)thrust::copy(thrust::device, dev_intersections, dev_intersections + num_paths_origin, dev_first_intersections);
#endif
#if SORT_BY_MATERIAL
            thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, sortByMaterial());
#endif
            if (depth == 0) {
                generateGBuffer << <numblocksPathSegmentTracing, blockSize1d >> > (num_paths, dev_intersections, dev_paths, dev_gBuffer);
            }

            depth++;


	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.
#if AI_DENOISE
    shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
        iter,
        num_paths,
        dev_intersections,
        dev_paths,
        dev_materials,
        dev_geoms,
        depth,
        dev_albedo
        );
#else  
    shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials,
    dev_geoms,
    depth,
    NULL
  );
#endif
  //thrust::remove_if(dev_paths, dev_paths+num_paths, isTerminate());
  dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + num_paths, isTerminate());
  num_paths= dev_path_end - dev_paths;
  if(num_paths==0)iterationComplete = true; // TODO: should be based off stream compaction results.
	}
    timer().endGpuTimer();
  // Assemble this iteration and apply it to the image
    if (true) {
        pathToBuffer << <numBlocksPixels, blockSize1d >> > (dev_gBuffer, num_paths_origin, dev_image, dev_paths);
        denoiseGbuffer(blocksPerGrid2d, blockSize2d, cam.resolution, c_weight, p_weight, n_weight, filterSize, dev_gBuffer);
        finalGatherDeniose << <numBlocksPixels, blockSize1d >> > (dev_gBuffer, num_paths_origin, dev_image, dev_paths);
    }
    else {
        finalGather << <numBlocksPixels, blockSize1d >> > (num_paths_origin, dev_image, dev_paths);
    }

    ///////////////////////////////////////////////////////////////////////////
#if !AI_DENOISE
    // Send results to OpenGL buffer for rendering
    //sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
#endif
    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

#if AI_DENOISE
    cudaMemcpy(hst_scene->state.albedo.data(), dev_albedo,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
#endif
    checkCUDAError("pathtrace copy");
}

void sendToGPU(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;
    cudaMemcpy(dev_denoised_output, hst_scene->state.output.data(), pixelcount*sizeof(glm::vec3),cudaMemcpyHostToDevice);

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    sendDenosiedImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_denoised_output);
}

void showGBuffer(uchar4* pbo) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
    gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer);
}

void showImage(uchar4* pbo, int iter) {
    const Camera& cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);
}