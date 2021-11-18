#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE,
    OBJ,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 color;
    glm::vec2 texcoord;
};

struct Face {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    glm::vec3 normal;
};

struct Texture {
    int width;
    int height;
    unsigned char* image;
    int channels;

    Texture() {
        width = 0;
        height = 0;
        channels = 0;
        image = NULL;
    }
};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    int faceSize;
    Face* dev_faces;

    Texture kd;
    Texture ks;
    Texture bump;
    Texture ke;

    glm::vec3 minPos;
    glm::vec3 maxPos;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::vector<glm::vec3> albedo;
    std::vector<glm::vec3> output;
    std::string imageName;
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  glm::vec2 texcoord;
  int geomId;
};

struct GBufferPixel {
    float t;
    glm::vec3 normal;
    glm::vec3 position;
    glm::vec3 denoise_color;
    glm::vec3 updated_denoise_color;
};