#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float boudingBoxIntersectionTest(Geom geom, Ray r)
{
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;

    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (geom.minPos[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (geom.maxPos[xyz] - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            if (ta > 0 && ta > tmin) {
                tmin = ta;
            }
            if (tb < tmax) {
                tmax = tb;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        return true;
    }
    return false;
}

__host__ __device__ float triangleIntersectionLocalTest(Geom obj, glm::vec3 ro, glm::vec3 rd, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2,
    glm::vec3& intersectionPoint, glm::vec3& normal)
{

    //1. Ray-plane intersection
    glm::vec3 planeNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    float t = glm::dot(planeNormal, (v0 - ro)) / glm::dot(planeNormal, rd);

    if (t < 0)
    {
        return -1;
    }

    glm::vec3 p = ro + t * rd;	// Intersection point

    //2. Barycentric test
    float S = 0.5f * glm::length(glm::cross(v0 - v1, v0 - v2));
    float s1 = 0.5f * glm::length(glm::cross(p - v1, p - v2)) / S;
    float s2 = 0.5f * glm::length(glm::cross(p - v2, p - v0)) / S;
    float s3 = 0.5f * glm::length(glm::cross(p - v0, p - v1)) / S;
    float sum = s1 + s2 + s3;

    if (s1 >= 0 && s1 <= 1 && s2 >= 0 && s2 <= 1 && s3 >= 0 && s3 <= 1 && abs(sum - 1.0f) < FLT_EPSILON) {
        intersectionPoint = p;
        normal = planeNormal;
        return t;
    }
    return -1;
}

__host__ __device__ float meshIntersectionTest(Geom geom, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& texcoord, bool& outside) {
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = FLT_MAX;
    int nearest = -1;

    for (int j = 0; j < geom.faceSize; j++) {
        Face& tri = geom.dev_faces[j];
        glm::vec3 bary;
        if (glm::intersectRayTriangle(q.origin, q.direction, tri.v0.position, tri.v1.position, tri.v2.position, bary)) {
            // Get the actual intersect from barycentric coordinates
            glm::vec3 p = (1 - bary[0] - bary[1]) * tri.v0.position + bary[0] * tri.v1.position + bary[1] * tri.v2.position;
            float t = glm::distance(p, q.origin);
            if (t < tmin) {
                tmin = t;
                nearest = j;
                texcoord= (1 - bary[0] - bary[1]) * tri.v0.texcoord + bary[0] * tri.v1.texcoord + bary[1] * tri.v2.texcoord;
                //texcoord= texcoord - glm::floor(texcoord);
            }
        }
    }
    if (nearest == -1) {
        return -1;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(q, tmin);

    glm::vec3 e1 = geom.dev_faces[nearest].v1.position - geom.dev_faces[nearest].v0.position;
    glm::vec3 e2 = geom.dev_faces[nearest].v2.position - geom.dev_faces[nearest].v0.position;
    glm::vec3 objspaceNormal = glm::normalize(glm::cross(e1, e2));

    intersectionPoint = multiplyMV(geom.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(objspaceNormal, 0.f)));
    outside = glm::dot(normal, r.direction) < 0;
    
    if (geom.type == OBJ && geom.bump.channels) {
        //modify normal with bump texture
        Texture bump = geom.bump;
        glm::vec2 deltaUV1 = geom.dev_faces[nearest].v1.texcoord - geom.dev_faces[nearest].v0.texcoord;
        glm::vec2 deltaUV2 = geom.dev_faces[nearest].v2.texcoord - geom.dev_faces[nearest].v0.texcoord;

        glm::vec3 tangent, bitangent;
        float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

        tangent.x = f * (deltaUV2.y * e1.x - deltaUV1.y * e2.x);
        tangent.y = f * (deltaUV2.y * e1.y - deltaUV1.y * e2.y);
        tangent.z = f * (deltaUV2.y * e1.z - deltaUV1.y * e2.z);
        tangent = glm::normalize(tangent);

        bitangent.x = f * (-deltaUV2.x * e1.x + deltaUV1.x * e2.x);
        bitangent.y = f * (-deltaUV2.x * e1.y + deltaUV1.x * e2.y);
        bitangent.z = f * (-deltaUV2.x * e1.z + deltaUV1.x * e2.z);
        bitangent = glm::normalize(bitangent);

        glm::vec3 T = normalize(glm::vec3(multiplyMV(geom.transform, glm::vec4(tangent, 0.f))));
        glm::vec3 B= normalize(glm::vec3(multiplyMV(geom.transform, glm::vec4(bitangent, 0.f))));
        glm::vec3 N = normal;

        glm::mat3 TBN = glm::mat3(T,B,N);

        int coordU = (int)(texcoord.x * bump.width);
        int coordV = (int)(texcoord.y * bump.height);
        int pixelID = coordV * bump.width + coordU;
        unsigned int colR = (unsigned int)bump.image[pixelID * bump.channels];
        unsigned int colG = (unsigned int)bump.image[pixelID * bump.channels + 1];
        unsigned int colB = (unsigned int)bump.image[pixelID * bump.channels + 2];
        glm::vec3 tangentSpaceNormal = normalize(glm::vec3(colR / 255.f, colG / 255.f, colB / 255.f));
        tangentSpaceNormal = normalize(tangentSpaceNormal*2.0f-1.0f);
        normal = normalize(glm::vec3(TBN * tangentSpaceNormal));
    }
    
    return tmin;
}

__host__ __device__ float objTriIntersectionTest(Geom geom, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    float min_tri_t = FLT_MAX;
    glm::vec3 tmp_tri_intersect;
    glm::vec3 tmp_tri_normal;
    glm::vec3 min_tri_intersect;
    glm::vec3 min_tri_normal;
    int nearest = -1;
    // to object space
    Ray q;
    q.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    for (int j = 0; j < geom.faceSize; j++) {
        Face& tri = geom.dev_faces[j];   
        float tmp_tri_t=triangleIntersectionLocalTest(geom, q.origin, q.direction, tri.v0.position, tri.v1.position, tri.v2.position, tmp_tri_intersect, tmp_tri_normal);
        if(tmp_tri_t>0 && tmp_tri_t < min_tri_t){
            min_tri_intersect = tmp_tri_intersect;
            min_tri_normal = tmp_tri_normal;
            min_tri_t = tmp_tri_t;
            nearest = j;
        }
    }
    if (nearest == -1) {
        return -1;
    }
    intersectionPoint = multiplyMV(geom.transform, glm::vec4(min_tri_intersect, 1.f));
    normal = glm::normalize(multiplyMV(geom.invTranspose, glm::vec4(min_tri_normal, 0.f)));
    outside = glm::dot(normal, r.direction) < 0;
    return  min_tri_t;

}