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

__host__ __device__ float triangleIntersectionLocalTest(Geom obj, Ray r, glm::vec3 p0, glm::vec3 p1, glm::vec3 p2,
    glm::vec3& intersectionPoint, glm::vec3& normal)
{
    //1. Ray-plane intersection
    glm::vec3 planeNormal = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    float t = glm::dot(planeNormal, (p0 - r.origin)) / glm::dot(planeNormal, r.direction);

    if (t < 0)
    {
        return -1;
    }

    glm::vec3 p = r.origin + t * r.direction;	// Intersection point

    //2. Barycentric test
    float S = 0.5f * glm::length(glm::cross(p0 - p1, p0 - p2));
    float s1 = 0.5f * glm::length(glm::cross(p - p1, p - p2)) / S;
    float s2 = 0.5f * glm::length(glm::cross(p - p2, p - p0)) / S;
    float s3 = 0.5f * glm::length(glm::cross(p - p0, p - p1)) / S;
    float sum = s1 + s2 + s3;

    if (s1 >= 0 && s1 <= 1 && s2 >= 0 && s2 <= 1 && s3 >= 0 && s3 <= 1 && abs(sum - 1.0f) < FLT_EPSILON) {
        intersectionPoint = p;
        normal = planeNormal;
        return t;
    }
    return -1;
}

__host__ __device__ float objIntersectionTest(Geom geom, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal) {
    float min_tri_t = FLT_MAX;
    glm::vec3 tmp_tri_intersect;
    glm::vec3 tmp_tri_normal;
    glm::vec3 min_tri_intersect;
    glm::vec3 min_tri_normal;


    // to object space
    glm::vec3 ro = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));

    for (int j = 0; j < geom.faceSize; j++) {
        Face& tri = geom.dev_faces[j];


        // check if there is an intersection
        bool did_isect = glm::intersectRayTriangle(ro, rd, tri.v0, tri.v1, tri.v2, tmp_tri_intersect);

        if (did_isect) {
            
            // to world space
            tmp_tri_normal = tri.normal;
            tmp_tri_intersect = glm::normalize(tmp_tri_intersect);

            // convert barycentric to local
            tmp_tri_intersect = ro + rd * tmp_tri_intersect.z;
            
            // local to world
            tmp_tri_intersect = multiplyMV(geom.transform, glm::vec4(tmp_tri_intersect, 1.f));
            float tmp_tri_t = glm::length(r.origin - tmp_tri_intersect);
            
                /*
            tmp_tri_normal = tri.normal;

            tmp_tri_intersect = glm::normalize(tmp_tri_intersect);
            tmp_tri_intersect = (tmp_tri_intersect.x * tri.v0) + (tmp_tri_intersect.y * tri.v1) + (tmp_tri_intersect.z * tri.v2);
            tmp_tri_intersect = multiplyMV(geom.transform, glm::vec4(tmp_tri_intersect,1.0f));
            
            float tmp_tri_t = glm::length(r.origin - tmp_tri_intersect);
            */
            if (tmp_tri_t < min_tri_t) {
                min_tri_intersect = tmp_tri_intersect;
                min_tri_normal = tmp_tri_normal;
                min_tri_t = tmp_tri_t;
            }
        }
    }
    if (min_tri_t > 0.0f) {
        intersectionPoint = min_tri_intersect;
        normal = multiplyMV(geom.invTranspose, glm::vec4(min_tri_normal, 0.0f));
        return  min_tri_t;
    }
    else {
        return -1;
    }
}