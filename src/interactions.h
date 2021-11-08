#pragma once

#include "intersections.h"

#define JITTERED_SAMPLING 0
// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}


__host__ __device__ glm::vec3 calculateJitteredDirectionHemisphere(
    glm::vec3 normal, thrust::default_random_engine& rng, int iter, int max_iter) {

    int samples = max_iter;
    int sqrtVal = (int)(sqrt((float)samples) + 0.5f);
    float invSqrtVal = 1.f / (float)sqrtVal;

    int x = iter % sqrtVal;
    int y = (float)(iter) / (float)sqrtVal;

    thrust::uniform_real_distribution<float> u01(0, 1);
    float x_point = glm::clamp((x + u01(rng)) * invSqrtVal, 0.f, 1.f);
    float y_point = glm::clamp((y + u01(rng)) * invSqrtVal, 0.f, 1.f);

    float up = sqrt(y_point); // cos(theta)
    float over = sqrt(1.f - (up * up)); // sin(theta)
    float around = x_point * TWO_PI;


    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}
/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 * 
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng,
        int iter,
        int depth) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (m.hasReflective>0) {

        glm::vec3 reflectDir= glm::reflect(pathSegment.ray.direction, normal);
        float spec = glm::pow(glm::max(glm::dot(-pathSegment.ray.direction,reflectDir),(float)0.0), m.specular.exponent);
        pathSegment.color *= m.hasReflective*spec*m.specular.color;
        //pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect+normal*0.01f;
        pathSegment.ray.direction = reflectDir;
    }
    else if (m.hasRefractive>0) {
        float IoR1 = 1.0f;
        float IoR2 = m.indexOfRefraction;

        float cosTheta = glm::dot(-pathSegment.ray.direction , normal);
        if (cosTheta < 0) {
            normal *= -1;
            IoR1 = IoR2;
            IoR2 = 1.0f;
            cosTheta = abs(cosTheta);
        }

        float sinTheta = sqrt(1.0-cosTheta*cosTheta);

        if (IoR1 / IoR2 * sinTheta > 1.0f) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
        }
        else {
            float reflect_coeff0 = ((IoR1 - IoR2) / (IoR1 + IoR2)) * ((IoR1 - IoR2) / (IoR1 + IoR2));
            float reflect_coeff = reflect_coeff0 + (1.0f - reflect_coeff0) * pow((1.0-cosTheta), 5);

            thrust::uniform_real_distribution<float> u01(0, 1);
            float random = u01(rng);
            if (random < reflect_coeff) {
                pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
            }
            else {
                pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, normal, IoR1/IoR2);
            }
        }
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin= intersect + pathSegment.ray.direction * 0.01f;
    }
    //diffuse
    else {
#if JITTERED_SAMPLING
        if (depth == 1) {
            pathSegment.ray.direction = calculateJitteredDirectionHemisphere(normal, rng, iter, 5000);
        }
        else
        {
            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        }
#endif
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
        pathSegment.color *= m.color;
    }
}
