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
        ShadeableIntersection intersection,
        const Material &m,
        thrust::default_random_engine &rng,
        Geom* geoms,
        int iter,
        int depth) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (m.hasReflective>0) {

        glm::vec3 reflectDir= glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
        float spec = glm::pow(glm::max(glm::dot(-pathSegment.ray.direction,reflectDir),(float)0.0), m.specular.exponent);
        pathSegment.color *= m.hasReflective*spec*m.specular.color;
        //pathSegment.color *= m.specular.color;
        pathSegment.ray.origin = intersect+intersection.surfaceNormal*0.01f;
        pathSegment.ray.direction = reflectDir;
    }
    else if (m.hasRefractive>0) {
        float IoR1 = 1.0f;
        float IoR2 = m.indexOfRefraction;

        float cosTheta = glm::dot(-pathSegment.ray.direction , intersection.surfaceNormal);
        if (cosTheta < 0) {
            intersection.surfaceNormal *= -1;
            IoR1 = IoR2;
            IoR2 = 1.0f;
            cosTheta = abs(cosTheta);
        }

        float sinTheta = sqrt(1.0-cosTheta*cosTheta);

        if (IoR1 / IoR2 * sinTheta > 1.0f) {
            pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
        }
        else {
            float reflect_coeff0 = ((IoR1 - IoR2) / (IoR1 + IoR2)) * ((IoR1 - IoR2) / (IoR1 + IoR2));
            float reflect_coeff = reflect_coeff0 + (1.0f - reflect_coeff0) * pow((1.0-cosTheta), 5);

            thrust::uniform_real_distribution<float> u01(0, 1);
            float random = u01(rng);
            if (random < reflect_coeff) {
                pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
            }
            else {
                pathSegment.ray.direction = glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, IoR1/IoR2);
            }
        }
        pathSegment.color *= m.specular.color;
        pathSegment.ray.origin= intersect + pathSegment.ray.direction * 0.01f;
    }
    //cal for object color from texture specular plus diffuse
    else if (geoms[intersection.geomId].type == OBJ) {
        Geom geom = geoms[intersection.geomId];

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
        if (emission.x > FLT_EPSILON || emission.y > FLT_EPSILON || emission.z > FLT_EPSILON) {
            pathSegment.color *= (emission * 5.0f);
            pathSegment.remainingBounces = 1;
            return;
        }

        float IoR1 = 1.0f;
        float IoR2 = m.indexOfRefraction;
        float cosTheta = glm::dot(-pathSegment.ray.direction, intersection.surfaceNormal);
        float reflect_coeff0 = ((IoR1 - IoR2) / (IoR1 + IoR2)) * ((IoR1 - IoR2) / (IoR1 + IoR2));
        float reflect_coeff = reflect_coeff0 + (1.0f - reflect_coeff0) * pow((1.0 - cosTheta), 5);

        thrust::uniform_real_distribution<float> u01(0, 1);
        float random = u01(rng);

        if (random < reflect_coeff) {
            //specular color
            int coordU = (int)(intersection.texcoord.x * geom.ks.width);
            int coordV = (int)(intersection.texcoord.y * geom.ks.height);
            int pixelID = coordV * geom.ks.width + coordU;

            glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
            float spec = glm::pow(glm::max(glm::dot(-pathSegment.ray.direction, reflectDir), (float)0.0), 0.0f);
        
            glm::vec3 specColor;
            if (geom.ks.channels) {
                unsigned int colR = (unsigned int)geom.ks.image[pixelID * geom.ks.channels];
                unsigned int colG = (unsigned int)geom.ks.image[pixelID * geom.ks.channels + 1];
                unsigned int colB = (unsigned int)geom.ks.image[pixelID * geom.ks.channels + 2];
                specColor = glm::vec3(colR / 255.f, colG / 255.f, colB / 255.f);
            }
            else specColor = m.specular.color;
            specColor *= spec;
            pathSegment.color *= (specColor);
            pathSegment.ray.origin = intersect + intersection.surfaceNormal * 0.01f;
            pathSegment.ray.direction = reflectDir;

        }
        else {
            int coordU = (int)(intersection.texcoord.x * geom.kd.width);
            int coordV = (int)(intersection.texcoord.y * geom.kd.height);
            int pixelID = coordV * geom.kd.width + coordU;
            //diffuse color
            coordU = (int)(intersection.texcoord.x * geom.kd.width);
            coordV = (int)(intersection.texcoord.y * geom.kd.height);
            pixelID = coordV * geom.kd.width + coordU;
            glm::vec3 diffuseColor;
            if (geom.kd.channels) {
                unsigned int colR = (unsigned int)geom.kd.image[pixelID * geom.kd.channels];
                unsigned int colG = (unsigned int)geom.kd.image[pixelID * geom.kd.channels + 1];
                unsigned int colB = (unsigned int)geom.kd.image[pixelID * geom.kd.channels + 2];
                diffuseColor = glm::vec3(colR / 255.f, colG / 255.f, colB / 255.f);
            }
            else diffuseColor = m.color;
            pathSegment.color *= (diffuseColor);
            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
            pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
        }
        
    }
    //pure diffuse
    else{
#if JITTERED_SAMPLING
        if (depth == 1) {
            pathSegment.ray.direction = calculateJitteredDirectionHemisphere(normal, rng, iter, 5000);
        }
        else
        {
            pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        }
#else
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
#endif
        pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.01f;
        pathSegment.color *= m.color;
    }
}
