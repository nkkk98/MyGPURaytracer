#pragma once
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "stb_image.h"
using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadObj(string filename, Geom& newGeom);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Texture> kdTextures;
    std::vector<Texture> ksTextures;
    std::vector<Texture> bumpTextures;
    std::vector<Texture> keTextures;

    std::vector<std::vector<Face>> allFaces;
    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
