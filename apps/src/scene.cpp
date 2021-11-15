#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }
}

int Scene::loadObj(string inputfile, Geom& newGeom) {

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "../models/materials";

    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& objMaterials = reader.GetMaterials();

    float minX = FLT_MAX;
    float maxX = FLT_MAX;
    float minY = FLT_MAX;
    float maxY = FLT_MIN;
    float minZ = FLT_MIN;
    float maxZ = FLT_MIN;

    std::vector<Face> faces;
    int geomMaterialID=0;//Assump all meshes of the model have the same material.
    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++) {
        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            Face newTri;

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++) {
                // access to vertex
                Vertex newV;
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                newV.position= glm::vec3(vx, vy, vz);

                if (idx.normal_index >= 0) {
                    tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
                    tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
                    tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];
                    newV.normal = glm::vec3(nx,ny,nz);
                }

                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                if (idx.texcoord_index >= 0) {
                    tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
                    tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];
                    newV.texcoord = glm::vec2(tx,ty);
                }
                // Optional: vertex colors
                // tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
                // tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
                // tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];
                if (v == 0) { newTri.v0 = newV; }
                else if (v == 1) { newTri.v1 = newV; }
                else if (v == 2) { newTri.v2 = newV; }

                // update bounding box
                if (vx < minX) { minX = vx; }
                if (vy < minY) { minY = vy; }
                if (vz < minZ) { minZ = vz; }
                if (vx > maxX) { maxX = vx; }
                if (vy > maxY) { maxY = vy; }
                if (vz > maxZ) { maxZ = vz; }
            }

            index_offset += fv;

            newTri.normal = glm::normalize(glm::cross(newTri.v2.position - newTri.v0.position, newTri.v1.position - newTri.v0.position));

            // per-face material
            shapes[s].mesh.material_ids[f];
            faces.push_back(newTri);
        }
    }
    newGeom.minPos = glm::vec3(minX, minY, minZ);
    newGeom.maxPos = glm::vec3(maxX, maxY, maxZ);
    newGeom.faceSize = faces.size();
    // std::cout << "triangle size: " << tris.size() << std::endl;
    allFaces.push_back(faces);

    //load textures for this object
    stbi_set_flip_vertically_on_load(true);
    tinyobj::material_t tm = objMaterials[geomMaterialID];
    int width, height, nrChannels;
    unsigned char* data;
    //Kd
    if (tm.diffuse_texname!="") {
        data = stbi_load(tm.diffuse_texname.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            Texture t;
            t.height = height;
            t.width = width;
            t.image = data;
            t.channels = nrChannels;
            kdTextures.push_back(t);
            cout << "kdTextures are: " << tm.diffuse_texname << endl;
            cout << "height: " << height << " width:" << width << " channels:" << nrChannels << " sample value:" << (unsigned int)data[2055*3]<<" "<< (unsigned int)data[2055*3+1]<<" " << (unsigned int)data[2055*3+2] <<endl;
        }
        else {
            Texture* t = new Texture();
            kdTextures.push_back(*t);
            cout << "Failed to load Kd texture file " << tm.diffuse_texname << endl;
        }
    }
    //Ks
    if (tm.specular_texname != "") {
        data = stbi_load(tm.specular_texname.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            Texture t;
            t.height = height;
            t.width = width;
            t.image = data;
            t.channels = nrChannels;
            ksTextures.push_back(t);
            cout << "ksTextures are " << tm.specular_texname << endl;
            cout << "height: " << height << " width:" << width << " channels:" << nrChannels << " char size:" << sizeof(data)<<endl;


        }
        else {
            Texture* t = new Texture();
            ksTextures.push_back(*t);
            cout<< "Failed to load Ks texture file " << tm.specular_highlight_texname << endl;
        }
    }
    //Ke
    if (tm.emissive_texname != "") {
        data = stbi_load(tm.emissive_texname.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            Texture t;
            t.height = height;
            t.width = width;
            t.image = data;
            t.channels = nrChannels;
            keTextures.push_back(t);
            cout << "keTextures are " << tm.emissive_texname << endl;
            cout << "height: " << height << " width:" << width << " channels:" << nrChannels << " sample value:" << (unsigned int)data[512 * 3*2] << " " << (unsigned int)data[512*2 * 3 + 1] << " " << (unsigned int)data[512*2 * 3 + 2] << endl;


        }
        else {
            Texture* t = new Texture();
            keTextures.push_back(*t);
            cout << "Failed to load Ks texture file " << tm.emissive_texname << endl;
        }
    }
    //Bump
    if (tm.bump_texname != "") {
        data = stbi_load(tm.bump_texname.c_str(), &width, &height, &nrChannels, 0);
        if (data) {
            Texture t;
            t.height = height;
            t.width = width;
            t.image = data;
            t.channels = nrChannels;
            bumpTextures.push_back(t);
            cout << "bumpTextures are " << tm.bump_texname << endl;
            cout << "height: " << height << " width:" << width << " channels:" << nrChannels << " sample value:" << (unsigned int)data[2055 * 3] << " " << (unsigned int)data[2055 * 3 + 1] << " " << (unsigned int)data[2055 * 3 + 2] << endl;


        }
        else {
            Texture* t = new Texture();
            bumpTextures.push_back(*t);
            cout << "Failed to load Bump texture file " << tm.bump_texname << endl;
        }
    }

    //New material for this object
    Material newMaterial;
    newMaterial.specular.color = glm::vec3(tm.specular[0], tm.specular[1], tm.specular[2]);
    newMaterial.specular.exponent = 0.0f;
    newMaterial.color= glm::vec3(tm.diffuse[0], tm.diffuse[1], tm.diffuse[2]);
    newMaterial.indexOfRefraction = tm.ior;
    newMaterial.emittance = tm.emission[0];
    newMaterial.hasReflective = 0.0f;
    newMaterial.hasRefractive = 0.0f;

    materials.push_back(newMaterial);
    newGeom.materialid = materials.size() - 1;
    geoms.push_back(newGeom);
    return 1;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        string objFileName;
        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }else if (strcmp(line.c_str(), "triangle") == 0) {
                cout << "Creating new triangle..." << endl;
                newGeom.type = TRIANGLE;
            }else if (strcmp(line.c_str(), "obj") == 0) {
                cout << "Creating new object..." << endl;
                newGeom.type = OBJ;

                //load object file name
                utilityCore::safeGetline(fp_in, line);
                if (!line.empty() && fp_in.good()) {
                    objFileName = line.c_str();
                    cout << "Loading object file: " << objFileName << endl;
                }
            }

        }

        //link material for non-obj file
        if (newGeom.type != OBJ) {
            utilityCore::safeGetline(fp_in, line);
            if (!line.empty() && fp_in.good()) {
                vector<string> tokens = utilityCore::tokenizeString(line);
                newGeom.materialid = atoi(tokens[1].c_str());
                cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
            }
        }
        else newGeom.materialid = -1;

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (newGeom.type == OBJ) {
            return loadObj(objFileName, newGeom);
        }
        else {
            newGeom.faceSize = 0;
            geoms.push_back(newGeom);
            std::vector<Face> faces;
            allFaces.push_back(faces);
            Texture* t=new Texture();
            kdTextures.push_back(*t);
            ksTextures.push_back(*t);
            bumpTextures.push_back(*t);
            keTextures.push_back(*t);
            return 1;
        }
    }
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}
