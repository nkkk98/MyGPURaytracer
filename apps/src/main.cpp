#include "main.h"
#include "preview.h"
#include <cstring>
#include <OpenImageDenoise/oidn.hpp>

#include "common/timer.h"
#include "apps/utils/image_io.h"
#include "apps/utils/arg_parser.h"

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"
OIDN_NAMESPACE_USING
using namespace oidn;

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

int ui_iterations = 0;
int startupIterations = 0;
int lastLoopIterations = 0;
bool ui_denoise = false;
int ui_filterSize = 80;
float ui_colorWeight = 0.45f;
float ui_normalWeight = 0.35f;
float ui_positionWeight = 0.2f;
bool ui_saveAndExit = false;
bool ui_showGbuffer = false;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;

int width;
int height;

double totalTime = 0.0f;

#define AI_DENOISE 0

//-------------------------------
//-------------MAIN--------------
//-------------------------------
std::vector<glm::vec3> inputColor;

int main(int argc, char** argv) {

    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    ui_iterations = renderState->iterations;
    startupIterations = ui_iterations;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}

void errorCallback(void* userPtr, Error error, const char* message)
{
    throw std::runtime_error(message);
}

volatile bool isCancelled = false;

void signalHandler(int signal)
{
    isCancelled = true;
}

bool progressCallback(void* userPtr, double n)
{
    if (isCancelled)
    {
        std::cout << std::endl;
        return false;
    }
    std::cout << "\rDenoising " << int(n * 100.) << "%" << std::flush;
    return true;
}

void saveImage() {
    float samples = iteration;
    // output image file
    image img(width, height);
    image alb(width, height);
    image out(width, height);
    image input(width, height);
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);

            glm::vec3 pixa = renderState->albedo[index];
            alb.setPixel(width - 1 - x, y, glm::vec3(pixa));

            glm::vec3 pixo = renderState->output[index];
            out.setPixel(width - 1 - x, y, glm::vec3(pixo));

            glm::vec3 pixi = inputColor[index];
            input.setPixel(width - 1 - x, y, glm::vec3(pixi));
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    std::string albedo_filename = renderState->imageName;
    std::ostringstream ss1;
    ss1 << albedo_filename << "." << startTimeString << "." << samples << "albedo";
    albedo_filename = ss1.str();

    std::string out_filename = renderState->imageName;
    std::ostringstream ss2;
    ss2 << out_filename << "." << startTimeString << "." << samples << "output";
    out_filename = ss2.str();

    std::string input_filename = renderState->imageName;
    std::ostringstream ss3;
    ss3 << input_filename << "." << startTimeString << "." << samples << "input";
    input_filename = ss3.str();
    // CHECKITOUT
    img.savePNG(filename);
    alb.savePNG(albedo_filename);
    out.savePNG(out_filename);
    input.savePNG(input_filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file

}

void CPUdenoise() {

    DeviceType deviceType = DeviceType::Default;
    std::string filterType = "RT";
    // Initialize the denoising device
    std::cout << "Initializing device" << std::endl;
    Timer timer;

    DeviceRef device = newDevice(deviceType);

    const char* errorMessage;
    if (device.getError(errorMessage) != Error::None)
        throw std::runtime_error(errorMessage);
    device.setErrorFunction(errorCallback);

    device.commit();

    const double deviceInitTime = timer.query();

    std::cout << "  device=" << (deviceType == DeviceType::Default ? "default" : (deviceType == DeviceType::CPU ? "CPU" : "unknown"))
        << ", msec=" << (1000. * deviceInitTime) << std::endl;
    //denoise with albedo auxiliary image
    FilterRef filter = device.newFilter(filterType.c_str());

    inputColor = renderState->image;
    
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            inputColor[index] = glm::vec3(pix)/(float)iteration;
        }
    }

    filter.setImage("color", inputColor.data(), oidn::Format::Float3, width, height); // beauty
    filter.setImage("albedo", renderState->albedo.data(), oidn::Format::Float3, width, height); // auxiliary
    filter.setImage("output", renderState->output.data(), oidn::Format::Float3, width, height); // denoised beauty

    filter.commit();

    const double filterInitTime = timer.query();

    std::cout << "  filter=" << filterType
        << ", msec=" << (1000. * filterInitTime) << std::endl;

    std::cout << "Denoising" << std::endl;
    timer.reset();

    filter.execute();

    const double denoiseTime = timer.query();
    std::cout << "  msec=" << (1000. * denoiseTime) << std::endl;
}

void runCuda() {
    if (lastLoopIterations != ui_iterations) {
        lastLoopIterations = ui_iterations;
        camchanged = true;
    }

    if (camchanged) {
        iteration = 0;
        Camera &cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
      }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

    uchar4* pbo_dptr = NULL;
    cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

    if (iteration < ui_iterations) {

        iteration++;

        // execute the kernel
        int frame = 0;
        //pathtrace(pbo_dptr, frame, iteration);
        pathtrace(frame, iteration, ui_denoise, (int)(log2(ui_filterSize / 2) + 1.f), ui_colorWeight, ui_positionWeight, ui_normalWeight);
        double time = timer().getGpuElapsedTimeForPreviousOperation();
        totalTime += time;

#if AI_DENOISE
        //denoise
        CPUdenoise();
        //update pbo_dptr
        sendToGPU(pbo_dptr,iteration);
#endif

    }

    if (ui_showGbuffer) {
        showGBuffer(pbo_dptr);
    }
    else {
        showImage(pbo_dptr, iteration);
    }

    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);

    if (ui_saveAndExit) {
        std::cout << "time: " << totalTime << std::endl;
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_SPACE:
        camchanged = true;
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        cam.lookAt = ogLookAt;
        break;
      }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (ImGui::GetIO().WantCaptureMouse) return;
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
