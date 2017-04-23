//[header]
// A simple program to demonstrate how to ray-trace a polygon mesh
//[/header]
//[compile]
// Download the raytracepolygonmesh.cpp, geometry.h and cow.geo file to a folder.
// Open a shell/terminal, and run the following command where the files is saved:
//
// c++ -o raytracepolymesh raytracepolymesh.cpp -std=c++11 -O3
//
// Run with: ./raytracepolygonmesh. Open the file ./out.0000.png in Photoshop or any program
// reading PPM files.
//[/compile]
//[ignore]
// Copyright (C) 2012  www.scratchapixel.com
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//[/ignore]


#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>
#include <utility>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <chrono>

#include <omp.h>

#include "geometry.h"

static const float kInfinity = std::numeric_limits<float>::max();
static const float kEpsilon = 1e-8;
static const Vec3f kDefaultBackgroundColor = Vec3f(0.235294, 0.67451, 0.843137);

inline
float clamp(const float &lo, const float &hi, const float &v)
{ return std::max(lo, std::min(hi, v)); }

inline
float deg2rad(const float &deg)
{ return deg * M_PI / 180; }

struct Options
{
    uint32_t width = 1024;
    uint32_t height = 768;
    float fov = 90;
    Vec3f backgroundColor = kDefaultBackgroundColor;
    Matrix44f cameraToWorld;
};

class Object
{
 public:
    Object() {}
    virtual ~Object() {}
    virtual bool intersect(const Vec3f &, const Vec3f &, float &, uint32_t &, Vec2f &) const = 0;
    virtual void getSurfaceProperties(const Vec3f &, const Vec3f &, const uint32_t &, const Vec2f &, Vec3f &, Vec2f &) const = 0;
};

bool rayTriangleIntersect(
    const Vec3f &orig, const Vec3f &dir,
    const Vec3f &v0, const Vec3f &v1, const Vec3f &v2,
    float &t, float &u, float &v)
{
    Vec3f v0v1 = v1 - v0;
    Vec3f v0v2 = v2 - v0;
    Vec3f pvec = dir.crossProduct(v0v2);
    float det = v0v1.dotProduct(pvec);

    // ray and triangle are parallel if det is close to 0
    if (fabs(det) < kEpsilon) return false;

    float invDet = 1 / det;

    Vec3f tvec = orig - v0;
    u = tvec.dotProduct(pvec) * invDet;
    if (u < 0 || u > 1) return false;

    Vec3f qvec = tvec.crossProduct(v0v1);
    v = dir.dotProduct(qvec) * invDet;
    if (v < 0 || u + v > 1) return false;
    
    t = v0v2.dotProduct(qvec) * invDet;
    
    return true;
}

class TriangleMesh : public Object
{
public:
    // Build a triangle mesh from a face index array and a vertex index array
    TriangleMesh(
        const uint32_t nfaces,
        const std::unique_ptr<uint32_t []> &faceIndex,
        const std::unique_ptr<uint32_t []> &vertsIndex,
        const std::unique_ptr<Vec3f []> &verts,
        std::unique_ptr<Vec3f []> &normals,
        std::unique_ptr<Vec2f []> &st) :
        numTris(0)
    {
        uint32_t k = 0, maxVertIndex = 0;
        // find out how many triangles we need to create for this mesh
        for (uint32_t i = 0; i < nfaces; ++i) {
            numTris += faceIndex[i] - 2;
            for (uint32_t j = 0; j < faceIndex[i]; ++j)
                if (vertsIndex[k + j] > maxVertIndex)
                    maxVertIndex = vertsIndex[k + j];
            k += faceIndex[i];
        }
        maxVertIndex += 1;
        
        // allocate memory to store the position of the mesh vertices
        P = std::unique_ptr<Vec3f []>(new Vec3f[maxVertIndex]);
        for (uint32_t i = 0; i < maxVertIndex; ++i) {
            P[i] = verts[i];
        }
        
        // allocate memory to store triangle indices
        trisIndex = std::unique_ptr<uint32_t []>(new uint32_t [numTris * 3]);
        uint32_t l = 0;
        // [comment]
        // Generate the triangle index array
        // Keep in mind that there is generally 1 vertex attribute for each vertex of each face.
        // So for example if you have 2 quads, you only have 6 vertices but you have 2 * 4
        // vertex attributes (that is 8 normals, 8 texture coordinates, etc.). So the easiest
        // lazziest method in our triangle mesh, is to create a new array for each supported
        // vertex attribute (st, normals, etc.) whose size is equal to the number of triangles
        // multiplied by 3, and then set the value of the vertex attribute at each vertex
        // of each triangle using the input array (normals[], st[], etc.)
        // [/comment]
        N = std::unique_ptr<Vec3f []>(new Vec3f[numTris * 3]);
        texCoordinates = std::unique_ptr<Vec2f []>(new Vec2f[numTris * 3]);
        for (uint32_t i = 0, k = 0; i < nfaces; ++i) { // for each  face
            for (uint32_t j = 0; j < faceIndex[i] - 2; ++j) { // for each triangle in the face
                trisIndex[l] = vertsIndex[k];
                trisIndex[l + 1] = vertsIndex[k + j + 1];
                trisIndex[l + 2] = vertsIndex[k + j + 2];
	/*
                N[l] = normals[i];
                N[l + 1] = normals[k + j + 1];
                N[l + 2] = normals[k + j + 2];
                texCoordinates[l] = st[k];
                texCoordinates[l + 1] = st[k + j + 1];
                texCoordinates[l + 2] = st[k + j + 2];
	*/
                l += 3;
            }                                                                                                                                                                                                                                
            k += faceIndex[i];
        }
        // you can use move if the input geometry is already triangulated
        N = std::move(normals); // transfer ownership
        texCoordinates = std::move(st); // transfer ownership
    }
    // Test if the ray interesests this triangle mesh
    bool intersect(const Vec3f &orig, const Vec3f &dir, float &tNear, uint32_t &triIndex, Vec2f &uv) const
    {
        uint32_t j = 0;
        bool isect = false;
        for (uint32_t i = 0; i < numTris; ++i) {
            const Vec3f &v0 = P[trisIndex[j]];
            const Vec3f &v1 = P[trisIndex[j + 1]];
            const Vec3f &v2 = P[trisIndex[j + 2]];
            float t = kInfinity, u, v;
            if (rayTriangleIntersect(orig, dir, v0, v1, v2, t, u, v) && t < tNear) {
              tNear = t;
              uv.x = u;
              uv.y = v;
              triIndex = i;
              isect = true;
            }                                                                                                                                                                                                                                
            j += 3;
        }

        return isect;
    }
    void getSurfaceProperties(
        const Vec3f &hitPoint,
        const Vec3f &viewDirection,
        const uint32_t &triIndex,
        const Vec2f &uv,
        Vec3f &hitNormal,
        Vec2f &hitTextureCoordinates) const
    {
        // face normal
        const Vec3f &v0 = P[trisIndex[triIndex * 3]];
        const Vec3f &v1 = P[trisIndex[triIndex * 3 + 1]];
        const Vec3f &v2 = P[trisIndex[triIndex * 3 + 2]];
        hitNormal = (v1 - v0).crossProduct(v2 - v0);
        hitNormal.normalize();
        
        // texture coordinates
        const Vec2f &st0 = texCoordinates[trisIndex[triIndex * 3]];
        const Vec2f &st1 = texCoordinates[trisIndex[triIndex * 3 + 1]];
        const Vec2f &st2 = texCoordinates[trisIndex[triIndex * 3 + 2]];
        hitTextureCoordinates = (1 - uv.x - uv.y) * st0 + uv.x * st1 + uv.y * st2;

        // vertex normal
        /*
        const Vec3f &n0 = N[triIndex * 3];
        const Vec3f &n1 = N[triIndex * 3 + 1];
        const Vec3f &n2 = N[triIndex * 3 + 2];
        hitNormal = (1 - uv.x - uv.y) * n0 + uv.x * n1 + uv.y * n2;
        */
    }
    // member variables
    uint32_t numTris;                         // number of triangles
    std::unique_ptr<Vec3f []> P;              // triangles vertex position
    std::unique_ptr<uint32_t []> trisIndex;   // vertex index array
    std::unique_ptr<Vec3f []> N;              // triangles vertex normals
    std::unique_ptr<Vec2f []> texCoordinates; // triangles texture coordinates
};

TriangleMesh* generatePolyShphere(float rad, uint32_t divs)
{
    // generate points                                                                                                                                                                                      
    uint32_t numVertices = (divs - 1) * divs + 2;                                                                                                                              
    std::unique_ptr<Vec3f []> P(new Vec3f[numVertices]);
    std::unique_ptr<Vec3f []> N(new Vec3f[numVertices]);
    std::unique_ptr<Vec2f []> st(new Vec2f[numVertices]);
                                                                                                                                                                                                                                             
    float u = -M_PI_2;                                                                                                                                                          
    float v = -M_PI;                                                                                                                                                                                           
    float du = M_PI / divs;                                                                                                                                                                                    
    float dv = 2 * M_PI / divs;                                                                                                                                                 
                                                                                                                                                                                                                                             
    P[0] = N[0] = Vec3f(0, -rad, 0);
    uint32_t k = 1;                                                                                                                                                                                           
    for (uint32_t i = 0; i < divs - 1; i++) {                                                                                            
        u += du;                                                                                                                                                                                                                             
        v = -M_PI;                                                                                                                                                                                                                           
        for (uint32_t j = 0; j < divs; j++) {                                                                                                                           
            float x = rad * cos(u) * cos(v);                                                                                                                                                                   
            float y = rad * sin(u);                                                                                                                                                                            
            float z = rad * cos(u) * sin(v) ;                                                                                                                                                                  
            P[k] = N[k] = Vec3f(x, y, z);
            st[k].x = u / M_PI + 0.5;
            st[k].y = v * 0.5 / M_PI + 0.5;
            v += dv, k++;                                                                                                                                                                                                                    
        }                                                                                                                                                                                                                                    
    }                                                                                                                                                                                                                                        
    P[k] = N[k] = Vec3f(0, rad, 0);
    
    uint32_t npolys = divs * divs;                                                                                                                                                                                                           
    std::unique_ptr<uint32_t []> faceIndex(new uint32_t[npolys]);
    std::unique_ptr<uint32_t []> vertsIndex(new uint32_t[(6 + (divs - 1) * 4) * divs]);
                                                                                                                                                                                                                                             
    // create the connectivity lists                                                                                                                                                                        
    uint32_t vid = 1, numV = 0, l = 0;                                                                                                          
    k = 0;                                                                                                                                                                                                    
    for (uint32_t i = 0; i < divs; i++) {                                                                                                                               
        for (uint32_t j = 0; j < divs; j++) {                                                                                                                           
            if (i == 0) {                                                                                                                                                  
                faceIndex[k++] = 3;
                vertsIndex[l] = 0;
                vertsIndex[l + 1] = j + vid;
                vertsIndex[l + 2] = (j == (divs - 1)) ? vid : j + vid + 1;
                l += 3;                                                                                                                                                                                       
            }                                                                                                                                                                                                                                
            else if (i == (divs - 1)) {                                                                                                 
                faceIndex[k++] = 3;
                vertsIndex[l] = j + vid + 1 - divs;
                vertsIndex[l + 1] = vid + 1;
                vertsIndex[l + 2] = (j == (divs - 1)) ? vid + 1 - divs : j + vid + 2 - divs;
                l += 3;
            }
            else {
                faceIndex[k++] = 4;
                vertsIndex[l] = j + vid + 1 - divs;
                vertsIndex[l + 1] = j + vid + 1;
                vertsIndex[l + 2] = (j == (divs - 1)) ? vid + 1 : j + vid + 2;
                vertsIndex[l + 3] = (j == (divs - 1)) ? vid + 1 - divs : j + vid + 2 - divs;
                l += 4;
            }
            numV++;
        }
        vid = numV;
    }
    
    return new TriangleMesh(npolys, faceIndex, vertsIndex, P, N, st);
}

#include "dragon_edges.h"
#include "dragon_normals_xyz.h"
#include "dragon_st.h"
#include "dragon_vertices_xyz_real.h"

TriangleMesh* loadDragon(void)
{
        // read from header 
	    uint32_t numFaces = dragonNumFaces; //37986

	printf("numFaces = %d\n", numFaces);

        std::unique_ptr<uint32_t []> faceIndex(new uint32_t[numFaces]);
        uint32_t vertsIndexArraySize = 0;

        // reading face index array
        for (uint32_t i = 0; i < numFaces; ++i) {
            faceIndex[i] = 3; // hardcode to 3 for triangles? #yolo
            vertsIndexArraySize += faceIndex[i]; // 113958 at end
        }

	printf("vertsIndexArraySize = %d\n", vertsIndexArraySize);

        std::unique_ptr<uint32_t []> vertsIndex(new uint32_t[vertsIndexArraySize]);
        uint32_t vertsArraySize = 0;
        // reading vertex index array
        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
            vertsIndex[i] = dragonVertsIndexes[i];
            if (vertsIndex[i] > vertsArraySize) vertsArraySize = vertsIndex[i];
        }
        vertsArraySize += 1;

	printf("vertsArraySize = %d\n", vertsArraySize);

        // reading vertices
        std::unique_ptr<Vec3f []> verts(new Vec3f[vertsArraySize]);
        for (uint32_t i = 0; i < vertsArraySize; ++i) {
            verts[i].x = dragonVerts[i][0];
            verts[i].y = dragonVerts[i][1];
            verts[i].z = dragonVerts[i][2];
        }
        // reading normals
        std::unique_ptr<Vec3f []> normals(new Vec3f[vertsArraySize]);
        for (uint32_t i = 0; i < vertsArraySize; ++i) {
            normals[i].x = dragonNormals[i][0];
            normals[i].y = dragonNormals[i][1];
            normals[i].z = dragonNormals[i][2];
        }
        // reading st coordinates
        std::unique_ptr<Vec2f []> st(new Vec2f[vertsArraySize]);
        for (uint32_t i = 0; i < vertsArraySize; ++i) {
            st[i].x = dragonST[i][0];
            st[i].y = dragonST[i][1];
        }
        
        return new TriangleMesh(numFaces, faceIndex, vertsIndex, verts, normals, st);
    
}

#include "uh60_vertices_xyz.h"
#include "uh60_vertices_normals.h"
#include "uh60_vertices_st.h"
#include "uh60_edges.h"

TriangleMesh* loaduh60(void)
{
        // read from header 
	uint32_t numFaces = uh60NumFaces; //37986

	printf("numFaces = %d\n", numFaces);

        std::unique_ptr<uint32_t []> faceIndex(new uint32_t[numFaces]);
        uint32_t vertsIndexArraySize = 0;

        // reading face index array
        for (uint32_t i = 0; i < numFaces; ++i) {
            faceIndex[i] = 3; // hardcode to 3 for triangles? #yolo
            vertsIndexArraySize += faceIndex[i]; // 113958 at end
        }

	printf("vertsIndexArraySize = %d\n", vertsIndexArraySize);

        std::unique_ptr<uint32_t []> vertsIndex(new uint32_t[vertsIndexArraySize]);
        uint32_t vertsArraySize = 0;
        // reading vertex index array
        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
            vertsIndex[i] = uh60VertsIndexes[i];
            if (vertsIndex[i] > vertsArraySize) vertsArraySize = vertsIndex[i];
        }
        vertsArraySize += 1;

	printf("vertsArraySize = %d\n", vertsArraySize);

        // reading vertices
        std::unique_ptr<Vec3f []> verts(new Vec3f[vertsArraySize]);
        for (uint32_t i = 0; i < vertsArraySize; ++i) {
            verts[i].x = uh60Verts[i][0];
            verts[i].y = uh60Verts[i][1];
            verts[i].z = uh60Verts[i][2];
        }
        // reading normals
        std::unique_ptr<Vec3f []> normals(new Vec3f[vertsArraySize]);
        for (uint32_t i = 0; i < vertsArraySize; ++i) {
            normals[i].x = uh60Normals[i][0];
            normals[i].y = uh60Normals[i][1];
            normals[i].z = uh60Normals[i][2];
        }
        // reading st coordinates
        std::unique_ptr<Vec2f []> st(new Vec2f[vertsArraySize]);
        for (uint32_t i = 0; i < vertsArraySize; ++i) {
            st[i].x = uh60ST[i][0];
            st[i].y = uh60ST[i][1];
        }
        
        return new TriangleMesh(numFaces, faceIndex, vertsIndex, verts, normals, st);
    
}

TriangleMesh* loadPolyMeshFromFile(const char *file)
{
    std::ifstream ifs;
    try {
        ifs.open(file);
        if (ifs.fail()) throw;
        std::stringstream ss;
        ss << ifs.rdbuf();
        uint32_t numFaces;
        ss >> numFaces;
        std::unique_ptr<uint32_t []> faceIndex(new uint32_t[numFaces]);
        uint32_t vertsIndexArraySize = 0;
        // reading face index array
        for (uint32_t i = 0; i < numFaces; ++i) {
            ss >> faceIndex[i];
            vertsIndexArraySize += faceIndex[i];
        }
        std::unique_ptr<uint32_t []> vertsIndex(new uint32_t[vertsIndexArraySize]);
        uint32_t vertsArraySize = 0;
        // reading vertex index array
        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
            ss >> vertsIndex[i];
            if (vertsIndex[i] > vertsArraySize) vertsArraySize = vertsIndex[i];
        }
        vertsArraySize += 1;
        // reading vertices
        std::unique_ptr<Vec3f []> verts(new Vec3f[vertsArraySize]);
        for (uint32_t i = 0; i < vertsArraySize; ++i) {
            ss >> verts[i].x >> verts[i].y >> verts[i].z;
        }
        // reading normals
        std::unique_ptr<Vec3f []> normals(new Vec3f[vertsIndexArraySize]);
        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
            ss >> normals[i].x >> normals[i].y >> normals[i].z;
        }
        // reading st coordinates
        std::unique_ptr<Vec2f []> st(new Vec2f[vertsIndexArraySize]);
        for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
            ss >> st[i].x >> st[i].y;
        }
        
        return new TriangleMesh(numFaces, faceIndex, vertsIndex, verts, normals, st);
    }
    catch (...) {
        ifs.close();
    }
    ifs.close();
    
    return nullptr;
}

bool trace(
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    float &tNear, uint32_t &index, Vec2f &uv, Object **hitObject)
{
    *hitObject = nullptr;
    for (uint32_t k = 0; k < objects.size(); ++k) {
        float tNearTriangle = kInfinity;
        uint32_t indexTriangle;
        Vec2f uvTriangle;
        if (objects[k]->intersect(orig, dir, tNearTriangle, indexTriangle, uvTriangle) && tNearTriangle < tNear) {
            *hitObject = objects[k].get();
            tNear = tNearTriangle;
            index = indexTriangle;
            uv = uvTriangle;
        }
    }

    return (*hitObject != nullptr);
}

Vec3f castRay(
    const Vec3f &orig, const Vec3f &dir,
    const std::vector<std::unique_ptr<Object>> &objects,
    const Options &options)
{
    Vec3f hitColor = options.backgroundColor;
    float tnear = kInfinity;
    Vec2f uv;
    uint32_t index = 0;
    Object *hitObject = nullptr;
    if (trace(orig, dir, objects, tnear, index, uv, &hitObject)) {
        Vec3f hitPoint = orig + dir * tnear;
        Vec3f hitNormal;
        Vec2f hitTexCoordinates;
        hitObject->getSurfaceProperties(hitPoint, dir, index, uv, hitNormal, hitTexCoordinates);
        float NdotView = std::max(0.f, hitNormal.dotProduct(-dir));
        const int M = 10;
        float checker = (fmod(hitTexCoordinates.x * M, 1.0) > 0.5) ^ (fmod(hitTexCoordinates.y * M, 1.0) < 0.5);
        float c = 0.3 * (1 - checker) + 0.7 * checker;
        
        hitColor = c * NdotView; //Vec3f(uv.x, uv.y, 0);
    }

    return hitColor;
}

// [comment]
// The main render function. This where we iterate over all pixels in the image, generate
// primary rays and cast these rays into the scene. The content of the framebuffer is
// saved to a file.
// [/comment]
void render(
    const Options &options,
    const std::vector<std::unique_ptr<Object>> &objects,
    const uint32_t &frame)
{
    std::unique_ptr<Vec3f []> framebuffer(new Vec3f[options.width * options.height]);
    Vec3f *pix = framebuffer.get();
    float scale = tan(deg2rad(options.fov * 0.5));
    float imageAspectRatio = options.width / (float)options.height;
    Vec3f orig;
    options.cameraToWorld.multVecMatrix(Vec3f(0), orig);
    auto timeStart = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (uint32_t j = 0; j < options.height; ++j) {
        for (uint32_t i = 0; i < options.width; ++i) {
            //printf("render current pixel height x width = %d,%d\n", j, i);
            // generate primary ray direction
            float x = (2 * (i + 0.5) / (float)options.width - 1) * imageAspectRatio * scale;
            float y = (1 - 2 * (j + 0.5) / (float)options.height) * scale;
            Vec3f dir;
            options.cameraToWorld.multDirMatrix(Vec3f(x, y, -1), dir);
            dir.normalize();
            pix[j*options.width + i] = castRay(orig, dir, objects, options);
            //*(pix++) = castRay(orig, dir, objects, options);
        }
        fprintf(stderr, "\r%3d%c", uint32_t(j / (float)options.height * 100), '%');
    }
    auto timeEnd = std::chrono::high_resolution_clock::now();
    auto passedTime = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
    fprintf(stderr, "\rDone: %.2f (sec)\n", passedTime / 1000);
    
    // save framebuffer to file
    char buff[256];
    sprintf(buff, "out.%04d.ppm", frame);
    std::ofstream ofs;
    ofs.open(buff);
    ofs << "P6\n" << options.width << " " << options.height << "\n255\n";
    for (uint32_t i = 0; i < options.height * options.width; ++i) {
        char r = (char)(255 * clamp(0, 1, framebuffer[i].x));
        char g = (char)(255 * clamp(0, 1, framebuffer[i].y));
        char b = (char)(255 * clamp(0, 1, framebuffer[i].z));
        ofs << r << g << b;
    }
    ofs.close();
    
}

// [comment]
// In the main function of the program, we create the scene (create objects and lights)
// as well as set the options for the render (image widht and height, maximum recursion
// depth, field-of-view, etc.). We then call the render function().
// [/comment]
int main(int argc, char **argv)
{
    // setting up options
    Options options;
    //options.cameraToWorld[3][2] = 10;

    /*
a =  [[ 0.707107  -0.331295   0.624695   0] 
   [ 0          0.883452   0.468521   0] 
   [-0.707107  -0.331295   0.624695   0]
   [-1.63871   -5.747777 -40.400412   1]]


rotation_orig =  [[ 0.707107  -0.331295   0.624695] 
   [ 0          0.883452   0.468521   ] 
   [-0.707107  -0.331295   0.624695   ]]

    */

    Matrix44f tmp = Matrix44f(0.707107, -0.331295, 0.624695, 0, 0, 0.883452, 0.468521, 0, -0.707107, -0.331295, 0.624695, 0, -1.63871, -5.747777-10, -40.400412-20, 1);
    Matrix44f dragon_cam = Matrix44f(0.92388, 0.19134, 0.33141, 0,
                                     0.38268, -0.46194, -0.80010, 0, 
                                     0, 0.86603, -0.50000, 0, 
                                   -15, -15, -100, 1);
    Matrix44f uh60_cam = Matrix44f(-0.5, 0, 0.86603, 0,
                                   0.86603, 0, 0.5, 0, 
                                    0, 1, 0, 0, 
                                   0, 0, -20, 1);
    options.cameraToWorld = dragon_cam.inverse();
    //options.cameraToWorld = uh60_cam.inverse();
    //options.cameraToWorld = tmp.inverse();
    options.fov = 50.0393;
#if 1
    std::vector<std::unique_ptr<Object>> objects;
    TriangleMesh *mesh = loadDragon();
    //TriangleMesh *mesh = loaduh60();
    //TriangleMesh *mesh = loadPolyMeshFromFile("./cow.geo");
    if (mesh != nullptr) objects.push_back(std::unique_ptr<Object>(mesh));
    
    // finally, render
    render(options, objects, 0);
#else
    for (uint32_t i = 0; i < 10; ++i) {
        int divs = 5 + i;
        // creating the scene (adding objects and lights)
        std::vector<std::unique_ptr<Object>> objects;
        TriangleMesh *mesh = generatePolyShphere(2, divs);
        objects.push_back(std::unique_ptr<Object>(mesh));
    
        auto timeStart = std::chrono::high_resolution_clock::now();
        // finally, render
        render(options, objects, i);
        auto timeEnd = std::chrono::high_resolution_clock::now();
        auto passedTime = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
        std::cerr << mesh->numTris << " " << passedTime << std::endl;
    }
#endif

    return 0;
}
