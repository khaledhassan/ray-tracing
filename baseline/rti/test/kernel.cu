#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <cassert>
#include <string>
#include <fstream>
#include <time.h>
//#include <pthread.h>
#include <ctime>
#include "gpuRTIStruct.h"
//#include "geometry.h"
#include "cublas_v2.h"

#define NTRI_TOT 102400
#define NRAY_TOT 20480
#define RTRT_TOT NTRI_TOT*NRAY_TOT
#define EPSILON 0.00000000001
#define GPU_WORD_SIZE 8
#define T float
#define NTRI 4
#define NRAY 8

float tri[NTRI][9] =
{ { 1, 0, 0, 0, 1, 0, 0, 0, 0 },
{ -1, 0, 0, 0, 1, 0, 0, 0, 0 },
{ -1, 0, 0, 0, -1, 0, 0, 0, 0 },
{ 1, 0, 0, 0, -1, 0, 0, 0, 0 } };

float ray[NRAY][6] = {
	{ 2, 2, 1, 0, 0, 1 },
	{ 2, 2, 1, 0, 0, 1 },
	{ 2, 2, 1, 0, 0, 1 },
	{ 0.5, 0.5, 0.8, 0, 0, 1 },
	{ -0.5, -0.5, 0.3, 0, 0, 1 },
	{ 0.5, -0.5, 0.9, 0, 0, 1 },
	{ -0.5, 0.5, 0.5, 0, 0, 1 },
	{ 0.5, 0.5, 1, 0, 0, -1 }, };

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

using namespace std;


//template<typename T>
class Vec3
{
public:
	CUDA_CALLABLE_MEMBER Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
	CUDA_CALLABLE_MEMBER Vec3(T xx) : x(xx), y(xx), z(xx) {}
	CUDA_CALLABLE_MEMBER Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}
	CUDA_CALLABLE_MEMBER Vec3 operator + (const Vec3 &v) const
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}
	
	CUDA_CALLABLE_MEMBER Vec3 operator - (const Vec3 &v) const
	{
		return Vec3(x - v.x, y - v.y, z - v.z);
	}
	CUDA_CALLABLE_MEMBER Vec3 operator - () const
	{
		return Vec3(-x, -y, -z);
	}
	CUDA_CALLABLE_MEMBER Vec3 operator * (const T &r) const
	{
		return Vec3(x * r, y * r, z * r);
	}
	CUDA_CALLABLE_MEMBER Vec3 operator * (const Vec3 &v) const
	{
		return Vec3(x * v.x, y * v.y, z * v.z);
	}
	CUDA_CALLABLE_MEMBER T dotProduct(const Vec3 &v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}
	CUDA_CALLABLE_MEMBER Vec3& operator /= (const T &r)
	{
		x /= r, y /= r, z /= r; return *this;
	}
	CUDA_CALLABLE_MEMBER Vec3& operator *= (const T &r)
	{
		x *= r, y *= r, z *= r; return *this;
	}
	CUDA_CALLABLE_MEMBER Vec3 crossProduct(const Vec3 &v) const
	{
		return Vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}
	CUDA_CALLABLE_MEMBER T norm() const
	{
		return x * x + y * y + z * z;
	}
	CUDA_CALLABLE_MEMBER T length() const
	{
		return sqrt(norm());
	}
	//[comment]
	// The next two operators are sometimes called access operators or
	// accessors. The Vec coordinates can be accessed that way v[0], v[1], v[2],
	// rather than using the more traditional form v.x, v.y, v.z. This useful
	// when vectors are used in loops: the coordinates can be accessed with the
	// loop index (e.g. v[i]).
	//[/comment]
	CUDA_CALLABLE_MEMBER const T& operator [] (uint8_t i) const { return (&x)[i]; }
	CUDA_CALLABLE_MEMBER T& operator [] (uint8_t i) { return (&x)[i]; }
	CUDA_CALLABLE_MEMBER Vec3& normalize()
	{
		T n = norm();
		if (n > 0) {
			T factor = 1 / sqrt(n);
			x *= factor, y *= factor, z *= factor;
		}

		return *this;
	}

	CUDA_CALLABLE_MEMBER friend Vec3 operator * (const T &r, const Vec3 &v)
	{
		return Vec3(v.x * r, v.y * r, v.z * r);
	}
	CUDA_CALLABLE_MEMBER friend Vec3 operator / (const T &r, const Vec3 &v)
	{
		return Vec3(r / v.x, r / v.y, r / v.z);
	}

	CUDA_CALLABLE_MEMBER friend std::ostream& operator << (std::ostream &s, const Vec3 &v)
	{
		return s << '[' << v.x << ' ' << v.y << ' ' << v.z << ']';
	}

	T x, y, z;
};

cudaError_t checkCuda(cudaError_t result)
{
	if (result != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
	}
	return result;
}

int verifyResults(int * d_result, int* cpu_result, int size)
{
	int i, er = 1;
	for (i = 0; i < size; i++)
	{
		if (d_result[i] != cpu_result[i])
		{
			er = 0;
			break;
		}
	}
	return er;
}

#define Vec3f Vec3

CUDA_CALLABLE_MEMBER
bool Intersect(const Vec3f &orig, const Vec3f &dir, const Vec3f &p1, const Vec3f &p2, const Vec3f &p3, float &tHit, float &u, float &v)
{
	// Get triangle vertices in _p1_, _p2_, and _p3_
	Vec3f e1 = p2 - p1;
	Vec3f e2 = p3 - p1;
	Vec3f s1 = dir.crossProduct(e2);
	float divisor = s1.dotProduct(e1);

	if (abs(divisor) < EPSILON)
		return false;
	float invDivisor = 1.f / divisor;

	// Compute first barycentric coordinate
	Vec3f s = orig - p1;
	float b1 = s.dotProduct(s1) * invDivisor;
	if (b1 < 0. || b1 > 1.)
		return false;

	// Compute second barycentric coordinate
	Vec3f s2 = s.crossProduct(e1);
	float b2 = dir.dotProduct(s2) * invDivisor;
	if (b2 < 0. || b1 + b2 > 1.)
		return false;

	// Compute _t_ to intersection point
	float t = e2.dotProduct(s2) * invDivisor;
	//if (t < ray.mint || t > ray.maxt)
	//    return false;

	// Compute triangle partial derivatives


	tHit = t;
	return true;
}


__global__
void moellerT(const gpuRay *d_input1, const gpuTri *d_input2, gpuIsect* d_output)
{
	int id1 = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d\n", id1);
	float u; float v; float t;
	d_output[id1].t = 10000;
	d_output[id1].rayId = d_input1[id1].id;
	for (int i = 0; i<NTRI_TOT; i++)
	{
		Vec3f ro(d_input1[id1].orig[0], d_input1[id1].orig[1], d_input1[id1].orig[2]);
		Vec3f rd(d_input1[id1].dir[0], d_input1[id1].dir[1], d_input1[id1].dir[2]);
		Vec3f v1(d_input2[i].v1[0], d_input2[i].v1[1], d_input2[i].v1[2]);
		Vec3f v2(d_input2[i].v2[0], d_input2[i].v2[1], d_input2[i].v2[2]);
		Vec3f v3(d_input2[i].v3[0], d_input2[i].v3[1], d_input2[i].v3[2]);

		if (Intersect(ro, rd, v1, v2, v3, t, u, v))
		{
			
			if (t<d_output[id1].t)
			{
				d_output[id1].dataValid[0] = 1;
				d_output[id1].dataValid[1] = 1;
				d_output[id1].rayId = d_input1[id1].id;
				d_output[id1].triId = d_input2[i].id;
				d_output[id1].t = t;
			}
		}
	}

}

int main(int argc, char* argv[])
{
	// kernel launch
	int nrays = NRAY_TOT;
	int grid = nrays / 1024 + 1;
	int block = 1024; // number of rays
	unsigned long long t1, t2;
	srand(time(NULL));
	float t, u, v;
	clock_t start, end;
	double cpu_time_used;

	gpuRay *input1 = (gpuRay*)malloc(NRAY_TOT * sizeof(gpuRay));
	gpuTri *input2 = (gpuTri*)malloc(NTRI_TOT * sizeof(gpuTri));
	gpuIsect *output = (gpuIsect*)malloc(NRAY_TOT * sizeof(gpuIsect));
	gpuIsect *verify = (gpuIsect*)malloc(NRAY_TOT * sizeof(gpuIsect));

	memset(output, 0, sizeof(gpuIsect)*NRAY_TOT);
	memset(verify, 0, sizeof(gpuIsect)*NRAY_TOT);

	// GPU pointers
	gpuRay *d_input1;
	gpuTri *d_input2;
	gpuIsect *d_output;

	// GPU allocations
	checkCuda(cudaMalloc((void**)&d_input1, NRAY_TOT * sizeof(gpuRay)));
	checkCuda(cudaMalloc((void**)&d_input2, NTRI_TOT * sizeof(gpuTri)));
	checkCuda(cudaMalloc((void**)&d_output, NRAY_TOT * sizeof(gpuIsect)));
	checkCuda(cudaMemset((void*)d_input1, 0, NRAY_TOT * sizeof(gpuRay)));
	checkCuda(cudaMemset((void*)d_input2, 0, NTRI_TOT * sizeof(gpuTri)));
	checkCuda(cudaMemset((void*)d_output, 0, NRAY_TOT * sizeof(gpuIsect)));

	// random in[put on host

	for (int i = 0; i<NRAY_TOT; i++)
		output[i].t = 10000;
	for (int i = 0; i<NTRI_TOT; i++)
	{
		input2[i].v1[0] = rand() % 100 / 100.0;
		input2[i].v1[1] = rand() % 100 / 100.0;
		input2[i].v1[2] = rand() % 100 / 100.0;

		input2[i].v2[0] = rand() % 100 / 100.0;
		input2[i].v2[1] = rand() % 100 / 100.0;
		input2[i].v2[2] = rand() % 100 / 100.0;

		input2[i].v3[0] = rand() % 100 / 100.0;
		input2[i].v3[1] = rand() % 100 / 100.0;
		input2[i].v3[2] = rand() % 100 / 100.0;
		input2[i].id = i;
		input2[i].pad[0] = 0;
		input2[i].pad[1] = 0;
	}

	for (int i = 0; i<NRAY_TOT; i++)
	{
		input1[i].orig[0] = rand() % 100 / 100.0;
		input1[i].orig[1] = rand() % 100 / 100.0;
		input1[i].orig[2] = rand() % 100 / 100.0;
		input1[i].dir[0] = rand() % 100 / 100.0;
		input1[i].dir[1] = rand() % 100 / 100.0;
		input1[i].dir[2] = rand() % 100 / 100.0;
		input1[i].id = i;
		input1[i].pad = 0;
	}

	// memory transfer from host to GPU
	// GPU time calculations
	cudaEvent_t start_gpupci, stop_gpupci;
	cudaEventCreate(&start_gpupci);
	cudaEventCreate(&stop_gpupci);
	
	checkCuda(cudaMemcpy(d_input1, input1, NRAY_TOT * sizeof(gpuRay), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_input2, input2, NTRI_TOT * sizeof(gpuTri), cudaMemcpyHostToDevice));
	//checkCuda(cudaMemcpy(d_output, output, NRAY_TOT * sizeof(gpuIsect), cudaMemcpyHostToDevice));

	// GPU time calculations
	//cudaEvent_t start_gpu, stop_gpu;
	//cudaEventCreate(&start_gpu);
	//cudaEventCreate(&stop_gpu);

	
	cudaEventRecord(start_gpupci);

	//cudaEventRecord(start_gpu);
	moellerT <<<grid, block >>>(d_input1, d_input2, d_output);
	//cudaDeviceSynchronize();
	//cudaEventRecord(stop_gpu);
	//cudaEventSynchronize(stop_gpu);
	//float gpu_time_used = 0;
	//cudaEventElapsedTime(&gpu_time_used, start_gpu, stop_gpu);
	//printf("GPU time : %f\n", gpu_time_used);
	cudaEventRecord(stop_gpupci);
	cudaEventSynchronize(stop_gpupci);

	// memory transfer from GPU to host
	checkCuda(cudaMemcpy(output, d_output, NRAY_TOT * sizeof(gpuIsect), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	float gpu_time_used = 0;
	cudaEventElapsedTime(&gpu_time_used, start_gpupci, stop_gpupci);

	// verification OVM / UVM
	start = clock();
	for (int j = 0; j<NRAY_TOT; j++)
	{
		verify[j].t = 10000;
		for (int i = 0; i<NTRI_TOT; i++)
		{
			Vec3f ro(input1[j].orig[0], input1[j].orig[1], input1[j].orig[2]);
			Vec3f rd(input1[j].dir[0], input1[j].dir[1], input1[j].dir[2]);
			Vec3f v1(input2[i].v1[0], input2[i].v1[1], input2[i].v1[2]);
			Vec3f v2(input2[i].v2[0], input2[i].v2[1], input2[i].v2[2]);
			Vec3f v3(input2[i].v3[0], input2[i].v3[1], input2[i].v3[2]);
			if (Intersect(ro, rd, v1, v2, v3, t, u, v))
			{
				if (t<verify[j].t)
				{
					verify[j].dataValid[0] = 1;
					verify[j].dataValid[1] = 1;
					verify[j].rayId = j;
					verify[j].triId = i;
					verify[j].t = t;
				}
			}
		}
	}
	end = clock();
	cpu_time_used = (((float)(end - start)) / CLOCKS_PER_SEC) * 1000;

	printf("CPU time : %f\n", cpu_time_used);
	printf("GPU time with PCI : %f\n", gpu_time_used);
	printf("CPU RTIT/s : %f\n", RTRT_TOT/cpu_time_used * 1000);
	printf("GPU RTIT/s with PCI : %f\n", RTRT_TOT/gpu_time_used*1000);


	float Speedup = cpu_time_used / gpu_time_used;
	printf("speedup : %f \n", Speedup);
	int err = 0;
	for (int i = 0; i < NRAY_TOT; i++)
	{
		if (verify[i].dataValid[1] != output[i].dataValid[1])
		{
			printf("GPU and CPU results are not same\n");
			printf("CPU rayid : %d, triid : %d, t :  %f, val1 : %d, val2 : %d\n", verify[i].rayId, verify[i].triId, verify[i].t, verify[i].dataValid[0], verify[i].dataValid[1]);
			printf("GPU rayid : %d, triid : %d, t :  %f, val1 : %d, val2 : %d\n", output[i].rayId, output[i].triId, output[i].t, output[i].dataValid[0], output[i].dataValid[1]);
			err++;
		}
		//else
			//printf("GPU and CPU results are same\n");
	}
	printf("GPU and CPU results are not same in %d instances\n",err);
	
	checkCuda(cudaFree(d_input1));
	checkCuda(cudaFree(d_input2));
	checkCuda(cudaFree(d_output));
	// compare the result on CPU
	system("pause");
	return 0;
}

