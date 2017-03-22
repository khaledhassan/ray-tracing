#ifndef GPU_RT_STRUCTS
#define GPU_RT_STRUCTS
#include <stdint.h>

struct gpuRay
{
	float orig[3];
	float dir[3];
	uint32_t id;
	uint32_t pad;
};
struct gpuTri
{
	float v1[3];
	float v2[3];
	float v3[3];
	uint32_t id;
	uint32_t pad[2];
};
struct gpuIsect
{
	uint16_t dataValid[2];
	float t;
	uint32_t triId;
	uint32_t rayId;
};


#endif
