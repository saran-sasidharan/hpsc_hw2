#Importing required modules
import pyopencl as cl
import numpy as np
from time import time

N = 1024 # Size of matrix
block_size = 16

# Creating context and command queue for opencl calls
ctx = cl.create_some_context()
que = cl.CommandQueue(ctx)

a = np.random.rand(N,N).astype(np.float32)
b = np.random.rand(N,N).astype(np.float32)
c = np.empty((N,N)).astype(np.float32)

kernel_parameters = {"block_size": block_size, "N":N}

code = """

#define block_size %(block_size)d
#define N %(N)d

__kernel __attribute__((reqd_work_group_size(block_size,block_size,1))) void mat_multiply(__global float* a, __global float* b, __global float* c)
{
	__local float al[block_size*block_size];
	__local float bl[block_size*block_size];
	
	int gid_x = get_group_id(0);
	int gid_y = get_group_id(1);
	int lid_x = get_local_id(0);
	int lid_y = get_local_id(1);
	
	float c_temp = 0;
	
	for (int i = N*block_size*gid_y, j = block_size*gid_x; i <= N*block_size*gid_y + N-1; i += block_size, j += N*block_size) {

		al[lid_y*block_size+lid_x] = a[i+N*lid_y+lid_x];
		bl[lid_y*block_size+lid_x] = b[j+N*lid_y+lid_x];
		
		barrier(CLK_LOCAL_MEM_FENCE);
	
		
