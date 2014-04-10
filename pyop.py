
import pyopencl as cl
import numpy as np

ctx = cl.create_some_context()
que = cl.CommandQueue(ctx)

N = 4

a = np.array(np.ones((N,N)),dtype = np.float32)
b = np.array(np.random.rand(N,N),dtype = np.float32)
c = np.empty(a.shape , dtype = np.float32);

a_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = a)
b_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = b)
c_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY , a.nbytes)
#N_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf = N)

code = """
__kernel void test_add(__global float* a, __global float* b, __global float* c) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    const int N = 4;
    c[j*N+i] = a[j*N+i] + b[j*N+i];
}
"""

prg = cl.Program(ctx, code).build()

loch = prg.test_add(que,a.shape,None,a_buf,b_buf,c_buf)
loch.wait()

cl.enqueue_copy(que,c,c_buf)

print(c)
print(b)
