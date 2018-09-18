#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <ctype.h>
//#include "ocl-kernel-gen.h"
#include "parse-args.h"
#include "sgtype.h"
#include "sgbuf.h"
#include "mytime.h"

#if defined( USE_OPENCL )
	#include "../opencl/ocl-backend.h"
#endif
#if defined( USE_OPENMP )
	#include <omp.h>
	#include "../openmp/omp-backend.h"
	#include "../openmp/openmp_kernels.h"
#endif
#if defined ( USE_CUDA )
    #include <cuda.h>
    #include "../cuda/cuda-backend.h"
#endif

#define ALIGNMENT (64)

//SGBench specific enums
enum sg_backend backend = INVALID_BACKEND;
enum sg_kernel  kernel  = INVALID_KERNEL;
enum sg_op      op      = OP_COPY;

//Strings defining program behavior
char platform_string[STRING_SIZE];
char device_string[STRING_SIZE];
char kernel_file[STRING_SIZE];
char kernel_name[STRING_SIZE];

size_t source_len;
size_t target_len;
size_t index_len;
size_t block_len;
size_t seed;
size_t R = 10;
size_t N = 100;
size_t workers = 1;
size_t vector_len = 1;

int json_flag = 0, validate_flag = 0, print_header_flag = 1;

void print_header(){
    if (backend == OPENCL) printf("backend platform device kernel op time source_size target_size idx_size worksets bytes_moved usable_bandwidth loops runs workers\n");
    if (backend == OPENMP) printf("backend kernel op time source_size target_size idx_size worksets bytes_moved usable_bandwidth loops runs workers\n");
}

void make_upper (char* s) {
    while (*s) {
        *s = toupper(*s);
        s++;
    }
}

long posmod (long i, long n) {
    return (i % n + n) % n;
}

void *sg_safe_cpu_alloc (size_t size) {
    void *ptr = aligned_alloc (ALIGNMENT, size);
    if (!ptr) {
        printf("Falied to allocate memory on cpu\n");
        exit(1);
    }
    return ptr;
}

/** Time reported in seconds, sizes reported in bytes, bandwidth reported in mib/s"
 */
void report_time(double time, size_t source_size, size_t target_size, size_t index_len, size_t worksets){
    if(backend == OPENMP) printf("OPENMP ");
    if(backend == OPENCL) printf("OPENCL ");

    if(backend == OPENCL){
        make_upper(platform_string);
        make_upper(device_string);
        printf("%s %s ", platform_string, device_string);
    }

    if(kernel == SCATTER) printf("SCATTER ");
    if(kernel == GATHER) printf("GATHER ");
    if(kernel == SG) printf("SG ");

    if(op == OP_COPY) printf("COPY ");
    if(op == OP_ACCUM) printf("ACCUM ");

    printf("%lf %zu %zu %zu ", time, source_size, target_size, index_len);
    printf("%zu ", worksets);

    size_t bytes_moved = 2 * index_len * sizeof(SGTYPE_C);
    double usable_bandwidth = bytes_moved / time / 1024. / 1024.;
    printf("%zu %lf ", bytes_moved, usable_bandwidth);
    printf("%zu %zu %zu", N, R, workers);

    printf("\n");

}

void print_data(double *buf, size_t len){
    for (size_t i = 0; i < len; i++){
        printf("%.0lf ", buf[i]);
    }
    printf("\n");
}
void print_sizet(size_t *buf, size_t len){
    for (size_t i = 0; i < len; i++){
        printf("%zu ", buf[i]);
    }
    printf("\n");
}

int main(int argc, char **argv)
{

    sgDataBuf  source;
    sgDataBuf  target;
    sgIndexBuf si; //source_index
    sgIndexBuf ti; //target_index

    size_t cpu_cache_size = 30720 * 1000; 
    size_t cpu_flush_size = cpu_cache_size * 8;
    #ifdef USE_OPENCL
	cl_ulong device_cache_size = 0;
    	cl_uint work_dim = 1;
    #endif
    size_t device_flush_size = 0;
    size_t worksets = 1;
    size_t global_work_size = 1;
    size_t local_work_size = 1;
    size_t current_ws;
    long os, ot, oi;
    
    char *kernel_string;

    #ifdef USE_CUDA
        printf("Using cuda\n");
#define dim (3)
        unsigned int grid[dim]  = {2,2,2};
        unsigned int block[dim] = {2,1,1};
        my_kernel_wrapper(dim , grid, block);
        exit(0);
    #endif

    /* Parse command line arguments */
    parse_args(argc, argv);


    /* =======================================
	Initalization
       =======================================
    */

    /* Create a context and corresponding queue */
    #ifdef USE_OPENCL
    	initialize_dev_ocl(platform_string, device_string);
    #endif

    source.len = source_len;
    target.len = target_len;
    si.len     = index_len;
    ti.len     = index_len;

    /* TBD - cache flushing
    // Determine how many worksets we will need to flush the cache
    if (backend == OPENMP) {
        worksets = cpu_flush_size / 
            ((source.len + target.len) * sizeof(SGTYPE_C) 
            + (si.len + ti.len) * sizeof(cl_ulong)) + 1;
    }
    else if (backend == OPENCL) {
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, 
                sizeof(device_cache_size), &device_cache_size, NULL); 
        device_flush_size = device_cache_size * 8;
        worksets = device_flush_size / 
            ((source.len + target.len) * sizeof(SGTYPE_C) 
            + (si.len + ti.len) * sizeof(cl_ulong)) + 1;
    }*/

    /* These are the total size of the data allocated for each buffer */
    source.size = worksets * source.len * sizeof(SGTYPE_C);
    target.size = worksets * target.len * sizeof(SGTYPE_C);
    si.size     = worksets * si.len * sizeof(cl_ulong);
    ti.size     = worksets * ti.len * sizeof(cl_ulong);
    
    /* This is the number of SGTYPEs in a workset */
    //TODO: remove since this is obviously useless
    source.block_len = source.len;
    target.block_len = target.len;

    /* Create the kernel */
    #ifdef USE_OPENCL
        //kernel_string = ocl_kernel_gen(index_len, vector_len, kernel);
        kernel_string = read_file(kernel_file);
        sgp = kernel_from_string(context, kernel_string, kernel_name, NULL);
        if (kernel_string) {
            free(kernel_string);
        }
    #endif

    /* Create buffers on host */
    source.host_ptr = (SGTYPE_C*) sg_safe_cpu_alloc(source.size); 
    target.host_ptr = (SGTYPE_C*) sg_safe_cpu_alloc(target.size); 
    si.host_ptr = (cl_ulong*) sg_safe_cpu_alloc(si.size); 
    ti.host_ptr = (cl_ulong*) sg_safe_cpu_alloc(ti.size); 

    /* Populate buffers on host */
    random_data(source.host_ptr, source.len * worksets);
    linear_indices(si.host_ptr, si.len, worksets, source.len / si.len);
    linear_indices(ti.host_ptr, ti.len, worksets, target.len / ti.len);

    /* Create buffers on device and transfer data from host */
    #ifdef USE_OPENCL
	create_dev_buffers_ocl(&source, &target, &si, &ti, block_len);
    #elif defined USE_CUDA
    create_dev_buffers_cuda(&source, &target, &si, &ti, block_len);
    printf("ran this\n");
    #endif

    
    /* =======================================
	Benchmark Execution
       =======================================
    */

    /* Begin benchmark */
    if (print_header_flag) print_header();
    
    current_ws = worksets-1;
    /* Time OpenCL Kernel */
    #ifdef USE_OPENCL

        os = 0; 
        ot = 0; 
        oi = 0;

        global_work_size = si.len / vector_len;
        cl_ulong start = 0, end = 0; 
        for (int i = 0; i <= R; i++) {
             
            start = 0; end = 0;
            ot = current_ws * target.len;
            os = current_ws * source.len;
            oi = current_ws * si.len;

           cl_event e = 0; 

            SET_7_KERNEL_ARGS(sgp, target.dev_ptr, source.dev_ptr,
                    ti.dev_ptr, si.dev_ptr, ot, os, oi);

            CALL_CL_GUARDED(clEnqueueNDRangeKernel, (queue, sgp, work_dim, NULL, 
                       &global_work_size, &local_work_size, 
                      0, NULL, &e)); 
            clWaitForEvents(1, &e);

            CALL_CL_GUARDED(clGetEventProfilingInfo, 
                    (e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL));
            CALL_CL_GUARDED(clGetEventProfilingInfo, 
                    (e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL));

            cl_ulong time_ns = end - start;
            double time_s = time_ns / 1000000000.;
            if (i!=0) report_time(time_s, source.size, target.size, index_len, worksets);

            current_ws = posmod(current_ws-1, worksets);

        }

    #endif // USE_OPENCL


    /* Time OpenMP Kernel */

    #ifdef USE_OPENMP
        current_ws = worksets-1;
        omp_set_num_threads(workers);
        for (int i = 0; i <= R; i++) {

            ot = current_ws * target.len;
            os = current_ws * source.len;
            oi = current_ws * si.len;

            zero_time();

            switch (kernel) {
                case SG:
                    if (op == OP_COPY) 
                        sg_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len, ot, os, oi, block_len);
                    else 
                        sg_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len, ot, os, oi, block_len);
                    break;
                case SCATTER:
                    if (op == OP_COPY)
                        scatter_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len, ot, os, oi, block_len);
                    else
                        scatter_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len, ot, os, oi, block_len);
                    break;
                case GATHER:
                    if (op == OP_COPY)
                        gather_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len, ot, os, oi, block_len);
                    else
                        gather_accum_omp (target.host_ptr, ti.host_ptr, source.host_ptr, si.host_ptr, 
                        index_len, ot, os, oi, block_len);
                    break;
                default:
                    printf("Error: Unable to determine kernel\n");
                    break;
            }

            double time_ms = get_time();
            if (i!=0) report_time(time_ms/1000., source.size, target.size, si.size, worksets);
            current_ws = posmod(current_ws-1, worksets);

        }

    #endif // USE_OPENMP
    

    // Validate results 
    if(validate_flag) {

#ifdef USE_OPENCL
        if (backend == OPENCL) {
            clEnqueueReadBuffer(queue, target.dev_ptr, 1, 0, target.size, 
                target.host_ptr, 0, NULL, &e);
            clWaitForEvents(1, &e);
        }
#endif

        SGTYPE_C *target_backup_host = (SGTYPE_C*) sg_safe_cpu_alloc(target.len * sizeof(SGTYPE_C)); 
        memcpy(target_backup_host, target.host_ptr, target.size);

        switch (kernel) {
            case SG:
                for (size_t i = 0; i < index_len; i++){
                    for (size_t b = 0; b < block_len; b++) {
                        target.host_ptr[ti.host_ptr[i+b]] = source.host_ptr[si.host_ptr[i+b]];
                    }
                }
                break;
            case SCATTER:
                for (size_t i = 0; i < index_len; i++){
                    for (size_t b = 0; b < block_len; b++) {
                        target.host_ptr[ti.host_ptr[i+b]] = source.host_ptr[i+b];
                    }
                }
                break;
            case GATHER:
                for (size_t i = 0; i < index_len; i++){
                    for (size_t b = 0; b < block_len; b++) {
                        target.host_ptr[i+b] = source.host_ptr[si.host_ptr[i+b]];
                    }
                }
                break;
        }


        for (size_t i = 0; i < target.len; i++) {
            if (target.host_ptr[i] != target_backup_host[i]) {
                printf(":(\n");
            }
        }
    }
}