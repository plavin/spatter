CL_HELPER_NO_COMPILER_OUTPUT_NAG=1 ./sgbench --backend=openmp --source-len=1000 --target-len=1000 --index-len=1000 --kernel-name=sg --runs=2 --block-len=30 --validate -W10 --verbose
