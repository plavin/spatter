#include "ConfigurationOpenMP.hh"

namespace Spatter {

#ifdef USE_MPI
#include "mpi.h"
#endif

#ifdef USE_OPENMP
Configuration<Spatter::OpenMP>::Configuration(const size_t id,
    const std::string name, const std::string kernel,
    const aligned_vector<size_t> &pattern,
    const aligned_vector<size_t> &pattern_gather,
    aligned_vector<size_t> &pattern_scatter,
    aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
    aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
    size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
    double *&dev_sparse_scatter, size_t &sparse_scatter_size,
    aligned_vector<double> &dense,
    aligned_vector<aligned_vector<double>> &dense_perthread,
    double *&dev_dense, size_t &dense_size,const size_t delta,
    const size_t delta_gather, const size_t delta_scatter, const long int seed,
    const size_t wrap, const size_t count, const int nthreads,
    const unsigned long nruns, const bool aggregate, const bool atomic,
    const bool atomic_fence, const bool dense_buffers,
    const unsigned long verbosity)
    : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
          pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
          dev_sparse_gather, sparse_gather_size, sparse_scatter,
          dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
          dev_dense, dense_size, delta, delta_gather, delta_scatter, seed, wrap,
          count, 0, 1024, nthreads, nruns, aggregate, atomic, atomic_fence,
          dense_buffers, verbosity) {
  ConfigurationBase::setup();
}

int Configuration<Spatter::OpenMP>::run(bool timed, unsigned long run_id) {
  omp_set_num_threads(omp_threads);
  return ConfigurationBase::run(timed, run_id);
}

void Configuration<Spatter::OpenMP>::gather(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();
    double *source = sparse.data();
    double *target = (dense_buffers ? dense_perthread[t].data() : dense.data());

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *sl = source + delta * i;
      double *tl = target + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[j] = sl[pattern[j]];
      }
    }
  }

  if (atomic_fence)
    std::atomic_thread_fence(std::memory_order_release);

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }

}

void Configuration<Spatter::OpenMP>::scatter(bool timed, unsigned long run_id) {
  size_t pattern_length = pattern.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();
    double *source = (dense_buffers ? dense_perthread[t].data() : dense.data());
    double *target = sparse.data();

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *tl = target + delta * i;
      double *sl = source + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[pattern[j]] = sl[j];
      }
    }
  }

  if (atomic_fence)
    std::atomic_thread_fence(std::memory_order_release);

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::OpenMP>::gather_scatter(
    bool timed, unsigned long run_id) {
  assert(pattern_scatter.size() == pattern_gather.size());
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel for
  for (size_t i = 0; i < count; ++i) {
    double *tl = sparse_scatter.data() + delta_scatter * i;
    double *sl = sparse_gather.data() + delta_gather * i;

#pragma omp simd
    for (size_t j = 0; j < pattern_length; ++j) {
      tl[pattern_scatter[j]] = sl[pattern_gather[j]];
    }
  }

  if (atomic_fence)
    std::atomic_thread_fence(std::memory_order_release);

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}

void Configuration<Spatter::OpenMP>::multi_gather(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_gather.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();
    double *source = sparse.data();
    double *target = (dense_buffers ? dense_perthread[t].data() : dense.data());

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *sl = source + delta * i;
      double *tl = target + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[j] = sl[pattern[pattern_gather[j]]];
      }
    }
  }

  if (atomic_fence)
    std::atomic_thread_fence(std::memory_order_release);

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}


void Configuration<Spatter::OpenMP>::multi_scatter(
    bool timed, unsigned long run_id) {
  size_t pattern_length = pattern_scatter.size();

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  if (timed)
    timer.start();

#pragma omp parallel
  {
    int t = omp_get_thread_num();
    double *target = sparse.data();
    double *source = (dense_buffers ? dense_perthread[t].data() : dense.data());

#pragma omp for
    for (size_t i = 0; i < count; ++i) {
      double *tl = target + delta * i;
      double *sl = source + pattern_length * (i % wrap);

#pragma omp simd
      for (size_t j = 0; j < pattern_length; ++j) {
        tl[pattern[pattern_scatter[j]]] = sl[j];
      }
    }
  }

  if (atomic_fence)
    std::atomic_thread_fence(std::memory_order_release);

  if (timed) {
    timer.stop();
    time_seconds[run_id] = timer.seconds();
    timer.clear();
  }
}
#endif
}

