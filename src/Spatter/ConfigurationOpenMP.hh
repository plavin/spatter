#ifndef CONFIGURATIONOPENMP_HH
#define CONFIGURATIONOPENMP_HH
#include "ConfigurationBase.hh"

namespace Spatter {

#ifdef USE_OPENMP
#include <omp.h>

template <> class Configuration<Spatter::OpenMP> : public ConfigurationBase {
public:
  Configuration(const size_t id, const std::string name,
      const std::string kernel, const aligned_vector<size_t> &pattern,
      const aligned_vector<size_t> &pattern_gather,
      aligned_vector<size_t> &pattern_scatter,
      aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
      aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
      size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
      double *&dev_sparse_scatter, size_t &sparse_scatter_size,
      aligned_vector<double> &dense,
      aligned_vector<aligned_vector<double>> &dense_perthread,
      double *&dev_dense, size_t &dense_size, const size_t delta,
      const size_t delta_gather, const size_t delta_scatter,
      const long int seed, const size_t wrap, const size_t count,
      const int nthreads, const unsigned long nruns, const bool aggregate,
      const bool atomic, const bool atomic_fence, const bool dense_buffers,
      const unsigned long verbosity);

  int run(bool timed, unsigned long run_id);

  void gather(bool timed, unsigned long run_id);
  void scatter(bool timed, unsigned long run_id);
  void gather_scatter(bool timed, unsigned long run_id);
  void multi_gather(bool timed, unsigned long run_id);
  void multi_scatter(bool timed, unsigned long run_id);
};
#endif

}

#endif
