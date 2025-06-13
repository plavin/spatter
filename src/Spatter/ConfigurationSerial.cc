#include "ConfigurationSerial.hh"

namespace Spatter {
  
  Configuration<Spatter::Serial>::Configuration(const size_t id,
      const std::string name, const std::string kernel,
      const aligned_vector<size_t> &pattern,
      const aligned_vector<size_t> &pattern_gather,
      const aligned_vector<size_t> &pattern_scatter,
      aligned_vector<double> &sparse, double *&dev_sparse, size_t &sparse_size,
      aligned_vector<double> &sparse_gather, double *&dev_sparse_gather,
      size_t &sparse_gather_size, aligned_vector<double> &sparse_scatter,
      double *&dev_sparse_scatter, size_t &sparse_scatter_size,
      aligned_vector<double> &dense,
      aligned_vector<aligned_vector<double>> &dense_perthread,
      double *&dev_dense, size_t &dense_size,const size_t delta,
      const size_t delta_gather, const size_t delta_scatter, const long int seed,
      const size_t wrap, const size_t count, const unsigned long nruns,
      const bool aggregate, const unsigned long verbosity)
      : ConfigurationBase(id, name, kernel, pattern, pattern_gather,
            pattern_scatter, sparse, dev_sparse, sparse_size, sparse_gather,
            dev_sparse_gather, sparse_gather_size, sparse_scatter,
            dev_sparse_scatter, sparse_scatter_size, dense, dense_perthread,
            dev_dense, dense_size, delta, delta_gather,
            delta_scatter, seed, wrap, count, 0, 1024, 1, nruns, aggregate, false,
            false, false, verbosity) {
    ConfigurationBase::setup();
  }

  void Configuration<Spatter::Serial>::gather(bool timed, unsigned long run_id) {
    size_t pattern_length = pattern.size();

  #ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        dense[j + pattern_length * (i % wrap)] = sparse[pattern[j] + delta * i];

    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
  }

  void Configuration<Spatter::Serial>::scatter(bool timed, unsigned long run_id) {
    size_t pattern_length = pattern.size();

  #ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse[pattern[j] + delta * i] = dense[j + pattern_length * (i % wrap)];

    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
  }

  void Configuration<Spatter::Serial>::gather_scatter(
      bool timed, unsigned long run_id) {
    assert(pattern_scatter.size() == pattern_gather.size());
    size_t pattern_length = pattern_scatter.size();

  #ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse_scatter[pattern_scatter[j] + delta_scatter * i] =
            sparse_gather[pattern_gather[j] + delta_gather * i];

    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
  }

  void Configuration<Spatter::Serial>::multi_gather(
      bool timed, unsigned long run_id) {
    size_t pattern_length = pattern_gather.size();

  #ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        dense[j + pattern_length * (i % wrap)] =
            sparse[pattern[pattern_gather[j]] + delta * i];

    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
  }

  void Configuration<Spatter::Serial>::multi_scatter(
      bool timed, unsigned long run_id) {
    size_t pattern_length = pattern_scatter.size();

  #ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
  #endif

    if (timed)
      timer.start();

    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < pattern_length; ++j)
        sparse[pattern[pattern_scatter[j]] + delta * i] =
            dense[j + pattern_length * (i % wrap)];

    if (timed) {
      timer.stop();
      time_seconds[run_id] = timer.seconds();
      timer.clear();
    }
  }
}
