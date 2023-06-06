#ifndef KGRAPH_VERSION
#define KGRAPH_VERSION unknown
#endif
#ifndef KGRAPH_BUILD_NUMBER
#define KGRAPH_BUILD_NUMBER
#endif
#ifndef KGRAPH_BUILD_ID
#define KGRAPH_BUILD_ID
#endif
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#define STRINGIFY_HELPER(x) #x
static char const *kgraph_version = STRINGIFY(KGRAPH_VERSION) "-" STRINGIFY(
    KGRAPH_BUILD_NUMBER) "," STRINGIFY(KGRAPH_BUILD_ID);
#define likely(x) __builtin_expect((x), 1)
#define unlikely(x) __builtin_expect((x), 0)
#ifdef _OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <boost/timer/timer.hpp>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <unordered_set>
#define timer timer_for_boost_progress_t
#include <boost/progress.hpp>
#include <boost/timer/timer.hpp>
#undef timer
#include <atomic>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/dynamic_bitset.hpp>
// #include <boost/circular_buffer.hpp>
#include "boost/smart_ptr/detail/spinlock.hpp"
#include "kgraph.h"
#include "kgraph-data.h"

namespace kgraph {

using namespace std;
using namespace boost;
using namespace boost::accumulators;

unsigned verbosity = default_verbosity;

typedef boost::detail::spinlock Lock;
typedef std::lock_guard<Lock> LockGuard;

template <typename RNG>
static void GenRandom(RNG &rng, unsigned *addr, unsigned size, unsigned N) {
  if (N == size) {
    std::iota(addr, addr + size, 0);
    return;
  }

  for (unsigned i = 0; i < size; ++i) {
    addr[i] = rng() % (N - size);
  }
  sort(addr, addr + size);
  for (unsigned i = 1; i < size; ++i) {
    if (addr[i] <= addr[i - 1]) {
      addr[i] = addr[i - 1] + 1;
    }
  }
  unsigned off = rng() % N;
  for (unsigned i = 0; i < size; ++i) {
    addr[i] = (addr[i] + off) % N;
  }
}

struct Neighbor {
  uint32_t id;
  float dist;
  bool flag;  // whether this entry is a newly found one
  Neighbor() {
    dist = 0;
    id = 0;
    flag = false;
  }
  Neighbor(unsigned i) : id(i) {}
  Neighbor(unsigned i, float d, bool f = true) : id(i), dist(d), flag(f) {}
};

// extended neighbor structure for search time
struct NeighborX : public Neighbor {
  uint16_t m;
  uint16_t M;  // actual M used
  NeighborX() {}
  NeighborX(unsigned i, float d) : Neighbor(i, d, true), m(0), M(0) {}
};

static inline bool operator<(Neighbor const &n1, Neighbor const &n2) {
  return n1.dist < n2.dist;
}

static inline bool operator==(Neighbor const &n1, Neighbor const &n2) {
  return n1.id == n2.id;
}

typedef vector<Neighbor> Neighbors;

// both pool and knn should be sorted in ascending order
static float EvaluateRecall(Neighbors const &pool, Neighbors const &knn) {
  if (knn.empty()) return 1.0;
  unsigned found = 0;
  unsigned n_p = 0;
  unsigned n_k = 0;
  for (;;) {
    if (n_p >= pool.size()) break;
    if (n_k >= knn.size()) break;
    if (knn[n_k].dist < pool[n_p].dist) {
      ++n_k;
    } else if (knn[n_k].dist == pool[n_p].dist) {
      ++found;
      ++n_k;
      ++n_p;
    } else {
      cerr << "Distance is unstable." << endl;
      cerr << "Exact";
      for (auto const &p : knn) {
        cerr << ' ' << p.id << ':' << p.dist;
      }
      cerr << endl;
      cerr << "Approx";
      for (auto const &p : pool) {
        cerr << ' ' << p.id << ':' << p.dist;
      }
      cerr << endl;
      throw runtime_error("distance is unstable");
    }
  }
  return float(found) / knn.size();
}

static float EvaluateAccuracy(Neighbors const &pool, Neighbors const &knn) {
  unsigned m = std::min(pool.size(), knn.size());
  float sum = 0;
  unsigned cnt = 0;
  for (unsigned i = 0; i < m; ++i) {
    if (knn[i].dist > 0) {
      sum += abs(pool[i].dist - knn[i].dist) / knn[i].dist;
      ++cnt;
    }
  }
  return cnt > 0 ? sum / cnt : 0;
}

static float EvaluateOneRecall(Neighbors const &pool, Neighbors const &knn) {
  if (pool[0].dist == knn[0].dist) return 1.0;
  return 0;
}

static float EvaluateDelta(Neighbors const &pool, unsigned K) {
  unsigned c = 0;
  unsigned N = K;
  if (pool.size() < N) N = pool.size();
  for (unsigned i = 0; i < N; ++i) {
    if (pool[i].flag) ++c;
  }
  return float(c) / K;
}

struct Control {
  unsigned id;
  Neighbors neighbors;
};

template <typename NeighborT>
inline unsigned UpdateKnnListHelper(NeighborT *addr, unsigned &K,
                                    NeighborT nn) {
  // optimize with memmove, binary search
  unsigned i = 0, j = K;
  while (i < j) {
    unsigned m = (i + j) >> 1;
    if (likely(nn.dist > addr[m].dist)) {
      i = m + 1;
    } else if (unlikely(nn.dist < addr[m].dist)) {
      j = m;
    } else {  // handle equal distances
      if (nn.id == addr[m].id) {
        return K;  // neighbor with same ID already exists
      } else if (nn.id < addr[m].id) {
        j = m;
      } else {
        i = m + 1;
      }
    }
  }

  std::memmove(addr + i + 1, addr + i, (K - i) * sizeof(NeighborT));

  addr[i] = nn;
  return i;
}

static inline unsigned UpdateKnnList(Neighbor *addr, unsigned K, Neighbor nn) {
  prefetch_vector((char *)addr, K * sizeof(Neighbor));
  return UpdateKnnListHelper<Neighbor>(addr, K, nn);
}

static inline unsigned UpdateKnnList(NeighborX *addr, unsigned K,
                                     NeighborX nn) {
  return UpdateKnnListHelper<NeighborX>(addr, K, nn);
}

// The neighborhood structure maintains a pool of near neighbors of an object.
// The neighbors are stored in the pool.  "n" (<="params.L") is the number of
// valid entries in the pool, with the beginning "k" (<="n") entries sorted.
struct Nhood {   // neighborhood
  float radius;  // distance of interesting range
  Lock lock;
  float radiusM;
  Neighbors pool;
  unsigned L;  // # valid items in the pool,  L + 1 <= pool.size()
  unsigned M;  // we only join items in pool[0..M)
  bool found;  // helped found new NN in this round
  vector<unsigned> nn_old;
  vector<unsigned> nn_new;
  vector<unsigned> rnn_old;
  vector<unsigned> rnn_new;
  // std::atomic<uint32_t> hub_score;
  // boost::circular_buffer<uint32_t> expel_keep;

  void clear_all() {
    std::vector<uint32_t>().swap(nn_old);
    std::vector<uint32_t>().swap(nn_new);
    std::vector<uint32_t>().swap(rnn_old);
    std::vector<uint32_t>().swap(rnn_new);
    radius = pool.back().dist;
  }
  // only non-readonly method which is supposed to be called in parallel
  void parallel_try_insert(unsigned id, float &dist) {
    if (dist > radius) {
      return;
    }

    LockGuard guard(lock);

    unsigned l = UpdateKnnList(&pool[0], L, Neighbor(id, dist, true));

    if (l <= L) {  // inserted
      if (unlikely(L + 1 <
                   pool.size())) {  // if l == L + 1, there's a duplicate
        ++L;
      } else {
        radius = pool[L - 1].dist;
      }
    }
  }
};

inline void LinearSearch(IndexOracle const &oracle, unsigned i, unsigned K,
                         vector<Neighbor> *pnns) {
  vector<Neighbor> nns(K + 1);
  unsigned N = oracle.size();
  Neighbor nn;
  nn.id = 0;
  nn.flag = true;  // we don't really use this
  unsigned k = 0;
  while (nn.id < N) {
    if (nn.id != i) {
      nn.dist = oracle(i, nn.id);
      UpdateKnnList(&nns[0], k, nn);
      if (k < K) ++k;
    }
    ++nn.id;
  }
  nns.resize(K);
  pnns->swap(nns);
}

unsigned SearchOracle::search(unsigned K, float epsilon, unsigned *ids,
                              float *dists) const {
  vector<Neighbor> nns(K + 1);
  unsigned N = size();
  unsigned L = 0;
  for (unsigned k = 0; k < N; ++k) {
    float k_dist = operator()(k);
    if (k_dist > epsilon) continue;
    UpdateKnnList(&nns[0], L, Neighbor(k, k_dist));
    if (L < K) ++L;
  }
  if (ids) {
    for (unsigned k = 0; k < L; ++k) {
      ids[k] = nns[k].id;
    }
  }
  if (dists) {
    for (unsigned k = 0; k < L; ++k) {
      dists[k] = nns[k].dist;
    }
  }
  return L;
}

void GenerateControl(IndexOracle const &oracle, unsigned C, unsigned K,
                     vector<Control> &pcontrols) {
  // auto controls = pcontrols;
  {
    vector<unsigned> index(oracle.size());
    int i = 0;
    for (unsigned &v : index) {
      v = i++;
    }
    std::random_device rd;
    std::default_random_engine rng(rd());
    std::shuffle(index.begin(), index.end(), rng);
    // random_shuffle(index.begin(), index.end());
#pragma omp parallel for
    for (unsigned i = 0; i < C; ++i) {
      pcontrols[i].id = index[i];
      LinearSearch(oracle, index[i], K, &pcontrols[i].neighbors);
    }
  }
  // pcontrols->swap(controls);
}

static char const *KGRAPH_MAGIC = "KNNGRAPH";
static unsigned constexpr KGRAPH_MAGIC_SIZE = 8;
static uint32_t constexpr SIGNATURE_VERSION = 2;

class KGraphImpl : public KGraph {
 protected:
  // std::bitset<10000000> hub_mask;
  vector<unsigned> M;
  vector<vector<Neighbor>> graph;
  vector<Nhood> nhoods;
  // vector<vector<uint32_t>> graph_only_ids;
  bool no_dist;  // Distance & flag information in Neighbor is not valid.

  // actual M for a node that should be used in search time
  unsigned actual_M(unsigned pM, unsigned i) const {
    return std::min(std::max(M[i], pM), unsigned(graph[i].size()));
  }

 public:
  virtual ~KGraphImpl() {}

  virtual void load(char const *path) {
    static_assert(sizeof(unsigned) == sizeof(uint32_t),
                  "unsigned must be 32-bit");
    ifstream is(path, ios::binary);
    char magic[KGRAPH_MAGIC_SIZE];
    uint32_t sig_version;
    uint32_t sig_cap;
    uint32_t N;
    is.read(magic, sizeof(magic));
    is.read(reinterpret_cast<char *>(&sig_version), sizeof(sig_version));
    is.read(reinterpret_cast<char *>(&sig_cap), sizeof(sig_cap));
    if (sig_version != SIGNATURE_VERSION)
      throw runtime_error("data version not supported.");
    is.read(reinterpret_cast<char *>(&N), sizeof(N));
    if (!is) runtime_error("error reading index file.");
    for (unsigned i = 0; i < KGRAPH_MAGIC_SIZE; ++i) {
      if (KGRAPH_MAGIC[i] != magic[i]) runtime_error("index corrupted.");
    }
    no_dist = sig_cap & FORMAT_NO_DIST;
    graph.resize(N);
    M.resize(N);
    vector<uint32_t> nids;
    for (unsigned i = 0; i < graph.size(); ++i) {
      auto &knn = graph[i];
      unsigned K;
      is.read(reinterpret_cast<char *>(&M[i]), sizeof(M[i]));
      is.read(reinterpret_cast<char *>(&K), sizeof(K));
      if (!is) runtime_error("error reading index file.");
      knn.resize(K);
      if (no_dist) {
        nids.resize(K);
        is.read(reinterpret_cast<char *>(&nids[0]), K * sizeof(nids[0]));
        for (unsigned k = 0; k < K; ++k) {
          knn[k].id = nids[k];
          knn[k].dist = 0;
          knn[k].flag = false;
        }
      } else {
        is.read(reinterpret_cast<char *>(&knn[0]), K * sizeof(knn[0]));
      }
    }
  }

  virtual void save(char const *path, int format) const {
    ofstream os(path, ios::binary | ios::trunc);
    const unsigned graphSize = nhoods.size();
    const unsigned K = 100;
    std::vector<uint32_t> buffer(K);
    const int bufferSize = K * sizeof(uint32_t);
    for (unsigned i = 0; i < graphSize; ++i) {
      auto &knn = nhoods[i].pool;
      // auto& knn = graph[i];
      std::transform(knn.begin(), knn.end(), buffer.begin(),
                     [](auto const &x) { return x.id; });
      os.write(reinterpret_cast<char const *>(buffer.data()), bufferSize);
    }
    os.close();
  }

  virtual void build(IndexOracle const &oracle, IndexParams const &param,
                     IndexInfo *info);

  /*
  virtual void prune (unsigned K) {
      for (auto &v: graph) {
          if (v.size() > K) {
              v.resize(K);
          }
      }
      return;
      vector<vector<unsigned>> pruned(graph.size());
      vector<set<unsigned>> reachable(graph.size());
      vector<bool> added(graph.size());
      for (unsigned k = 0; k < K; ++k) {
#pragma omp parallel for
          for (unsigned n = 0; n < graph.size(); ++n) {
              vector<unsigned> const &from = graph[n];
              if (from.size() <= k) continue;
              unsigned e = from[k];
              if (reachable[n].count(e)) {
                  added[n] = false;
              }
              else {
                  pruned[n].emplace_back(e);
                  added[n] = true;
              }
          }
          // expand reachable
#pragma omp parallel for
          for (unsigned n = 0; n < graph.size(); ++n) {
              vector<unsigned> const &to = pruned[n];
              set<unsigned> &nn = reachable[n];
              if (added[n]) {
                  for (unsigned v: pruned[to.back()]) {
                      nn.insert(v);
                  }
              }
              for (unsigned v: to) {
                  if (added[v]) {
                      nn.insert(pruned[v].back());
                  }
              }
          }
      }
      graph.swap(pruned);
  }
  */

  virtual unsigned search(SearchOracle const &oracle,
                          SearchParams const &params, unsigned *ids,
                          float *dists, SearchInfo *pinfo) const {
    if (graph.size() > oracle.size()) {
      throw runtime_error("dataset larger than index");
    }
    if (params.P >= graph.size()) {
      if (pinfo) {
        pinfo->updates = 0;
        pinfo->cost = 1.0;
      }
      return oracle.search(params.K, params.epsilon, ids, dists);
    }
    vector<NeighborX> knn(params.K + params.P + 1);
    vector<NeighborX> results;
    // flags access is totally random, so use small block to avoid
    // extra memory access
    boost::dynamic_bitset<> flags(graph.size(), false);

    if (params.init && params.T > 1) {
      throw runtime_error("when init > 0, T must be 1.");
    }

    unsigned seed = params.seed;
    unsigned updates = 0;
    if (seed == 0) seed = time(NULL);
    mt19937 rng(seed);
    unsigned n_comps = 0;
    for (unsigned trial = 0; trial < params.T; ++trial) {
      unsigned L = params.init;
      if (L == 0) {  // generate random starting points
        vector<unsigned> random(params.P);
        GenRandom(rng, &random[0], random.size(), graph.size());
        for (unsigned s : random) {
          if (!flags[s]) {
            knn[L++].id = s;
            // flags[s] = true;
          }
        }
      } else {  // user-provided starting points.
        if (!ids) throw invalid_argument("no initial data provided via ids");
        if (!(L < params.K)) throw invalid_argument("L < params.K");
        for (unsigned l = 0; l < L; ++l) {
          knn[l].id = ids[l];
        }
      }
      for (unsigned k = 0; k < L; ++k) {
        auto &e = knn[k];
        flags[e.id] = true;
        e.flag = true;
        e.dist = oracle(e.id);
        e.m = 0;
        e.M = actual_M(params.M, e.id);
      }
      sort(knn.begin(), knn.begin() + L);

      unsigned k = 0;
      while (k < L) {
        auto &e = knn[k];
        if (!e.flag) {  // all neighbors of this node checked
          ++k;
          continue;
        }
        unsigned beginM = e.m;
        unsigned endM = beginM + params.S;  // check this many entries
        if (endM > e.M) {                   // we are done with this node
          e.flag = false;
          endM = e.M;
        }
        e.m = endM;
        // all modification to knn[k] must have been done now,
        // as we might be relocating knn[k] in the loop below
        auto const &neighbors = graph[e.id];
        for (unsigned m = beginM; m < endM; ++m) {
          unsigned id = neighbors[m].id;
          // BOOST_VERIFY(id < graph.size());
          if (flags[id]) continue;
          flags[id] = true;
          ++n_comps;
          float dist = oracle(id);
          NeighborX nn(id, dist);
          unsigned r = UpdateKnnList(&knn[0], L, nn);
          BOOST_VERIFY(r <= L);
          // if (r > L) continue;
          if (L + 1 < knn.size()) ++L;
          if (r < L) {
            knn[r].M = actual_M(params.M, id);
            if (r < k) {
              k = r;
            }
          }
        }
      }
      if (L > params.K) L = params.K;
      if (results.empty()) {
        results.reserve(params.K + 1);
        results.resize(L + 1);
        copy(knn.begin(), knn.begin() + L, results.begin());
      } else {
        // update results
        for (unsigned l = 0; l < L; ++l) {
          unsigned r = UpdateKnnList(&results[0], results.size() - 1, knn[l]);
          if (r < results.size() /* inserted */ &&
              results.size() < (params.K + 1)) {
            results.resize(results.size() + 1);
          }
        }
      }
    }
    results.pop_back();
    // check epsilon
    {
      for (unsigned l = 0; l < results.size(); ++l) {
        if (results[l].dist > params.epsilon) {
          results.resize(l);
          break;
        }
      }
    }
    unsigned L = results.size();
    /*
    if (!(L <= params.K)) {
        cerr << L << ' ' << params.K << endl;
    }
    */
    if (!(L <= params.K)) throw runtime_error("L <= params.K");
    // check epsilon
    if (ids) {
      for (unsigned k = 0; k < L; ++k) {
        ids[k] = results[k].id;
      }
    }
    if (dists) {
      for (unsigned k = 0; k < L; ++k) {
        dists[k] = results[k].dist;
      }
    }
    if (pinfo) {
      pinfo->updates = updates;
      pinfo->cost = float(n_comps) / graph.size();
    }
    return L;
  }

  virtual void get_nn(unsigned id, unsigned *nns, float *dist, unsigned *pM,
                      unsigned *pL) const {
    if (!(id < graph.size())) throw invalid_argument("id too big");
    auto const &v = graph[id];
    *pM = M[id];
    *pL = v.size();
    if (nns) {
      for (unsigned i = 0; i < v.size(); ++i) {
        nns[i] = v[i].id;
      }
    }
    if (dist) {
      if (no_dist) throw runtime_error("distance information is not available");
      for (unsigned i = 0; i < v.size(); ++i) {
        dist[i] = v[i].dist;
      }
    }
  }

  void prune1() {
    for (unsigned i = 0; i < graph.size(); ++i) {
      if (graph[i].size() > M[i]) {
        graph[i].resize(M[i]);
      }
    }
  }

  void prune2() {
    vector<vector<unsigned>> reverse(graph.size());  // reverse of new graph
    vector<unsigned> new_L(graph.size(), 0);
    unsigned L = 0;
    unsigned total = 0;
    for (unsigned i = 0; i < graph.size(); ++i) {
      if (M[i] > L) L = M[i];
      total += M[i];
      for (auto &e : graph[i]) {
        e.flag = false;  // haven't been visited yet
      }
    }
    progress_display progress(total, cerr);
    vector<unsigned> todo(graph.size());
    for (unsigned i = 0; i < todo.size(); ++i) todo[i] = i;
    vector<unsigned> new_todo(graph.size());
    for (unsigned l = 0; todo.size(); ++l) {
      BOOST_VERIFY(l <= L);
      new_todo.clear();
      for (unsigned i : todo) {
        if (l >= M[i]) continue;
        new_todo.emplace_back(i);
        auto &v = graph[i];
        BOOST_VERIFY(l < v.size());
        if (v[l].flag) continue;  // we have visited this one already
        v[l].flag = true;         // now we have seen this one
        ++progress;
        unsigned T;
        {
          auto &nl = new_L[i];
          // shift the entry to add
          T = v[nl].id = v[l].id;
          v[nl].dist = v[l].dist;
          ++nl;
        }
        reverse[T].emplace_back(i);
        {
          auto const &u = graph[T];
          for (unsigned ll = l + 1; ll < M[i]; ++ll) {
            if (v[ll].flag) continue;
            for (unsigned j = 0; j < new_L[T]; ++j) {  // new graph
              if (v[ll].id == u[j].id) {
                v[ll].flag = true;
                ++progress;
                break;
              }
            }
          }
        }
        {
          for (auto r : reverse[i]) {
            auto &u = graph[r];
            for (unsigned ll = l; ll < M[r]; ++ll) {
              // must start from l: as item l might not have been checked
              // for reverse
              if (u[ll].id == T) {
                if (!u[ll].flag) ++progress;
                u[ll].flag = true;
              }
            }
          }
        }
      }
      todo.swap(new_todo);
    }
    BOOST_VERIFY(progress.count() == total);
    M.swap(new_L);
    prune1();
  }

  virtual void prune(IndexOracle const &oracle, unsigned level) {
    if (level & PRUNE_LEVEL_1) {
      prune1();
    }
    if (level & PRUNE_LEVEL_2) {
      prune2();
    }
  }

  void reverse(int rev_k) {
    if (rev_k == 0) return;
    if (no_dist)
      throw runtime_error("Need distance information to reverse graph");
    {
      cerr << "Graph completion with reverse edges..." << endl;
      vector<vector<Neighbor>> ng(
          graph.size());  // new graph adds on original one
      // ng = graph;
      progress_display progress(graph.size(), cerr);
      for (unsigned i = 0; i < graph.size(); ++i) {
        auto const &v = graph[i];
        unsigned K = M[i];
        if (rev_k > 0) {
          K = rev_k;
          if (K > v.size()) K = v.size();
        }
        // if (v.size() < XX) XX = v.size();
        for (unsigned j = 0; j < K; ++j) {
          auto const &e = v[j];
          auto re = e;
          re.id = i;
          ng[i].emplace_back(e);
          ng[e.id].emplace_back(re);
        }
        ++progress;
      }
      graph.swap(ng);
    }
    {
      cerr << "Reranking edges..." << endl;
      progress_display progress(graph.size(), cerr);
#pragma omp parallel for
      for (unsigned i = 0; i < graph.size(); ++i) {
        auto &v = graph[i];
        std::sort(v.begin(), v.end());
        v.resize(std::unique(v.begin(), v.end()) - v.begin());
        M[i] = v.size();
#pragma omp critical
        ++progress;
      }
    }
  }
};

class KGraphConstructor : public KGraphImpl {
  // // The neighborhood structure maintains a pool of near neighbors of an
  // object.
  // // The neighbors are stored in the pool.  "n" (<="params.L") is the number
  // of valid entries
  // // in the pool, with the beginning "k" (<="n") entries sorted.
 public:
  void load_nn_graph(char const *path, uint32_t &K) {
    std::ifstream in(path, std::ios::binary);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    size_t num = fsize / ((size_t)K) / 4;
    in.seekg(0, std::ios::beg);
    id_graph.resize(num);
    for (size_t i = 0; i < num; i++) {
      id_graph[i].resize(K);
      in.read((char *)id_graph[i].data(), K * sizeof(unsigned));
    }
    std::cout << "Load pre NN graph: " << id_graph.size()
              << "nn: " << id_graph[0].size() << std::endl;
    in.close();
  }

  void prepare_nhoods() {
    unsigned N = oracle.size();
    for (auto &nhood : nhoods) {
      nhood.nn_new.resize(params.S * 2);
      nhood.pool.resize(params.L + 1);
      nhood.radius = numeric_limits<float>::max();
    }
    const uint32_t K = params.K;
    BOOST_VERIFY(K == id_graph[0].size());
    std::random_device rd;
#pragma omp parallel for schedule(dynamic, 1)
    for (uint32_t i = 0; i < N; ++i) {
      thread_local mt19937 rng(rd());
      std::vector<uint32_t> &nn = id_graph[i];
      auto &nhood = nhoods[i];
      auto &pool = nhood.pool;
      nhood.L = params.L;
      nhood.M = params.L;
      std::vector<uint32_t> random(pool.size());
      GenRandom(rng, &random[0], random.size(), N);
      std::copy(nn.begin(), nn.begin() + K, nhood.nn_new.begin());

      for (uint32_t p = 0; p < nhood.L; ++p) {
        auto &cur_id = random[p];
        float dist = oracle(cur_id, i);
        pool[p] = Neighbor(cur_id, dist, true);
      }
      GenRandom(rng, &nhood.nn_new[K], nhood.nn_new.size() - K, N);
      std::sort(pool.begin(), pool.begin() + nhood.L);
    }
  }

 public:
  vector<Nhood> nhoods;
  std::vector<std::vector<uint32_t>> id_graph;

 private:
  IndexOracle const &oracle;
  IndexParams params;
  IndexInfo *pinfo;
  size_t n_comps;

  void init() {
    unsigned N = oracle.size();
    unsigned seed = params.seed;
    for (auto &nhood : nhoods) {
      // nhood.expel_keep.set_capacity(20);
      nhood.nn_new.resize(params.S * 2);
      nhood.pool.resize(params.L + 1);
      nhood.radius = numeric_limits<float>::max();
    }
    std::random_device rd;
#pragma omp parallel
    {
#ifdef _OPENMP
      // mt19937 rng(seed ^ omp_get_thread_num());
      thread_local mt19937 rng(rd());
#else
      mt19937 rng(rd());
      // mt19937 rng(seed);
#endif
      vector<unsigned> random(params.L + 1);
#pragma omp for
      for (unsigned n = 0; n < N; ++n) {
        auto &nhood = nhoods[n];
        // prefetch_vector_l2((char *)nhood.nn_new.data(),
        // nhoods[n].nn_new.size() * sizeof(uint32_t)); prefetch_vector_l2((char *)random.data(), nhoods[n].nn_new.size() * sizeof(uint32_t));
        // _mm_prefetch((char *)random.data(), _MM_HINT_T0);
        Neighbors &pool = nhood.pool;
        // prefetch_vector_l2((char *)pool.data(), sizeof(Neighbor));
        GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(),
                  N);  // at the beginning all random
        GenRandom(rng, &random[0], random.size(), N);
        nhood.L = params.L;
        nhood.M = params.S;
        unsigned i = 0;
        for (unsigned l = 0; l < nhood.L; ++l) {
          if (random[i] == n) ++i;
          auto &nn = nhood.pool[l];
          nn.id = random[i++];
          nn.dist = oracle(nn.id, n);
          nn.flag = true;
        }
        sort(pool.begin(), pool.begin() + nhood.L);
      }
    }
  }

  inline void inner_join_func(unsigned i, unsigned j) {
    float dist = oracle(i, j);
    nhoods[i].parallel_try_insert(j, dist);
    nhoods[j].parallel_try_insert(i, dist);
  }

  void join(IndexInfo &info) {
#pragma omp parallel for schedule(dynamic, 100)
    for (unsigned n = 0; n < oracle.size(); ++n) {
      auto &nn_new = nhoods[n].nn_new;
      auto &nn_old = nhoods[n].nn_old;

      for (unsigned i = 0; i < nn_new.size(); ++i) {
        for (unsigned j = i + 1; j < nn_new.size(); ++j) {
          inner_join_func(nn_new[i], nn_new[j]);
        }
        for (unsigned j = 0; j < nn_old.size(); ++j) {
          if (unlikely(nn_new[i] != nn_old[j])) {
            inner_join_func(nn_new[i], nn_old[j]);
          }
        }
      }
    }
  }
  inline void update(IndexInfo &info) {
    unsigned N = oracle.size();
    // std::atomic<int> sum_new_size;
    // std::atomic<int> sum_old_size;
    // sum_new_size.store(0);
    // sum_old_size.store(0);
    // std::vector<uint32_t> permut(N);
    // std::iota(permut.begin(), permut.end(), 0);
    // std::random_shuffle(permut.begin(), permut.end());

#pragma omp parallel for schedule(dynamic, 100)
    for (unsigned n = 0; n < N; ++n) {
      auto &nhood = nhoods[n];
      nhood.clear_all();
      nhood.M = nhood.pool.size();
      BOOST_VERIFY(nhood.M > 0);
      nhood.radiusM = nhood.pool[nhood.M - 1].dist;

      // if (info.iterations >= 2) {
      //     if (nhood.hub_score >= 8 * params.K) {
      //         hub_mask.set(n, 0);
      //     }
      // }
    }
    // some random strategy, time consuming is unaffordable
    /*
    uint16_t expect_gen = params.K / 4;
    std::random_device rd;
    mt19937 rng(rd());
    std::uniform_int_distribution<uint32_t> n_dist(0, N-1);
    std::mt19937 bool_rng{std::random_device{}()};
    std::bernoulli_distribution bool_dist{0.5};   // Generates bool values with 50% chance of true
    */

    // limit new neighbor stategy
    // const uint32_t nn_new_limit = params.K;
#pragma omp parallel for schedule(dynamic, 100)
    for (unsigned n = 0; n < N; ++n) {
      // if (hub_mask.test(n) == 0) continue;

      auto &nhood = nhoods[n];
      auto &nn_new = nhood.nn_new;
      auto &nn_old = nhood.nn_old;
      for (unsigned l = 0; l < nhood.M; ++l) {
        auto &nn = nhood.pool[l];
        auto &nhood_o = nhoods[nn.id];  // nhood on the other side of the edge
        if (nn.flag) {
          // limit nn strategy
          /*
          if (nn_new.size() < nn_new_limit) {
              nn_new.emplace_back(nn.id);
              nn.flag = false;
          }
          else {
          nn_old.emplace_back(nn.id);
          }
          */
          nn_new.emplace_back(nn.id);
          if (nn.dist >
              nhood_o.radiusM) {  // maybe detect valuable neighbor via radiusM
            LockGuard guard(nhood_o.lock);
            nhood_o.rnn_new.emplace_back(n);  // add reverse
          }
          nn.flag = false;
          // if (hub_mask.test(n) == 0) continue;
        } else {
          nn_old.emplace_back(nn.id);
          // if (hub_mask.test(n) == 0) continue;
          if (nn.dist > nhood_o.radiusM) {
            LockGuard guard(nhood_o.lock);
            // nhood_o.nn_old.emplace_back(n);
            nhood_o.rnn_old.emplace_back(n);
          }
          // hubness aware
          /*
          if (likely(hub_mask.test(nn.id) == 1)) {
              if (fifty_fifty()) {
                  nn.flag = true;
              }
          }
          */
        }
      }
      // random join if iter greater than a threshold
      /*
      if (unlikely(info.delta < 6e-5)) {
          uint16_t pre_last = nn_new.size();
          nn_new.resize(nn_new.size() + expect_gen);
          for (auto& nn: nhood.pool) {
              if (fifty_fifty()) {
                  nn_new.emplace_back(nhoods[nn.id].pool[50].id);
              }
          }
          //todo pre last occupied the origin data
          // GenRandom(rng, &nn_new[pre_last], nn_new.size() - pre_last, N);
      }
      if (unlikely(info.delta < 6e-5)) {
          for (auto it = nhood.expel_keep.begin(); it != nhood.expel_keep.end();
      ++it) { nn_new.emplace_back(*it);
          }
          nhood.expel_keep.clear();
          // std::copy(nhood.expel_keep.begin(), nhood.expel_keep.end(),
      nn_new.end());
          //todo pre last occupied the origin data
          // GenRandom(rng, &nn_new[pre_last], nn_new.size() - pre_last, N);
      }
      */
    }
    std::random_device rd;
#pragma omp parallel for
    for (unsigned i = 0; i < N; ++i) {
      thread_local std::default_random_engine rng(rd());
      auto &nn_new = nhoods[i].nn_new;
      auto &nn_old = nhoods[i].nn_old;
      auto &rnn_new = nhoods[i].rnn_new;
      auto &rnn_old = nhoods[i].rnn_old;

      if ((rnn_new.size() > params.R)) {
        std::shuffle(rnn_new.begin(), rnn_new.end(), rng);
        rnn_new.resize(params.R);
      }

      if ((rnn_old.size() > params.R)) {
        std::shuffle(rnn_old.begin(), rnn_old.end(), rng);
        rnn_old.resize(params.R);
      }

      nn_new.insert(nn_new.end(), std::make_move_iterator(rnn_new.begin()),
                    std::make_move_iterator(rnn_new.end()));
      nn_old.insert(nn_old.end(), std::make_move_iterator(rnn_old.begin()),
                    std::make_move_iterator(rnn_old.end()));
    }
  }

 public:
  KGraphConstructor(IndexOracle const &o, IndexParams const &p, IndexInfo *r)
      : oracle(o), params(p), pinfo(r), nhoods(o.size()), n_comps(0) {
    no_dist = true;
    boost::timer::cpu_timer timer;
    // hub_mask.set();
    // params.check();
    const unsigned N = oracle.size();
    // graph_only_ids = std::vector<std::vector<uint32_t>>(N,
    // std::vector<uint32_t>(100, 0));
    if (N <= params.K) throw runtime_error("K larger than dataset size");
    if (N < params.controls) {
      cerr << "Warning: small dataset, shrinking control size to " << N << "."
           << endl;
      params.controls = N;
    }
    if (N <= params.L) {
      cerr << "Warning: small dataset, shrinking L to " << (N - 1) << "."
           << endl;
      params.L = N - 1;
    }
    if (N <= params.S) {
      cerr << "Warning: small dataset, shrinking S to " << (N - 1) << "."
           << endl;
      params.S = N - 1;
    }

    // vector<Control> controls(params.controls);
    // if (verbosity > 0) cerr << "Generating control..." << endl;
    // GenerateControl(oracle, params.controls, params.K, controls);
    if (verbosity > 0) cerr << "Initializing..." << '\n';

    // initialize nhoods
    if (!params.in_graph_path.empty()) {
      std::cout << "use faiss pre-graph" << '\n';
      load_nn_graph(params.in_graph_path.c_str(), params.K);
      prepare_nhoods();
    } else {
      init();
    }

    // iterate until converge
    float total = N * float(N - 1) / 2;
    IndexInfo info;
    info.stop_condition = IndexInfo::ITERATION;
    info.recall = 0;
    info.accuracy = numeric_limits<float>::max();
    // info.cost = 0;
    info.iterations = 0;
    // info.delta = 1.0;
    graph.clear();
    graph.shrink_to_fit();
    std::cout << "begin iteration" << '\n';
    for (unsigned it = 0; (params.iterations <= 0) || (it < params.iterations);
         ++it) {
      ++info.iterations;
      join(info);
      {
        auto times = timer.elapsed();
        if (verbosity > 0) {
          cerr << "iteration: " << info.iterations
               << " time: " << times.wall / 1e9 << '\n';
        }
      }
      if (it < params.iterations) {
        update(info);
      }
    }

    if (params.reverse) {
      reverse(params.reverse);
      std::cout << "Reversing graph..." << '\n';
    }
    if (params.prune) {
      std::cout << "Pruning graph..." << '\n';
      prune(o, params.prune);
    }
    if (pinfo) {
      *pinfo = info;
    }
  }
};

void KGraphImpl::build(IndexOracle const &oracle, IndexParams const &param,
                       IndexInfo *info) {
  KGraphConstructor con(oracle, param, info);
  M.swap(con.M);
  nhoods.swap(con.nhoods);
  // graph_only_ids.swap(con.graph_only_ids);
  std::swap(no_dist, con.no_dist);
}

KGraph *KGraph::create() { return new KGraphImpl; }

char const *KGraph::version() { return kgraph_version; }
}  // namespace kgraph
