#pragma once

// STL includes
#include <cstddef>
#include <limits>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Start values for a scan point, taken from the nearest point already fitted.
 *
 * Every point of a scan is the same fit with one or two parameters moved a
 * little, so the converged nuisance parameters of a neighbouring point are a far
 * better starting guess than the configured defaults. Minuit then spends its
 * iterations polishing instead of travelling, which is where most of the cost of
 * a scan point sits once analytic gradients make each iteration cheap.
 *
 * Nothing here decides where a fit starts on its own: the store only remembers
 * results and answers "which known point is closest". The caller picks the
 * distance and applies the vector to whichever parameters are still free, so the
 * scanned parameter -- fixed to the node position -- is never overwritten.
 *
 * Only results the caller considers trustworthy should be handed to store():
 * seeding from a fit that stalled propagates its bad point to its neighbours,
 * and from there outwards through the rest of the scan.
 */
namespace scanseed {

  /**
   * @brief Converged parameter vectors of a scan, keyed by lattice position.
   *
   * Shared by the scan workers, so every method is internally synchronised. The
   * vectors are copied out rather than referenced: a worker holds its start
   * values for the whole of its fit, during which other workers keep inserting.
   *
   * @tparam Key Lattice coordinate of a scan point (an index in 1D, a node in 2D).
   */
  template <typename Key>
  class Store {
   public:
    /// Start values used when no point has been fitted yet, normally the free
    /// fit that opens the scan. Empty means "fall back to the configured values".
    void set_fallback(std::vector<double> values) {
      const std::scoped_lock lock(m_Mutex);
      m_Fallback = std::move(values);
    }

    /// Remembers the result of a scan point. Only converged, finite fits belong here.
    void store(const Key& key, std::vector<double> values) {
      const std::scoped_lock lock(m_Mutex);
      m_Values[key] = std::move(values);
    }

    /**
     * @brief Start values for a point about to be fitted.
     *
     * @param key      Position of the point.
     * @param distance Callable `double(const Key&, const Key&)` giving the
     *                 distance between two points in whatever metric the scan
     *                 measures its window in.
     * @return The vector of the nearest known point, the fallback if there is
     *         none, or an empty vector if there is no fallback either.
     */
    template <typename Distance>
    [[nodiscard]] std::vector<double> nearest(const Key& key, Distance&& distance) const {
      const std::scoped_lock lock(m_Mutex);

      const std::vector<double>* best         = nullptr;
      double                     best_distance = std::numeric_limits<double>::infinity();

      for (const auto& [known, values] : m_Values) {
        const double d = distance(known, key);
        if (d < best_distance) {
          best_distance = d;
          best          = &values;
        }
      }

      return best != nullptr ? *best : m_Fallback;
    }

    /// Number of points remembered. For reporting.
    [[nodiscard]] std::size_t size() const {
      const std::scoped_lock lock(m_Mutex);
      return m_Values.size();
    }

   private:
    mutable std::mutex          m_Mutex;
    std::map<Key, std::vector<double>> m_Values;
    std::vector<double>         m_Fallback;
  };

}  // namespace scanseed