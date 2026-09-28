#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "neural/encoder.h"
#include "neural/factory.h"
#include "neural/loader.h"
#include "neural/network.h"
#include "utils/optionsdict.h"

struct SparseSample {
  std::array<float, 1858> policy;
  std::array<float, 3> value_wdl;
};

extern "C" {
void* chess_model_load(const char* path);
void chess_model_free(void* model);
void chess_model_evaluate(void* model, const std::uint16_t* input_ptr,
                          std::size_t input_len, const bool* legality_mask_ptr,
                          std::size_t mask_len, float* out_policy,
                          float* out_value_wdl);
}

extern const char* kMoveStrs[];

namespace lczero {
namespace {

constexpr std::array<bool, 1858> kDefaultLegalityMask = []() {
  std::array<bool, 1858> mask{};
  mask.fill(true);
  return mask;
}();

constexpr std::array<uint16_t, 12> kPlaneToCell = {
    7, 8, 9, 10, 11, 12,  // Us: P, N, B, R, Q, K
    1, 2, 3, 4,  5,  6    // Them: P, N, B, R, Q, K
};

class SparseNetworkComputation : public NetworkComputation {
 public:
  explicit SparseNetworkComputation(void* model_handle) : model_(model_handle) {
    batch_.reserve(1);
    inputs_.reserve(1);
  }

  void AddInput(InputPlanes&& input) override {
    inputs_.push_back(std::move(input));
  }

  void ComputeBlocking() override {
    batch_.resize(inputs_.size());

    for (size_t i = 0; i < inputs_.size(); i++) {
      std::array<std::uint16_t, 72> csdr{};
      PlanesToCsdr(inputs_[i], csdr);

      chess_model_evaluate(model_, csdr.data(), csdr.size(),
                           kDefaultLegalityMask.data(),
                           kDefaultLegalityMask.size(), batch_[i].policy.data(),
                           batch_[i].value_wdl.data());
    }
  }

  int GetBatchSize() const override { return static_cast<int>(batch_.size()); }

  float GetQVal(int sample) const override {
    float w = batch_[sample].value_wdl[0];
    float l = batch_[sample].value_wdl[2];

    return w - l;
  }

  float GetDVal(int sample) const override {
    return batch_[sample].value_wdl[1];
  }

  float GetPVal(int sample, int move_id) const override {
    return batch_[sample].policy[move_id];
  }

  float GetMVal(int /*sample*/) const override { return 0.0f; }

 private:
  void* model_ = nullptr;
  std::vector<InputPlanes> inputs_{};
  std::vector<SparseSample> batch_{};

  static void PlanesToCsdr(const InputPlanes& planes,
                           std::array<uint16_t, 72>& out) {
    out.fill(0);

    for (size_t p = 0; p < kPlaneToCell.size(); p++) {
      uint64_t mask = planes[p].mask;
      const uint16_t piece_val = kPlaneToCell[p];

      while (mask != 0) {
        const int sq = std::countr_zero(mask);
        const int rank = sq / 8;
        const int file = sq % 8;
        out[rank * 8 + file] = piece_val;
        mask &= mask - 1;
      }
    }

    constexpr size_t META_BASE = 64;

    const uint16_t friendly_castle_qs = planes[kAuxPlaneBase + 0].mask != 0;
    const uint16_t friendly_castle_ks = planes[kAuxPlaneBase + 1].mask != 0;
    const uint16_t enemy_castle_qs = planes[kAuxPlaneBase + 2].mask != 0;
    const uint16_t enemy_castle_ks = planes[kAuxPlaneBase + 3].mask != 0;
    const uint16_t turn = planes[kAuxPlaneBase + 4].mask != 0;
    const float rule50_count = planes[kAuxPlaneBase + 5].value;

    out[META_BASE + 0] = friendly_castle_qs;
    out[META_BASE + 1] = friendly_castle_ks;
    out[META_BASE + 2] = enemy_castle_qs;
    out[META_BASE + 3] = enemy_castle_ks;
    out[META_BASE + 4] = turn;
    out[META_BASE + 5] = static_cast<uint16_t>(rule50_count * 13.0 / 150.0);
  }
};

class SparseNetwork : public Network {
 public:
  explicit SparseNetwork(const OptionsDict& options) {
    const auto model_path = options.Get<std::string>("model_path");
    model_ = chess_model_load(model_path.c_str());
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    return std::make_unique<SparseNetworkComputation>(model_);
  }

  bool IsCpu() const override { return true; }
  int GetMiniBatchSize() const override { return 1; }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  ~SparseNetwork() override { chess_model_free(model_); }

 private:
  NetworkCapabilities capabilities_{
      pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE,
      pblczero::NetworkFormat::OUTPUT_WDL,
      pblczero::NetworkFormat::MOVES_LEFT_NONE};

  void* model_;
};

std::unique_ptr<Network> MakeSparseNetwork(
    const std::optional<WeightsFile>& /*weights*/, const OptionsDict& options) {
  return std::make_unique<SparseNetwork>(options);
}

REGISTER_NETWORK("sparse", MakeSparseNetwork, 120)

}  // namespace
}  // namespace lczero
