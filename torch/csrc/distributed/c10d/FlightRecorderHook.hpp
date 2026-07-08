#pragma once

#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>
#include <torch/csrc/distributed/c10d/Hooks.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/DeviceGuard.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace c10d {

/**
 * FlightRecorderHook is a hook listener that records collective operations
 * to the FlightRecorder via pre/post hooks.
 *
 * This follows the same pattern as torchcomms' FlightRecorderHook but adapted
 * for c10d's hook system.
 *
 * Usage:
 *   auto fr_hook = std::make_unique<FlightRecorderHook>(
 *       process_group, pg_id, pg_name, pg_desc, timeout, pg_status);
 *   // Hooks are automatically registered in constructor
 */
class FlightRecorderHook {
 public:
  /**
   * Create a FlightRecorderHook and register it with the ProcessGroup.
   *
   * @param pg ProcessGroup to attach to
   * @param pg_id Process group ID (typically from reinterpret_cast<uint64_t>(this))
   * @param pg_name Process group name
   * @param pg_desc Process group description (backend name)
   * @param timeout Default timeout for operations
   * @param pg_status Shared status object for tracking (can be nullptr)
   */
  FlightRecorderHook(
      Backend* pg,
      size_t pg_id,
      const std::string& pg_name,
      const std::string& pg_desc,
      std::chrono::milliseconds timeout,
      std::shared_ptr<ProcessGroupStatus> pg_status);

  /**
   * Destructor - unregisters hooks from ProcessGroup
   */
  ~FlightRecorderHook();

  // Disable copy and move
  FlightRecorderHook(const FlightRecorderHook&) = delete;
  FlightRecorderHook(FlightRecorderHook&&) = delete;
  FlightRecorderHook& operator=(const FlightRecorderHook&) = delete;
  FlightRecorderHook& operator=(FlightRecorderHook&&) = delete;

  /**
   * Pre-hook callback - records operation to FlightRecorder.
   * Called before collective is issued.
   */
  void onPreHook(const PreHookArgs& args);

  /**
   * Post-hook callback - registers retirement callback.
   * Called after collective is issued (but before completion).
   */
  void onPostHook(const PostHookArgs& args);

  /**
   * Get the pre-hook ID (for registration)
   */
  int64_t getPreHookId() const {
    return pre_hook_id_;
  }

  /**
   * Get the post-hook ID (for registration)
   */
  int64_t getPostHookId() const {
    return post_hook_id_;
  }

 private:
  /**
   * Retire entry when work completes.
   * Called from work completion callback.
   */
  void retireEntry(int64_t op_id, c10::Device device);

  /**
   * Convert HookOpName enum to string for profiling_name
   */
  static std::string hookOpNameToString(HookOpName name);

  /**
   * Extract device from hook arguments
   */
  static c10::Device getDeviceFromArgs(const PreHookArgs& args);

  // Was: P2P helper (c10d-specific, not in torchcomms)
  // static bool isP2POperation(HookOpName name);

  // CARGO-CULTED FROM TORCHCOMMS: Event storage for CUDA timing
  // See torchcomms/comms/torchcomms/hooks/fr/FlightRecorder.hpp:254-257
  struct EventPair {
    std::unique_ptr<at::cuda::CUDAEvent> start;
    std::unique_ptr<at::cuda::CUDAEvent> end;
    c10::Device device{c10::DeviceType::CPU, 0};
  };

  // FlightRecorder instance (global singleton)
  FlightRecorder<at::cuda::CUDAEvent>* recorder_;

  // Backend we're attached to
  Backend* pg_;

  // Hook IDs for cleanup
  int64_t pre_hook_id_;
  int64_t post_hook_id_;

  // State captured from ProcessGroup at construction
  size_t pg_id_;
  std::string pg_name_;
  std::string pg_desc_;
  std::chrono::milliseconds timeout_;
  std::shared_ptr<ProcessGroupStatus> pg_status_;

  // CARGO-CULTED FROM TORCHCOMMS: Sequence counter
  // See torchcomms/comms/torchcomms/hooks/fr/FlightRecorder.cpp:252
  // p2p_seq_id hardcoded to 0, matching torchcomms
  std::atomic<size_t> collective_seq_id_{0};
  // Was: separate P2P counter (c10d-specific, not in torchcomms)
  // std::atomic<size_t> p2p_seq_id_{0};

  // CARGO-CULTED FROM TORCHCOMMS: Event storage
  // See torchcomms/comms/torchcomms/hooks/fr/FlightRecorder.hpp:355
  // Stores CUDA events keyed by op_id - critical for event lifetime
  std::unordered_map<int64_t, EventPair> pending_events_;

  // CARGO-CULTED FROM TORCHCOMMS: Map op_id to FR internal (id, reset_epoch)
  // See torchcomms/comms/torchcomms/hooks/fr/FlightRecorder.cpp:304
  std::unordered_map<int64_t, std::pair<size_t, size_t>> op_id_to_fr_id_;

  // Mutex protecting both maps above
  std::mutex events_mutex_;
};

} // namespace c10d
