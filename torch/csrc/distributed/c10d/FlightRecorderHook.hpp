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
 * FlightRecorderHook records collective operations to the FlightRecorder
 * via pre/post hooks on the ProcessGroup.
 */
class FlightRecorderHook {
 public:
  FlightRecorderHook(
      Backend* pg,
      const std::string& pg_name,
      const std::string& pg_desc,
      std::chrono::milliseconds timeout,
      std::shared_ptr<ProcessGroupStatus> pg_status);

  ~FlightRecorderHook();

  // Disable copy and move
  FlightRecorderHook(const FlightRecorderHook&) = delete;
  FlightRecorderHook(FlightRecorderHook&&) = delete;
  FlightRecorderHook& operator=(const FlightRecorderHook&) = delete;
  FlightRecorderHook& operator=(FlightRecorderHook&&) = delete;

  void onPreHook(const PreHookArgs& args);
  void onPostHook(const PostHookArgs& args);

  int64_t getPreHookId() const { return pre_hook_id_; }
  int64_t getPostHookId() const { return post_hook_id_; }

 private:
  void retireEntry(int64_t op_id, c10::Device device);
  static std::string hookOpNameToString(HookOpName name);
  static c10::Device getDeviceFromArgs(const PreHookArgs& args);

  struct EventPair {
    std::unique_ptr<at::cuda::CUDAEvent> start;
    std::unique_ptr<at::cuda::CUDAEvent> end;
    c10::Device device{c10::DeviceType::CPU, 0};
  };

  FlightRecorder<at::cuda::CUDAEvent>* recorder_;
  Backend* pg_;
  int64_t pre_hook_id_;
  int64_t post_hook_id_;
  size_t pg_id_;
  std::string pg_name_;
  std::string pg_desc_;
  std::chrono::milliseconds timeout_;
  std::shared_ptr<ProcessGroupStatus> pg_status_;
  std::atomic<size_t> collective_seq_id_{0};

  // Stores CUDA events keyed by op_id -- events must outlive the recorder's
  // raw pointers until retirement.
  std::unordered_map<int64_t, EventPair> pending_events_;
  std::unordered_map<int64_t, std::pair<size_t, size_t>> op_id_to_fr_id_;
  std::mutex events_mutex_;
};

} // namespace c10d
