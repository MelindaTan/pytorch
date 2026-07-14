
#include <torch/csrc/distributed/c10d/FlightRecorderHook.hpp>
#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/thread_name.h>


namespace c10d {

namespace {

std::atomic<int64_t> g_hook_id_counter{1};
std::atomic<size_t> g_pg_id_counter{0};

int64_t nextHookId() {
  return g_hook_id_counter.fetch_add(1, std::memory_order_relaxed);
}

size_t nextPgId() {
  return g_pg_id_counter.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

FlightRecorderHook::FlightRecorderHook(
    Backend* pg,
    const std::string& pg_name,
    const std::string& pg_desc,
    std::chrono::milliseconds timeout,
    std::shared_ptr<ProcessGroupStatus> pg_status)
    : recorder_(FlightRecorder<at::cuda::CUDAEvent>::get()),
      pg_(pg),
      pre_hook_id_(nextHookId()),
      post_hook_id_(nextHookId()),
      pg_id_(nextPgId()),
      pg_name_(pg_name),
      pg_desc_(pg_desc),
      timeout_(timeout),
      pg_status_(std::move(pg_status)) {
  LOG(INFO) << "FlightRecorderHook created for PG " << pg_name_;
}

FlightRecorderHook::~FlightRecorderHook() {
  LOG(INFO) << "FlightRecorderHook destroyed for PG " << pg_name_;
}

std::string FlightRecorderHook::hookOpNameToString(HookOpName name) {
  switch (name) {
    case HookOpName::SEND: return "send";
    case HookOpName::RECV: return "recv";
    case HookOpName::BROADCAST: return "broadcast";
    case HookOpName::ALLREDUCE: return "allreduce";
    case HookOpName::REDUCE: return "reduce";
    case HookOpName::ALLGATHER: return "allgather";
    case HookOpName::REDUCE_SCATTER: return "reduce_scatter";
    case HookOpName::ALLTOALL: return "alltoall";
    case HookOpName::BARRIER: return "barrier";
    case HookOpName::SCATTER: return "scatter";
    case HookOpName::GATHER: return "gather";
    case HookOpName::SPLIT: return "split";
    case HookOpName::NEW_WINDOW: return "new_window";
    default: return "unknown";
  }
}

c10::Device FlightRecorderHook::getDeviceFromArgs(const PreHookArgs& args) {
  if (!args.input_tensors.empty()) {
    return args.input_tensors[0].device();
  }
  if (!args.output_tensors.empty()) {
    return args.output_tensors[0].device();
  }
  return c10::Device(c10::DeviceType::CPU, 0);
}

void FlightRecorderHook::onPreHook(const PreHookArgs& args) {
  c10::Device device = getDeviceFromArgs(args);
  std::string profiling_name = pg_desc_ + ":" + hookOpNameToString(args.name);

  at::cuda::CUDAEvent* start_event_ptr = nullptr;
  at::cuda::CUDAEvent* end_event_ptr = nullptr;

  if (device.type() == c10::DeviceType::CUDA) {
    try {
      auto start_event = std::make_unique<at::cuda::CUDAEvent>(cudaEventDefault);
      auto end_event = std::make_unique<at::cuda::CUDAEvent>(cudaEventDefault);

      start_event->record(at::cuda::getCurrentCUDAStream(device.index()));

      start_event_ptr = start_event.get();
      end_event_ptr = end_event.get();

      {
        std::lock_guard<std::mutex> lock(events_mutex_);
        pending_events_[args.op_id] = EventPair{
            std::move(start_event),
            std::move(end_event),
            device
        };
      }

      VLOG(2) << "Created CUDA events for op_id=" << args.op_id;
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to create CUDA events: " << e.what();
    }
  }

  size_t collective_seq = collective_seq_id_.fetch_add(1, std::memory_order_relaxed);

  auto fr_id = recorder_->recordWithResetEnabled(
      pg_id_,
      std::make_tuple(pg_name_, pg_desc_),
      collective_seq,
      0,
      static_cast<size_t>(args.op_id),
      std::move(profiling_name),
      args.input_tensors,
      args.output_tensors,
      start_event_ptr,
      end_event_ptr,
      timeout_,
      pg_status_,
      false);

  if (fr_id.id.has_value() && fr_id.reset_epoch.has_value()) {
    std::lock_guard<std::mutex> lock(events_mutex_);
    op_id_to_fr_id_[args.op_id] = std::make_pair(*fr_id.id, *fr_id.reset_epoch);
  }
}

void FlightRecorderHook::onPostHook(const PostHookArgs& args) {
  c10::Device device(c10::DeviceType::CPU, 0);
  {
    std::lock_guard<std::mutex> lock(events_mutex_);
    auto it = pending_events_.find(args.op_id);
    if (it != pending_events_.end()) {
      device = it->second.device;
    }
  }

  if (args.work) {
    // Async - retire when work completes
    args.work->getFuture()->addCallback(
        [this, op_id = args.op_id, device](c10::ivalue::Future& /* unused */) {
          retireEntry(op_id, device);
        }
    );
  } else {
    // Sync - retire immediately
    retireEntry(args.op_id, device);
  }
}

// Move events out of maps under lock before recording -- prevents
// use-after-free when the circular buffer wraps.
void FlightRecorderHook::retireEntry(int64_t op_id, c10::Device device) {
  EventPair events;
  std::optional<size_t> fr_id;
  std::optional<size_t> reset_epoch;

  {
    std::lock_guard<std::mutex> lock(events_mutex_);

    auto id_it = op_id_to_fr_id_.find(op_id);
    if (id_it != op_id_to_fr_id_.end()) {
      fr_id = id_it->second.first;
      reset_epoch = id_it->second.second;
      op_id_to_fr_id_.erase(id_it);
    }

    auto events_it = pending_events_.find(op_id);
    if (events_it != pending_events_.end()) {
      events = std::move(events_it->second);
      pending_events_.erase(events_it);
    }
  }

  if (events.end && device.type() == c10::DeviceType::CUDA) {
    try {
      events.end->record(at::cuda::getCurrentCUDAStream(device.index()));
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to record end event: " << e.what();
      events.start.reset();
      events.end.reset();
    }
  }

  if (fr_id.has_value() && reset_epoch.has_value()) {
    recorder_->retire_id(fr_id, reset_epoch, /*compute_duration=*/true);
  }
}

} // namespace c10d
