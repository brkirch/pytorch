//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSAllocatorInterface.h>

@interface MPSGraphExecutionDescriptor ()
@property (readwrite, atomic) BOOL enableCommitAndContinue;
@end

namespace at {
namespace mps {

// threshold to perform adaptive commit if the accumulated size
// of resources encoded on the command buffer exceeds that.
static const size_t kCmdBufAdaptiveCommitThreshold = MB(64);

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

MPSStream::MPSStream(Stream stream) : _stream(stream) {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  TORCH_CHECK(_stream.device_type() == DeviceType::MPS);
  _serialQueue = dispatch_queue_create("metal gpu stream", nullptr);
  _executionDescriptor = [MPSGraphExecutionDescriptor new];
  _executableExecutionDescriptor = [MPSGraphExecutableExecutionDescriptor new];
  // disable commitAndContinue if Signpost tracing is enabled
  if (getMPSProfiler().isSignpostTracingEnabled()) {
    _enableCommitAndContinue = false;
  }
  _executionDescriptor.enableCommitAndContinue = _enableCommitAndContinue;
}

MPSStream::~MPSStream() {
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];
  [_executableExecutionDescriptor release];

  _executionDescriptor = nil;
  _executableExecutionDescriptor = nil;

  assert(_commandBuffer == nil);
}

MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }

  return _commandBuffer;
}

id<MTLComputeCommandEncoder> MPSStream::commandEncoder() {
  if (!_commandEncoder) {
    _commandEncoder = [commandBuffer() computeCommandEncoder].retain;
  }

  return _commandEncoder;
}

void MPSStream::synchronize(SyncType syncType) {
  endKernelCoalescing();
  switch(syncType) {
    case SyncType::NONE:
      // typically in GPU to GPU copies we won't commit explicitly
      break;
    case SyncType::COMMIT:
      commit();
      break;
    case SyncType::COMMIT_ADAPTIVE:
      // the adaptive commit only commits if we hit the low watermark memory threshold,
      // or when the sizes attached to the active command buffer exceeds the threshold
      if (getIMPSAllocator()->getLowWatermarkValue() <= 0 ||
          _commandBufferResourceSize > kCmdBufAdaptiveCommitThreshold) {
        commit();
      }
      break;
    case SyncType::COMMIT_AND_WAIT:
      commitAndWait();
      break;
    case SyncType::COMMIT_AND_CONTINUE:
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_enableCommitAndContinue,
                                       "CommitAndContinue is called but it is disabled globally!");
      commitAndContinue();
      break;
  }
}

void MPSStream::commit() {
  if (_enableCommitAndContinue) {
    [commandBuffer() commitAndContinue];
  } else {
    flush();
  }
  // reset the accumulated resource sizes for command buffer
  _commandBufferResourceSize = 0;
}

void MPSStream::commitAndWait() {
  if (_prevCommandBuffer) {
    // the previous command buffer (if exists) has already been committed,
    // so we just wait until it's completed and then dispose it.
    [_prevCommandBuffer waitUntilCompleted];
    [_prevCommandBuffer release];
    _prevCommandBuffer = nil;
  }

  if (_commandBuffer) {
    [_commandBuffer commit];
    [_commandBuffer waitUntilCompleted];
    [_commandBuffer release];
    _commandBuffer = nil;
    // reset the accumulated resource sizes for command buffer
    _commandBufferResourceSize = 0;
  }
}

void MPSStream::commitAndContinue() {
  assert(_commandBuffer);
  [_commandBuffer commitAndContinue];
}

void MPSStream::commitAdaptive(const TensorList& tensors, void* profilerHandle) {
  if (_enableCommitAndContinue) {
    for (const auto& tensor : tensors) {
      _commandBufferResourceSize += tensor.nbytes();
    }
  }
  auto& profiler = getMPSProfiler();
  if (profiler.isOperationProfilingEnabled()) {
    profiler.endProfileKernel(profilerHandle, SyncType::COMMIT_ADAPTIVE);
  } else {
    synchronize(SyncType::COMMIT_ADAPTIVE);
  }
}

void MPSStream::endKernelCoalescing() {
  if (_commandEncoder) {
    [_commandEncoder endEncoding];
    [_commandEncoder release];
    _commandEncoder = nil;
  }
}

void MPSStream::flush() {
  if (_commandBuffer) {
    [_commandBuffer commit];
    // if commitAndContinue is disabled (e.g., for Profiler), we keep the command
    // buffer so we could wait on it later, if required.
    if (!_enableCommitAndContinue) {
      TORCH_INTERNAL_ASSERT(getMPSProfiler().isSignpostTracingEnabled());
      _prevCommandBuffer = _commandBuffer;
    } else {
      [_commandBuffer release];
    }
    _commandBuffer = nil;
  }
}

void MPSStream::addCompletedHandler(MTLCommandBufferHandler block) {
 dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      [commandBuffer() addCompletedHandler:block];
    }
  });
}

void MPSStream::fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType)
{
  TORCH_INTERNAL_ASSERT(length >= offset);
  if (length == 0) return;
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      [blitEncoder fillBuffer:buffer
                        range:NSMakeRange(offset, length)
                        value:value];
      [blitEncoder endEncoding];
      synchronize(syncType);
    }
  });
}

void MPSStream::copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
                    size_t length, size_t srcOffset, size_t dstOffset,
                    uint64_t profileId, SyncType syncType) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      [blitEncoder copyFromBuffer:srcBuffer
                     sourceOffset:(NSUInteger)srcOffset
                         toBuffer:dstBuffer
                destinationOffset:(NSUInteger)dstOffset
                             size:(NSUInteger)length];
      [blitEncoder endEncoding];

      auto& profiler = getMPSProfiler();
      // check if copy profiling is enabled
      if (profiler.isCopyProfilingEnabled()) {
        profiler.endProfileCopy(profileId, syncType);
      } else {
        synchronize(syncType);
      }
    }
  });
}

void MPSStream::copy_and_sync(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer, size_t length,
                              size_t srcOffset, size_t dstOffset, bool non_blocking, uint64_t profileId) {
  copy(srcBuffer, dstBuffer, length, srcOffset, dstOffset, profileId,
       !non_blocking ? SyncType::COMMIT_AND_WAIT : SyncType::COMMIT);
}

void MPSStream::executeMPSGraph(MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results,
                                SyncType syncType, MPSGraphExecutable* executable) {
  auto& profiler = getMPSProfiler();
  const bool isGraphProfilingEnabled = profiler.isOperationProfilingEnabled();

  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      if (isGraphProfilingEnabled) {
        // this function call is only relevant for interval-based Signposts
        // which exclude schedule time (only includes GPU run time)
        profiler.beginProfileGPUInterval(mpsGraph);
      }

      if (executable) {
        NSMutableArray *inputsArray  = [NSMutableArray arrayWithCapacity:[feeds count]];
        NSMutableArray *resultsArray = [NSMutableArray arrayWithCapacity:[results count]];
        NSUInteger inputIndex = 0, ouputIndex = 0;
        for (MPSGraphTensor *tensor in [executable feedTensors]) {
          inputsArray[inputIndex++] = feeds[tensor];
        }

        for (MPSGraphTensor *tensor in [executable targetTensors]) {
          resultsArray[ouputIndex++] = results[tensor];
        }

        [executable encodeToCommandBuffer:commandBuffer()
                              inputsArray:inputsArray
                             resultsArray:resultsArray
                      executionDescriptor:_executableExecutionDescriptor];
      } else {
        // note: CommitAndContinue feature is enabled/disabled via "_executionDescriptor"
        [mpsGraph encodeToCommandBuffer:commandBuffer()
                                  feeds:feeds
                       targetOperations:nil
                      resultsDictionary:results
                    executionDescriptor:_executionDescriptor];
      }

      updateCommandBufferResourceSize([feeds allValues]);
      // if commitAndContinue is disabled, we need to always commit manually after encoding
      SyncType _syncType = _enableCommitAndContinue == false ? SyncType::COMMIT : syncType;

      // check if graph execution profiling is enabled
      if (isGraphProfilingEnabled) {
        // with profiler enabled, we commit after adding the completedHandler in MPSProfiler
        profiler.endProfileKernel(mpsGraph, _syncType);
      } else {
        synchronize(_syncType);
      }
    }
 });
}

void MPSStream::updateCommandBufferResourceSize(NSArray<MPSGraphTensorData*> *feeds) {
  if (_enableCommitAndContinue) {
    for (MPSGraphTensorData* tensorData in feeds) {
      size_t resource_size = tensorData.mpsndarray.resourceSize;
      _commandBufferResourceSize += resource_size;
    }
  }
}

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

MPSStream* MPSStreamImpl::_stream = nullptr;

MPSStream* MPSStreamImpl::getInstance() {
  if (_stream == nullptr) {
    _stream =
        new MPSStream(Stream(Stream::UNSAFE, c10::Device(DeviceType::MPS), 0));
  }
  return _stream;
}

MPSStreamImpl::MPSStreamImpl() {}

MPSStream* getCurrentMPSStream() {
  return getDefaultMPSStream();
}

MPSStream* getDefaultMPSStream() {
  return MPSStreamImpl::getInstance();
}

//-----------------------------------------------------------------
//  MPSEvent
//-----------------------------------------------------------------

MPSEvent::MPSEvent(bool deferInitialization) :
    is_initialized(false), _signalCounter(0), _stream(nil), _event(nil), _listener(nil) {
  if (!deferInitialization) {
    initialize();
  }
}

MPSEvent::~MPSEvent() {
  if (_event) {
    [_event release];
    _event = nil;
  }
  if (_listener) {
    [_listener release];
    _listener = nil;
  }
}

void MPSEvent::initialize() {
  _stream = getDefaultMPSStream();
  _event = [_stream->device() newSharedEvent];
  _listener = [[MTLSharedEventListener alloc] init];
  is_initialized = true;
}

void MPSEvent::recordEvent(bool syncEvent) {
  if (!is_initialized)
    initialize();

  dispatch_sync(_stream->queue(), ^() {
    @autoreleasepool {
      ++_signalCounter;
      id<MTLCommandBuffer> commandBuffer = _stream->commandBuffer();
      [commandBuffer encodeSignalEvent:_event value:_signalCounter];
      if (syncEvent)
        _stream->synchronize(SyncType::COMMIT);
    }
  });
}

void MPSEvent::waitForEvent(bool syncEvent) {
  TORCH_INTERNAL_ASSERT(is_initialized);
  dispatch_sync(_stream->queue(), ^() {
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = _stream->commandBuffer();
      [commandBuffer encodeWaitForEvent:_event value:_signalCounter];
      if (syncEvent)
        _stream->synchronize(SyncType::COMMIT);
    }
  });
}

void MPSEvent::notifyEvent(MTLSharedEventNotificationBlock block)
{
  if (!is_initialized)
    initialize();
  dispatch_sync(_stream->queue(), ^() {
    @autoreleasepool {
      ++_signalCounter;
      [_event notifyListener:_listener atValue:_signalCounter block:block];
    }
  });
}

bool MPSEvent::queryEvent() const {
  // return false if not recorded or signaled yet
  return _signalCounter && (_event.signaledValue >= _signalCounter);
}

} // namespace mps
} // namespace at
