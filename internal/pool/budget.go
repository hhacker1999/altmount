package pool

import (
	"context"
	"sync"
	"sync/atomic"
)

// streamHeadroom is how many connections are set aside per active stream when
// shrinking the import budget. Deliberately a constant, not config: the whole
// point of the budget is automatic balancing without knobs. Streams get hard
// priority via the pool's priority request lane regardless; the headroom just
// keeps free connections available so a stream never waits for an import
// article to finish.
const streamHeadroom = 2

// ImportBudget bounds the total number of in-flight import segment (body)
// fetches pool-wide, across all concurrent imports. Its capacity tracks the
// pool's total connection count and automatically shrinks while streams are
// active:
//
//	effective cap = capacity − min(streamHeadroom × activeStreams, capacity−1)
//
// so imports expand to the full pool when idle, yield headroom to streams
// under playback, and always keep at least 1 connection so a lone import can
// make progress. A capacity of 0 disables the budget (no-op), which keeps
// pool-less paths and test fakes deadlock-free.
//
// When pauseImportsWhileStreaming is set and any stream is active, Acquire
// blocks (separate from the semaphore) until streams finish — semaphore cap 0
// means "unlimited", so pause cannot reuse that sentinel.
type ImportBudget struct {
	sem          adaptiveSemaphore
	capacity     int
	streamSource StreamActivitySource
	// pauseImportsWhileStreaming, when true, blocks import Acquires while
	// ActiveStreams() > 0 so playback can saturate the pool.
	pauseImportsWhileStreaming atomic.Bool

	// pauseMu / pauseCond wake Acquires waiting on the pause gate.
	pauseMu   sync.Mutex
	pauseCond *sync.Cond
}

// NewImportBudget constructs a budget with capacity 0 (disabled). Use
// SetCapacity and SetStreamSource to configure it.
func NewImportBudget() *ImportBudget {
	b := &ImportBudget{}
	b.sem.capLocked = b.effectiveCapLocked
	b.pauseCond = sync.NewCond(&b.pauseMu)
	return b
}

// effectiveCapLocked computes the current cap. Called with sem.mu held.
func (b *ImportBudget) effectiveCapLocked() int {
	if b.capacity <= 0 {
		return 0 // disabled
	}
	reserve := 0
	if b.streamSource != nil {
		reserve = streamHeadroom * b.streamSource.ActiveStreams()
	}
	if reserve > b.capacity-1 {
		reserve = b.capacity - 1
	}
	return b.capacity - reserve
}

// SetPauseImportsWhileStreaming toggles whether import segment fetches are
// fully paused while any stream is active.
func (b *ImportBudget) SetPauseImportsWhileStreaming(pause bool) {
	b.pauseImportsWhileStreaming.Store(pause)
	b.pauseMu.Lock()
	b.pauseCond.Broadcast()
	b.pauseMu.Unlock()
	b.sem.mu.Lock()
	b.sem.wakeWaitersLocked()
	b.sem.mu.Unlock()
}

// SetCapacity updates the total connection capacity (sum of provider
// connections). Queued waiters are woken if the effective cap grew; on shrink,
// in-flight fetches drain naturally.
func (b *ImportBudget) SetCapacity(totalConns int) {
	if totalConns < 0 {
		totalConns = 0
	}
	b.sem.mu.Lock()
	b.capacity = totalConns
	b.sem.wakeWaitersLocked()
	b.sem.mu.Unlock()
}

// Capacity returns the configured total capacity (not the stream-adjusted
// effective cap). Useful for sizing worker pools.
func (b *ImportBudget) Capacity() int {
	b.sem.mu.Lock()
	defer b.sem.mu.Unlock()
	return b.capacity
}

// SetStreamSource wires the activity signal. nil sources are tolerated and
// pin the effective cap to the full capacity.
func (b *ImportBudget) SetStreamSource(src StreamActivitySource) {
	b.sem.mu.Lock()
	b.streamSource = src
	b.sem.wakeWaitersLocked()
	b.sem.mu.Unlock()
	b.pauseMu.Lock()
	b.pauseCond.Broadcast()
	b.pauseMu.Unlock()
}

// NotifyStreamChange should be called when the stream count changes so the
// budget can wake or hold waiters according to the new effective cap.
func (b *ImportBudget) NotifyStreamChange() {
	b.sem.mu.Lock()
	b.sem.wakeWaitersLocked()
	b.sem.mu.Unlock()
	b.pauseMu.Lock()
	b.pauseCond.Broadcast()
	b.pauseMu.Unlock()
}

// importsPaused reports whether Acquire should block on the pause gate.
func (b *ImportBudget) importsPaused() bool {
	if !b.pauseImportsWhileStreaming.Load() {
		return false
	}
	if b.streamSource == nil {
		return false
	}
	return b.streamSource.ActiveStreams() > 0
}

// Acquire blocks until a connection token is available or ctx is cancelled.
// The returned release function MUST be called exactly once when the fetch is
// done. When the capacity is 0 the call is a fast-path no-op.
func (b *ImportBudget) Acquire(ctx context.Context) (release func(), err error) {
	if err := b.waitWhileImportsPaused(ctx); err != nil {
		return noopRelease, err
	}
	return b.sem.Acquire(ctx)
}

// waitWhileImportsPaused blocks while pause-imports is enabled and streams
// are active. Uses a condvar woken by NotifyStreamChange / SetPause*.
func (b *ImportBudget) waitWhileImportsPaused(ctx context.Context) error {
	if !b.importsPaused() {
		return nil
	}

	// Wake the cond when ctx is cancelled.
	done := make(chan struct{})
	defer close(done)
	go func() {
		select {
		case <-ctx.Done():
			b.pauseMu.Lock()
			b.pauseCond.Broadcast()
			b.pauseMu.Unlock()
		case <-done:
		}
	}()

	b.pauseMu.Lock()
	defer b.pauseMu.Unlock()
	for b.importsPaused() {
		if err := ctx.Err(); err != nil {
			return err
		}
		b.pauseCond.Wait()
	}
	return ctx.Err()
}
