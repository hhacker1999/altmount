package usenet

import (
	"context"
	"io"
	"testing"
	"time"

	"github.com/javi11/altmount/internal/pool"
	"github.com/javi11/altmount/internal/testsupport/fakepool"
	"github.com/javi11/altmount/internal/testsupport/segments"
)

// TestAggressivePrefetch_MaintainsInflightUnderSlowRead pins the saturation
// contract: with aggressive mode and a slow reader, downloadManager must keep
// MaxInflight concurrent BodyPriority calls in flight while filling toward the
// high watermark — not collapse to ~1 after a small segment window completes.
func TestAggressivePrefetch_MaintainsInflightUnderSlowRead(t *testing.T) {
	t.Parallel()
	const (
		segCount    = 40
		segSize     = 64 * 1024 // 64 KiB → ~2.5 MB total
		maxPrefetch = 10        // legacy window alone would be tiny
		maxInflight = 8
		segLatency  = 50 * time.Millisecond
	)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	fp := fakepool.New()
	for i := 0; i < segCount; i++ {
		fp.SetBehavior(segments.MessageID(i), fakepool.SegmentBehavior{
			Latency: segLatency,
			Bytes:   segments.Payload(i, segSize),
		})
	}

	rg := buildEagerRange(ctx, t, segCount, segSize)
	getter := func() (pool.NntpClient, error) { return fp, nil }

	ur, err := NewUsenetReader(ctx, getter, rg, maxPrefetch, noopMetrics{}, "agg-test", nil,
		WithPrefetchConfig(PrefetchConfig{
			MaxPrefetch:        maxPrefetch,
			Aggressive:         true,
			HighWatermarkBytes: 2 * 1024 * 1024, // 2 MB
			LowWatermarkBytes:  512 * 1024,
			MaxInflight:        maxInflight,
		}),
	)
	if err != nil {
		t.Fatalf("NewUsenetReader: %v", err)
	}
	t.Cleanup(func() { _ = ur.Close() })
	ur.Start()

	// Slow consumer: read one small chunk at a time with delay so the
	// download manager has time to keep the inflight window full.
	buf := make([]byte, 4*1024)
	for {
		n, readErr := ur.Read(buf)
		if n > 0 {
			time.Sleep(20 * time.Millisecond)
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			t.Fatalf("Read: %v", readErr)
		}
	}

	got := fp.MaxInFlight()
	if got < int32(maxInflight) {
		t.Fatalf("MaxInFlight high-water = %d, want >= %d (aggressive mode must sustain inflight while filling watermark)",
			got, maxInflight)
	}
}
