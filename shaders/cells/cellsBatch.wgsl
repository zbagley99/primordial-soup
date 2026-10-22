// Integer sprite batches shared by the Cells planning, sprite and average passes.
// Each sum stays below 2^32 because a pass never lets more sprites cover one
// pixel than its quantum was sized for (cells.rs `quantum`, `sample`).
#include "cellsCommon.wgsl"
// Adjacent words so one fragment's four additions land in one memory sector.
struct CellAccum { red:atomic<u32>, green:atomic<u32>, blue:atomic<u32>, mass:atomic<u32> }
// Written once per frame on the CPU.
struct BatchSettings {
 limit:u32,     // sprites per pass that the reference quantum was sized for
 batches:u32,   // particle batches encoded per sample; always odd
 capacity:u32,  // listed sprite copies that fit the compact list
 copies:u32,    // 9 when wrapping, otherwise 1
 count:u32,     // allocated particles
 size:u32,      // particles per fallback batch
 slots:u32,     // offset of the dispatch arguments in the indirect buffer
 _pad:u32,
}
// Read-only view of the plan the GPU writes before the sprite passes.
struct BatchPlan { bound:u32, entries:u32, mode:u32, _pad:u32 }
// Listed batches retain the original particle ranges and integer conversion boundaries.
const COMPACT:u32=0u;
const LISTED_BATCHES:u32=2u;
