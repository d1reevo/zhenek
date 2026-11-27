//! Placed-view shapes contain only words and their numbers are never recycled.
//! Cache this immutable pair in both directions. Each record still supplies its
//! own current address. A cache collision always consults the canonical table.
use super::{Shape, views_read, views_write};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

const SLOTS: usize = 256;
type Item = (u32, Shape);
static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

#[derive(Clone, Copy, Eq, PartialEq)]
enum Mode {
    Off,
    On,
    Compare,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("VICTORY_PLACED_SHAPE_CACHE").as_deref() {
            Ok("1") => Mode::On,
            Ok("abba") => Mode::Compare,
            _ => Mode::Off,
        },
    )
}

static ACTIVE: AtomicBool = AtomicBool::new(false);
pub(crate) fn comparison_enabled() -> bool {
    ACTIVE.load(Relaxed)
}
fn enabled() -> bool {
    match mode() {
        Mode::Off => false,
        Mode::On => true,
        Mode::Compare => comparison_enabled(),
    }
}

fn table_number(shape: &Shape) -> Option<u32> {
    let known = views_read().shape_index.get(shape).copied();
    known.or_else(|| views_write().shape(*shape))
}

fn table_shape(number: u32) -> Option<Shape> {
    views_read().shapes.get(number as usize).copied()
}

struct Local {
    by_words: Box<[Option<Item>]>,
    by_number: Box<[Option<Item>]>,
    counts: [u32; 4],
    pending: u32,
}

impl Local {
    fn new() -> Self {
        Self {
            by_words: vec![None; SLOTS].into_boxed_slice(),
            by_number: vec![None; SLOTS].into_boxed_slice(),
            counts: [0; 4],
            pending: 0,
        }
    }

    fn count(&mut self, index: usize) {
        self.counts[index] += 1;
        self.pending += 1;
        if self.pending == 2048 {
            for (published, local) in COUNTS.iter().zip(&mut self.counts) {
                published.fetch_add(u64::from(std::mem::take(local)), Relaxed);
            }
            self.pending = 0;
        }
    }

    fn slot(words: &Shape) -> usize {
        let mut hasher = DefaultHasher::new();
        words.hash(&mut hasher);
        hasher.finish() as usize % SLOTS
    }

    fn number(&mut self, words: &Shape, lookup: impl FnOnce(&Shape) -> Option<u32>) -> Option<u32> {
        let slot = Self::slot(words);
        if let Some((number, stored)) = self.by_words[slot].as_ref()
            && words == stored
        {
            let number = *number;
            self.count(0);
            return Some(number);
        }
        self.count(1);
        let number = lookup(words)?;
        let entry = Some((number, *words));
        self.by_words[slot] = entry;
        self.by_number[number as usize % SLOTS] = entry;
        Some(number)
    }

    fn shape(&mut self, number: u32, lookup: impl FnOnce(u32) -> Option<Shape>) -> Option<Shape> {
        let slot = number as usize % SLOTS;
        if let Some((stored_number, words)) = self.by_number[slot]
            && stored_number == number
        {
            self.count(2);
            return Some(words);
        }
        self.count(3);
        let words = lookup(number)?;
        self.by_number[slot] = Some((number, words));
        Some(words)
    }
}

thread_local! { static LOCAL: RefCell<Local> = RefCell::new(Local::new()); }

pub(super) fn number(words: &Shape) -> Option<u32> {
    if !enabled() {
        return table_number(words);
    }
    LOCAL
        .try_with(|cell| match cell.try_borrow_mut() {
            Ok(mut cache) => cache.number(words, table_number),
            Err(_) => table_number(words),
        })
        .unwrap_or_else(|_| table_number(words))
}

pub(super) fn shape(number: u32) -> Option<Shape> {
    if !enabled() {
        return table_shape(number);
    }
    LOCAL
        .try_with(|cell| match cell.try_borrow_mut() {
            Ok(mut cache) => cache.shape(number, table_shape),
            Err(_) => table_shape(number),
        })
        .unwrap_or_else(|_| table_shape(number))
}

// For a benchmark only. Start after three dense reports, then keep every
// interval, including stutters. The first interval of each block is settling.
#[derive(Default)]
struct Schedule {
    warm: u8,
    interval: u8,
}

const BLOCKS: [bool; 4] = [false, true, true, false];
const INTERVALS: u8 = 4;

impl Schedule {
    fn advance(&mut self, dense: bool) -> Option<(u8, u8, bool)> {
        if self.warm < 3 {
            self.warm = if dense { self.warm + 1 } else { 0 };
            return None;
        }
        let block = self.interval / INTERVALS;
        if usize::from(block) >= BLOCKS.len() {
            return None;
        }
        let slot = self.interval % INTERVALS;
        self.interval += 1;
        let next = BLOCKS
            .get(usize::from(self.interval / INTERVALS))
            .copied()
            .unwrap_or(false);
        Some((block, slot, next))
    }
}

pub(crate) fn report(draws: f64, fps: f64) {
    let mode = mode();
    if mode != Mode::Off {
        let [number_hits, number_misses, shape_hits, shape_misses] =
            std::array::from_fn(|i| COUNTS[i].load(Relaxed));
        tracing::info!(
            number_hits,
            number_misses,
            shape_hits,
            shape_misses,
            "placed shape lookups cumulative"
        );
    }
    if mode == Mode::Compare {
        static SCHEDULE: Mutex<Schedule> = Mutex::new(Schedule {
            warm: 0,
            interval: 0,
        });
        let mut schedule = SCHEDULE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((block, slot, next)) = schedule.advance(draws >= 4000.0 && fps >= 8.0) {
            let cached = BLOCKS[usize::from(block)];
            let target = "placed";
            tracing::info!(
                block,
                slot,
                cached,
                draws,
                fps,
                target,
                "descriptor comparison interval ended"
            );
            ACTIVE.store(next, Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collisions_and_first_missing_number_preserve_the_exact_words() {
        let mut cache = Local::new();
        let a = (false, [42; 14], [7; 14]);
        let b = (true, [42; 14], [7; 14]);
        assert_eq!(cache.shape(2, |_| None), None);
        assert_eq!(cache.number(&a, |_| Some(2)), Some(2));
        assert_eq!(cache.shape(2, |_| panic!("stored")), Some(a));
        assert_eq!(
            cache.number(&b, |_| Some(2 + SLOTS as u32)),
            Some(2 + SLOTS as u32)
        );
        assert_eq!(cache.shape(2, |_| Some(a)), Some(a));
        assert_eq!(cache.shape(2 + SLOTS as u32, |_| Some(b)), Some(b));
        // Force a forward-slot collision independently of the hash algorithm.
        cache.by_words[Local::slot(&a)] = Some((17, b));
        assert_eq!(cache.number(&a, |_| Some(2)), Some(2));
        assert_eq!(cache.number(&a, |_| panic!("stored")), Some(2));
    }

    #[test]
    fn all_sixteen_intervals_survive_a_stutter_after_warmup() {
        let mut schedule = Schedule::default();
        for dense in [true, false, true, true, true] {
            assert_eq!(schedule.advance(dense), None);
        }
        for interval in 0..16 {
            let block = interval / 4;
            let next = BLOCKS.get((interval + 1) / 4).copied().unwrap_or(false);
            assert_eq!(
                schedule.advance(false),
                Some((block as u8, (interval % 4) as u8, next))
            );
        }
        assert_eq!(schedule.advance(true), None);
    }
}
