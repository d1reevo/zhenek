//! Thread-local copies of resource descriptor descriptions. The slot's striped revision
//! is changed under the table's exclusive lock before a generation or its value is replaced.
use super::{View, views_read};
use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

const SLOTS: usize = 128;
const STRIPES: usize = 256;
const EXHAUSTED: u64 = u64::MAX;
static REVISIONS: [AtomicU64; STRIPES] = [const { AtomicU64::new(0) }; STRIPES];
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_LOOKUP_CACHE").is_some_and(|v| v == "1"))
}

fn revision(i: u32) -> &'static AtomicU64 {
    &REVISIONS[(i as usize) % STRIPES]
}

fn invalidate(revision: &AtomicU64) {
    // All writers are serialized by the outer Views lock. Permanently disable the
    // stripe on wrap so no old entry can become valid again.
    revision.store(
        revision.load(Ordering::Relaxed).saturating_add(1),
        Ordering::Release,
    );
}

pub(super) fn replacing(i: u32) {
    if enabled() {
        invalidate(revision(i));
    }
}

struct Entry {
    index: u32,
    generation: u32,
    revision: u64,
    view: View,
}

struct Local {
    slots: [Option<Entry>; SLOTS],
    hits: u32,
    misses: u32,
}

impl Local {
    const fn new() -> Self {
        Self {
            slots: [const { None }; SLOTS],
            hits: 0,
            misses: 0,
        }
    }

    fn resolve(
        &mut self,
        i: u32,
        generation: u32,
        revision: &AtomicU64,
        read_locked: impl FnOnce() -> Option<(View, u64)>,
    ) -> Option<View> {
        let at = ((i ^ (i >> 7)) as usize) % SLOTS;
        let serial = revision.load(Ordering::Acquire);
        let value = if let Some(entry) = self.slots[at].as_ref()
            && serial != EXHAUSTED
            && entry.revision == serial
            && entry.index == i
            && entry.generation == generation
        {
            self.hits += 1;
            Some(entry.view.clone())
        } else {
            self.misses += 1;
            read_locked().map(|(view, locked_serial)| {
                if locked_serial != EXHAUSTED {
                    self.slots[at] = Some(Entry {
                        index: i,
                        generation,
                        revision: locked_serial,
                        view: view.clone(),
                    });
                }
                view
            })
        };
        // No contended statistics on individual draws.
        if self.hits + self.misses == 512 {
            HITS.fetch_add(u64::from(self.hits), Ordering::Relaxed);
            MISSES.fetch_add(u64::from(self.misses), Ordering::Relaxed);
            self.hits = 0;
            self.misses = 0;
        }
        value
    }
}

thread_local! { static LOCAL: RefCell<Local> = const { RefCell::new(Local::new()) }; }

pub(super) fn lookup(i: u32, generation: u32) -> Option<View> {
    let serial = revision(i);
    let read = || {
        let table = views_read();
        let (view, current) = table.list.get(i as usize)?;
        (*current == generation).then(|| (view.clone(), serial.load(Ordering::Relaxed)))
    };
    LOCAL
        .try_with(|cell| match cell.try_borrow_mut() {
            Ok(mut cache) => cache.resolve(i, generation, serial, read),
            Err(_) => read().map(|(v, _)| v),
        })
        .unwrap_or_else(|_| read().map(|(v, _)| v))
}

pub(crate) fn report() {
    if enabled() {
        let hits = HITS.load(Ordering::Relaxed);
        let misses = MISSES.load(Ordering::Relaxed);
        tracing::info!(hits, misses, "resource descriptor table lookups cumulative");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn same_words_hit_and_a_stale_generation_cannot_be_resurrected() {
        let mut local = Local::new();
        let serial = AtomicU64::new(0);
        let v = View::Unordered(123, 456, None);
        assert_eq!(
            local.resolve(2, 1, &serial, || Some((v.clone(), 0))),
            Some(v.clone())
        );
        assert_eq!(
            local.resolve(2, 1, &serial, || panic!("unnecessary table lock")),
            Some(v)
        );
        // Another writer can release the counter or recycle the slot for an unrelated resource.
        std::thread::scope(|scope| {
            scope.spawn(|| invalidate(&serial));
        });
        assert_eq!(local.resolve(2, 1, &serial, || None), None);
        let fresh = View::Shader(246, Some([4; 10]));
        assert_eq!(
            local.resolve(2, 2, &serial, || Some((fresh.clone(), 1))),
            Some(fresh.clone())
        );
        assert_eq!(
            local.resolve(2, 2, &serial, || panic!("unnecessary table lock")),
            Some(fresh)
        );
    }

    #[test]
    fn recycled_entry_cannot_return_the_old_view_even_for_the_same_words() {
        let mut local = Local::new();
        let serial = AtomicU64::new(7);
        let a = View::Shader(1000, None);
        let b = View::Shader(2000, None);
        assert_eq!(
            local.resolve(9, 3, &serial, || Some((a.clone(), 7))),
            Some(a)
        );
        invalidate(&serial);
        assert_eq!(
            local.resolve(9, 3, &serial, || Some((b.clone(), 8))),
            Some(b)
        );
        assert_eq!(local.resolve(9, 4, &serial, || None), None);
    }

    #[test]
    fn stamp_is_the_one_seen_under_the_table_lock_and_exhaustion_bypasses() {
        let mut local = Local::new();
        let serial = AtomicU64::new(1);
        let a = View::Sampler([5; 13]);
        assert_eq!(
            local.resolve(17, 3, &serial, || {
                invalidate(&serial);
                Some((a.clone(), 2))
            }),
            Some(a.clone())
        );
        assert_eq!(
            local.resolve(17, 3, &serial, || panic!("stored a stale stamp")),
            Some(a.clone())
        );
        serial.store(EXHAUSTED - 1, Ordering::Relaxed);
        invalidate(&serial);
        invalidate(&serial);
        assert_eq!(
            local.resolve(17, 3, &serial, || Some((a.clone(), EXHAUSTED))),
            Some(a)
        );
        assert!(local.resolve(17, 3, &serial, || None).is_none());
    }
}
