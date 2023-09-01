//! Similar to [`std::sync::RwLock`], yet it locks only on writes and reads are
//! wait-free.
//!
//! [`VLock`] is for read-heavy scenarios. Updates are strictly serialized and may
//! have to wait until readers of older data are finished. It works by keeping
//! multiple versions of the shared data and counting references to each of
//! the versions. Named `VLock`, because it uses versioning and behaves sort-of
//! like a lock. Naming is hard.
//!
//! # Why bother?
//!
//! [`VLock`] is a fast[^1] and scalable lock with stable read performance for
//! non-copy types at the expense of slower writes.
//!
//! It's a common pattern among various sorts of systems, where there is a
//! number of processing threads, each making decisions via shared state and
//! some other code that updates that shared state once in a while. In network
//! infrastructure, for example, this can be thought of data and control planes,
//! be it routers or various levels of load balancers. In data-heavy applications,
//! that can be a hot view of some stuff stored in a database.
//!
//! If that roughly describes your use case, you may benefit from this library
//! at the expense of having 1+ extra copies of data in memory, slightly more
//! state and slower write performance. The former is usually necessary even
//! when [`RwLock`][std::sync::RwLock] is used to minimize time under lock,
//! and [`VLock`] actually reuses old versions which may save you some time
//! by avoiding allocations. The extra state is one `usize` plus a `usize`
//! for every version... and there's not much you can do about slower writes.
//!
//! If read performance is critical and readers can't afford to wait for writes,
//! you may benefit significantly.
//!
//! As a bonus, the implementation is simple with about 200 lines of actual
//! code and without any external dependencies.
//!
//! [^1]: Based on synthetic benchmarks on `x86_64` laptops, read performance was
//! 1.3-2.0 times faster than `ArcSwap`, and may be order of magnitude faster
//! than fastest `RwLock` implementation in certain cases. Writes of `VLock`
//! are more efficient than `ArcSwap`. Comparing to `RwLock`, writes are
//! generally 1 to 10 times slower than `parking_lot` implementation, but an
//! improvement over the `std` implementation. With `SeqLock` results were
//! mixed: in some scenarios reads of `VLock` was 4 times slower, in some about
//! 1:1 and in other 2 times quicker. Although write performance of `VLock`
//! is significantly worse than that of `SeqLock`, it can be used for non-copy
//! types. Note that write performance of `VLock` may degrade significantly
//! when readers are not progressing and `N` is small, in other words `VLock`
//! is susceptible to write starvation by prioritizing reads.
//!
//! # How does it work?
//!
//! As mentioned above, it uses a combination of versioning and reference
//! counting with a lock to serialize writes.
//!
//! [`VLock<T, N>`] has a fixed size `N`, which is the maximum number of
//! *versions* to allocate. Version is identified by an *offset* in the allocated
//! space. *State* has both the offset and the *counter* in the same atomic.
//! [`VLock`] keeps the current state and a state for every version.
//!
//! Every time [`read`][VLock::read] is called, the current state counter is
//! incremented and the associated offset is used to return the current version
//! to the reader. After the returned pointer is dropped, the per-version state
//! counter is decremented.
//!
//! During [`update`][VLock::update], a first unused version is picked by checking
//! whether per-version counter is 0, or if nothing is available, a new version
//! is initialized if size allows. This is the place where waiting for readers
//! is happening. Once a free offset is found and it is updated, the current
//! state counter is reset and the new offset is written (which is one atomic
//! operation). The returned counter is then added to the per-version counter
//! associated with that version, after which it will reach 0 once all readers
//! are done with it.
//!
//! In other words, tracking access can be thought of balancing with two counters -
//! one incrementing in the current state, marking the start of the read access
//! to data, and the other decrementing in the per-version state, marking the
//! end of the access. Then, both are added together. If the resulting counter
//! is greater than 0, there are still some readers, if it's 0, then the
//! version is definitely unused.
//!
//! Old versions are not dropped, so it acts like a pool. That may be handy
//! when dealing with heap-allocated datastructures that have to be updated
//! or rebuilt, but may also lead to some old garbage lying around in memory.
//!
//! # Usage
//!
//! ```
//! use std::sync::Arc;
//! use std::thread;
//! use std::time;
//!
//! use vlock::VLock;
//!
//! let lock = Arc::new(VLock::<String, 4>::new(String::from("hi there!")));
//! let lock_clone = Arc::clone(&lock);
//! let t = thread::spawn(move || {
//!     for _ in 0..5 {
//!         println!("{}", *lock_clone.read());
//!         thread::sleep(time::Duration::from_millis(1));
//!     }
//!     lock_clone.update(
//!         |_, value| {
//!             value.clear();
//!             value.push_str("bye!");
//!         },
//!         String::new
//!     );
//! });
//! thread::sleep(time::Duration::from_millis(2));
//! lock.update(
//!     |_, value| {
//!         value.clear();
//!         value.push_str("here's some text for you");
//!     },
//!     String::new
//! );
//! if let Err(err) = t.join() {
//!     println!("thread has failed: {err:?}");
//! }
//! assert_eq!(*lock.read(), "bye!");
//! ```

#![warn(missing_docs)]
#![warn(clippy::pedantic)]

use std::{borrow, cell, fmt, hint, marker, mem, ops, ptr, sync::atomic, thread};

// The number of attempts to acquire one of old versions for updating, spinning
// in between attempts exponentially.
const ACQUIRE_ATTEMPTS: usize = 14;
// Number of times to yield to the scheduler before attempting to initialize
// a new version. This makes the number of initialized versions slowly adapt to
// the level of saturation.
const YIELDS_BEFORE_INIT: usize = 2;
const LOCKED: usize = 0;

/// A versioned lock.
///
/// Allows multiple readers and a single writer at a time. Reads are wait-free,
/// writes are protected by a lock.
///
/// The type parameter `T` describes the data that is protected by the lock.
/// Constant `N` is the maximum number of versions of the data that can be
/// allocated.
///
/// # Examples
///
/// ```
/// use vlock::VLock;
///
/// let lock: VLock<_, 2> = 10.into();
/// // there can be multiple reads
/// {
///     let read1 = lock.read();
///     let read2 = lock.read();
///     assert_eq!(*read1, 10);
///     assert_eq!(*read2, 10);
/// }
///
/// // old versions are still accessible after update
/// {
///     let read1 = lock.read();
///     // this triggers a new version to be initialized with 2
///     lock.update(|_, value| *value += 20, || 2);
///     let read2 = lock.read();
///     assert_eq!(*read1, 10);
///     assert_eq!(*read2, 22);
///     // attempt to update here will block until read1 is dropped
/// } // read1 is dropped, allowing more updates
///
/// // current version can be accessed directly when updating
/// {
///     lock.update(|curr, value| *value = *curr + 8, || 2);
///     let read1 = lock.read();
///     assert_eq!(*read1, 30);
/// }
/// ```
pub struct VLock<T, const N: usize> {
    // The state for the current version. Contains the offset in data of the
    // current version and a counter of acquired reads to it.
    state: atomic::AtomicUsize,

    // The initialized length of data. Since we need a lock for writes, this
    // value is exploited - LOCKED means the writes are locked.
    length: atomic::AtomicUsize,

    // Versions of data, initialized lazily. Each item in the array keeps its
    // own state to check whether this version is still in use.
    data: cell::UnsafeCell<[mem::MaybeUninit<Data<T>>; N]>,
}

impl<T, const N: usize> VLock<T, N> {
    const STEP: usize = {
        assert!(N > 1, "VLock requires at least 2 versions to work");
        N.next_power_of_two()
    };
    const OFFSET: usize = Self::STEP - 1;
    const COUNTER: usize = !Self::OFFSET;

    /// Creates a new unlocked instance of `VLock` with the initial version.
    pub fn new(value: T) -> Self {
        // SAFETY: The assume_init is for the array of MaybeUninits.
        let mut data: [mem::MaybeUninit<Data<T>>; N] =
            unsafe { mem::MaybeUninit::uninit().assume_init() };
        data[0].write(Data {
            state: atomic::AtomicUsize::new(0),
            value,
        });
        Self {
            state: atomic::AtomicUsize::new(0),
            length: atomic::AtomicUsize::new(1),
            data: cell::UnsafeCell::new(data),
        }
    }

    #[inline]
    unsafe fn at(&self, offset: usize) -> &mem::MaybeUninit<Data<T>> {
        &(&*self.data.get())[offset]
    }

    #[allow(clippy::mut_from_ref)]
    #[inline]
    unsafe fn at_mut(&self, offset: usize) -> &mut mem::MaybeUninit<Data<T>> {
        &mut (&mut *self.data.get())[offset]
    }

    /// Acquires a reference to the current version of `T`.
    ///
    /// This call never blocks. However, holding on to a version for too long
    /// may block concurrent updates, so aim to have frequent short reads if
    /// possible. Note, that repeated calls may return a newer version.
    ///
    /// Returned reference is RAII-style, releasing the access once dropped.
    ///
    /// # Panics
    ///
    /// Attempt to acquire will panic if there is an overflow in the internal
    /// counter for the current version. The counter max is
    /// `usize::MAX >> N.next_power_of_two().ilog2()` and corresponds to the
    /// number of total `read` calls that can happen to a single version. Every
    /// next [`update`][`VLock::update`] resets the counter.
    ///
    /// Normally, this should not happen, but one can imagine a system with
    /// high frequency of reads running for too long without any data updates.
    ///
    /// # Examples
    ///
    /// ```
    /// use vlock::VLock;
    ///
    /// let lock: VLock<_, 2> = 10.into();
    /// {
    ///     let value = lock.read();
    ///     assert_eq!(*value, 10);
    /// } // the access is released after value is dropped
    /// ```
    pub fn read(&self) -> ReadRef<'_, T, N> {
        // Relaxed is OK. Even if the old version is retrieved just before the
        // update happens, the counter will correctly reflect the state, because
        // retrieving offset and setting the counter is the same atomic operation.
        // In other words, we care only about atomicity here.
        let state = self.state.fetch_add(Self::STEP, atomic::Ordering::Relaxed);
        assert_ne!(
            state.wrapping_add(Self::STEP),
            state & Self::OFFSET,
            "counter overflow"
        );

        ReadRef {
            // SAFETY: Current offset always points to init data - see new() and
            // update(). No mutable borrow can happen: update() mutates non-current
            // version and prevents selecting versions for update with non-zero
            // counter, which is guaranteed until this ReadRef is dropped.
            ptr: unsafe { self.at(state & Self::OFFSET) }.as_ptr(),
            phantom: marker::PhantomData,
        }
    }

    #[inline]
    fn lock(&self) -> usize {
        loop {
            let value = self.length.swap(LOCKED, atomic::Ordering::Acquire);
            if value != LOCKED {
                return value;
            }
            thread::yield_now();
        }
    }

    #[inline]
    fn acquire<I>(&self, curr_offset: usize, length: usize, init: I) -> usize
    where
        I: FnOnce() -> T,
    {
        let mut remaining = ACQUIRE_ATTEMPTS;
        loop {
            'attempt: loop {
                if length == 1 {
                    break 'attempt;
                }

                // SAFETY: No concurrent mutations can happen because of the lock.
                if let Some(state) = unsafe { &*self.data.get() }
                    .iter()
                    .enumerate()
                    .take(length)
                    .filter(|(offset, _)| *offset != curr_offset)
                    // SAFETY: Length counts inits. It's safe to assume that
                    // the first length MaybeUninits are inits.
                    .map(|(_, init)| unsafe { init.assume_init_ref() })
                    // These versions are not "active", i.e. there are no new reads to
                    // these happening at this point, only some old readers may be
                    // holding on to these.
                    //
                    // Taking Acquire here to ensure that all access to that version
                    // has completed before it can be reused.
                    .map(|version| version.state.load(atomic::Ordering::Acquire))
                    .find(|&state| state & Self::COUNTER == 0)
                {
                    return state & Self::OFFSET;
                }

                // That was the last attempt. Keep it that way indefinitely.
                if remaining == 1 {
                    break 'attempt;
                }

                // Spin with exponential waiting time and only then yield
                // YIELDS_BEFORE_INIT times.
                if remaining > YIELDS_BEFORE_INIT + 1 {
                    for _ in 0..1 << ACQUIRE_ATTEMPTS.saturating_sub(remaining) {
                        hint::spin_loop();
                    }
                } else {
                    thread::yield_now();
                }
                remaining -= 1;
            }

            // Try to initialize a new version, if there's room for that.
            if length < N {
                // SAFETY: The version at length is uninit. It is safe to init.
                // No concurrent mutations can happen because of the lock.
                unsafe { self.at_mut(length) }.write(Data {
                    state: atomic::AtomicUsize::new(length),
                    value: init(),
                });
                return length;
            }

            // No new versions can be initialized and previous versions are busy.
            // From this point on there will be just one attempt to acquire,
            // yielding in between.
            thread::yield_now();
        }
    }

    #[inline]
    fn unlock(&self, value: usize) {
        assert_ne!(value, LOCKED, "locking unlock");
        assert_eq!(
            self.length.swap(value, atomic::Ordering::Release),
            LOCKED,
            "unlock without lock"
        );
    }

    /// Locks this `VLock` and calls `f` with the current and one of the
    /// previously used or a newly initialized versions, blocking the current
    /// thread if the lock can't be acquired or if all `N` versions of data are
    /// in use.
    ///
    /// If a new version needs initialization, `init` will be called before
    /// `f`. This happens when all initialized versions are in use and the
    /// number of versions is less than `N`.
    ///
    /// Note, that there is no guarantee which exact non-current version will
    /// be passed to `f`, because it depends on reader access patterns. Unless
    /// `N` equals 2, of course.
    ///
    /// Current implementation tries to avoid initializing new versions by
    /// attempting to access already-initialized non-current versions multiple
    /// times with exponential wait time in between, and then yielding few times
    /// in hope that readers will progress. This seems to be a good balance to
    /// avoid aggressive initialization when there is high contention, and
    /// while waiting longer initially, eventually the number of initialized
    /// versions will grow to match the saturation level if size allows.
    ///
    /// # Panics
    ///
    /// If the current version changes during `update` for any reason, or
    /// unlocking encounters unexpected state, the `update` will panic, leaving
    /// the state unrecoverable. Bit-flips are rare, but do happen nevertheless.
    ///
    /// # Examples
    ///
    /// ```
    /// use vlock::VLock;
    ///
    /// let lock: VLock<_, 2> = 10.into();
    /// assert_eq!(*lock.read(), 10);
    /// lock.update(|_, value| *value += 20, || 13);
    /// assert_eq!(*lock.read(), 33);
    /// lock.update(|_, value| *value += 20, || 13);
    /// assert_eq!(*lock.read(), 30);
    /// ```
    pub fn update<F, I>(&self, f: F, init: I)
    where
        F: FnOnce(&T, &mut T),
        I: FnOnce() -> T,
    {
        let mut length = self.lock();
        // Relaxed is fine, because all changes to the offset are behind a
        // lock which was acquired just above.
        let offset = self.state.load(atomic::Ordering::Relaxed) & Self::OFFSET;

        let new_offset = self.acquire(offset, length, init);
        if new_offset == length {
            length = length.saturating_add(1);
        }

        // SAFETY: Current version is init, which happened either in new() or
        // in acquire() during previous update(). Next mutable borrow at this
        // offset can happen only in a subsequent update(), which is serialized.
        let version = unsafe { self.at(offset).assume_init_ref() };
        f(
            &version.value,
            // SAFETY: Data at new_offset is ensured to be init and new_offset
            // cannot overlap with the current offset. Reference counting tracks
            // that there are no active "borrows". See acquire() for details.
            // No concurrent mutations can happen because of the lock.
            &mut unsafe { self.at_mut(new_offset).assume_init_mut() }.value,
        );

        // Update the state to point to the new offset and reset the counter for
        // that version. This is the point when the new read() calls start to
        // refer to the new version.
        //
        // Changes to the offset are behind a lock. Regarding the counter, we are
        // interested only in atomic operation in read(). So, using Relaxed here
        // should be OK.
        let prev_state = self.state.swap(new_offset, atomic::Ordering::Relaxed);

        // Who knows what is going on on a computer this is running on.
        assert_eq!(
            prev_state & Self::OFFSET,
            offset,
            "offset changed while writing"
        );

        // Add the total number of times the previous version was handed out to
        // the counter of that same version, so it can reach 0 when all borrows
        // are dropped.
        //
        // Relaxed should be OK. Next access to that version is in subsequent
        // update() call. If the counter happens to go to 0 at this point, the
        // next access to that version is synchronized by the lock. Otherwise,
        // synchronization is ensured via Acquire load.
        version
            .state
            .fetch_add(prev_state & Self::COUNTER, atomic::Ordering::Relaxed);

        self.unlock(length);
    }

    /// Returns a mutable reference to the current version.
    ///
    /// Because of a mutable borrow, exclusive access is guaranteed by the compiler.
    ///
    /// # Examples
    ///
    /// ```
    /// use vlock::VLock;
    ///
    /// let mut lock: VLock<_, 2> = 10.into();
    /// assert_eq!(*lock.get_mut(), 10);
    /// ```
    pub fn get_mut(&mut self) -> &mut T {
        let offset = *self.state.get_mut() & Self::OFFSET;
        // SAFETY: Exclusive mutable access is guaranteed by the compiler.
        // Current offset always points to init data - see new() and update().
        &mut unsafe { self.at_mut(offset).assume_init_mut() }.value
    }
}

impl<T, const N: usize> From<T> for VLock<T, N> {
    fn from(value: T) -> Self {
        VLock::new(value)
    }
}

impl<T: Default, const N: usize> Default for VLock<T, N> {
    fn default() -> Self {
        VLock::new(T::default())
    }
}

unsafe impl<T: Send + Sync, const N: usize> Send for VLock<T, N> {}
unsafe impl<T: Send + Sync, const N: usize> Sync for VLock<T, N> {}

impl<T: Clone, const N: usize> Clone for VLock<T, N> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.read().clone())
    }

    fn clone_from(&mut self, source: &Self) {
        let offset = *self.state.get_mut() & Self::OFFSET;
        // SAFETY: Exclusive mutable access is guaranteed by the compiler.
        let current = unsafe { self.at_mut(offset).assume_init_mut() };
        *current.state.get_mut() = 0;
        current.value.clone_from(&source.read());
        if offset != 0 {
            // SAFETY: Exclusive mutable access is guaranteed by the compiler.
            let first = unsafe { self.at_mut(0).assume_init_mut() };
            mem::swap(first, current);
        }
        *self.state.get_mut() = 0;

        // SAFETY: Exclusive mutable access is guaranteed by the compiler.
        for init in unsafe { &mut *self.data.get() }
            .iter_mut()
            .take(*self.length.get_mut())
            .skip(1)
        {
            // SAFETY: Length counts inits. It's safe to assume that the first
            // length MaybeUninits are inits.
            let state = &mut unsafe { init.assume_init_mut() }.state;
            assert_eq!(*state.get_mut() & Self::COUNTER, 0);
            unsafe { init.assume_init_drop() };
        }
        *self.length.get_mut() = 1;
    }
}

impl<T, const N: usize> fmt::Debug for VLock<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let length = self.length.load(atomic::Ordering::Relaxed);
        if length == LOCKED {
            f.debug_struct("VLock (locked)")
                .field("state", &self.state)
                .finish_non_exhaustive()
        } else {
            f.debug_struct("VLock")
                .field("state", &self.state)
                .field("length", &length)
                .finish_non_exhaustive()
        }
    }
}

impl<T, const N: usize> Drop for VLock<T, N> {
    #[inline]
    fn drop(&mut self) {
        // SAFETY: Exclusive mutable access is guaranteed by the compiler.
        for init in unsafe { &mut *self.data.get() }
            .iter_mut()
            .take(*self.length.get_mut())
        {
            // SAFETY: Length counts inits. It's safe to assume that the first
            // length MaybeUninits are inits.
            unsafe {
                init.assume_init_drop();
            }
        }
    }
}

#[derive(Debug)]
struct Data<T> {
    // The counter is decremented every time the read reference to that data
    // is dropped, and incremented by the total number of read() calls to that
    // version when the new version is written. We also keep an offset here
    // to match the state of VLock.
    //
    // Note, that overflow is expected. Values are not compared to anything
    // other than 0 anyway and it will reach 0 regardless of the sign.
    state: atomic::AtomicUsize,
    value: T,
}

/// A barely smart pointer referencing data owned by [`VLock`]. The access
/// to the data is RAII-style, and is released when dropped.
///
/// This type is created by [`VLock::read`].
///
/// This type can't be cloned. Holding on to a version for too long is no good.
/// Instead, call [`read`] to acquire more copies if necessary. The next [`read`]
/// may return a newer version, however.
///
/// [`read`]: VLock::read
pub struct ReadRef<'a, T, const N: usize> {
    ptr: *const Data<T>,
    phantom: marker::PhantomData<&'a Data<T>>,
}

impl<T, const N: usize> Eq for ReadRef<'_, T, N> {}

impl<T, const N: usize> PartialEq for ReadRef<'_, T, N> {
    /// Equality by comparing the addresses of two pointers, which if equal
    /// guarantee that two versions are in fact the same exact version.
    ///
    /// Dereference to compare the inner values.
    ///
    /// # Examples
    ///
    /// ```
    /// use vlock::VLock;
    ///
    /// let lock: VLock<_, 2> = 10.into();
    /// let read1 = lock.read();
    /// let read2 = lock.read();
    /// assert_eq!(read1, read2);
    ///
    /// lock.update(|curr, value| *value = *curr, || 0);
    /// let read3 = lock.read();
    /// assert_ne!(read2, read3);
    /// assert_eq!(*read2, *read3);
    /// ```
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.ptr == other.ptr
    }
}

impl<T, const N: usize> AsRef<T> for ReadRef<'_, T, N> {
    #[inline]
    fn as_ref(&self) -> &T {
        self
    }
}

impl<T, const N: usize> borrow::Borrow<T> for ReadRef<'_, T, N> {
    #[inline]
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for ReadRef<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&**self, f)
    }
}

impl<T: fmt::Display, const N: usize> fmt::Display for ReadRef<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&**self, f)
    }
}

impl<T, const N: usize> fmt::Pointer for ReadRef<'_, T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&ptr::addr_of!(**self), f)
    }
}

impl<T, const N: usize> ops::Deref for ReadRef<'_, T, N> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &T {
        // SAFETY: Pointer is non-null based on the context where it is set.
        // No mutable borrow of the same address can happen, until Self is
        // dropped due to reference counting. See VLock::read() for details.
        &unsafe { &*self.ptr }.value
    }
}

unsafe impl<T: Sync, const N: usize> Sync for ReadRef<'_, T, N> {}

impl<T, const N: usize> Drop for ReadRef<'_, T, N> {
    #[inline]
    fn drop(&mut self) {
        // Release to synchronize later reuse of the data this points to.
        //
        // SAFETY: Pointer is non-null based on the context where it is set.
        unsafe { &*self.ptr }.state.fetch_sub(
            /*STEP*/ N.next_power_of_two(),
            atomic::Ordering::Release,
        );
    }
}
