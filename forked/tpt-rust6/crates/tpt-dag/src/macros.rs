//! `task!` and `pipeline!` declarative macros.
//!
//! `macro_rules!` cannot define a *real* attribute macro (that requires a
//! proc-macro crate, which this crate deliberately avoids), so the attribute
//! is written **inside** a `task! { ... }` invocation:
//!
//! ```text
//! task! {
//!     #[task(cache = true, retries = 3)]
//!     fn ingest(path: &str) -> Frame { ... }
//! }
//! ```
//!
//! The expansion emits the original, normally-callable function *plus* a
//! same-named hidden module holding its [`crate::TaskMeta`] and a `register()`
//! function, so metadata is discoverable via [`crate::task_meta`].

/// Declare a pipeline task.
///
/// Accepted forms (options may also be omitted entirely):
///
/// * `task! { fn f(a: A) -> R { .. } }`
/// * `task! { #[task] fn f(a: A) -> R { .. } }`
/// * `task! { #[task(cache = true)] .. }` / `#[task(retries = 3)]`
/// * `task! { #[task(cache = true, retries = 3)] .. }` (either order)
///
/// Scope: plain (non-generic, non-`async`) free functions with `ident: Type`
/// parameters. The return type may be `T` or `Result<T, E: Display>`; the
/// latter drives the retry logic. `T` must be `Clone + Send + Sync + 'static`
/// to be passed between nodes.
#[macro_export]
macro_rules! task {
    // ---- internal emitter -------------------------------------------------
    (@emit $retries:expr, $cache:expr,
        $(#[$m:meta])* $vis:vis fn $name:ident ( $($arg:ident : $ty:ty),* $(,)? ) $(-> $ret:ty)? $body:block
    ) => {
        $(#[$m])*
        #[allow(dead_code)]
        $vis fn $name ( $($arg : $ty),* ) $(-> $ret)? {
            $name::register();
            $body
        }

        #[doc(hidden)]
        #[allow(non_snake_case, dead_code)]
        $vis mod $name {
            /// Metadata for this task.
            pub const META: $crate::TaskMeta =
                $crate::TaskMeta::new(stringify!($name), $retries, $cache);
            static ONCE: ::std::sync::Once = ::std::sync::Once::new();
            /// Idempotently publish `META` into the global task registry.
            pub fn register() {
                ONCE.call_once(|| $crate::register_task(META));
            }
        }
    };

    // ---- attribute forms --------------------------------------------------
    ( #[task(cache = $c:expr, retries = $r:expr)] $($rest:tt)* ) => {
        $crate::task!(@emit $r, $c, $($rest)*);
    };
    ( #[task(retries = $r:expr, cache = $c:expr)] $($rest:tt)* ) => {
        $crate::task!(@emit $r, $c, $($rest)*);
    };
    ( #[task(cache = $c:expr)] $($rest:tt)* ) => {
        $crate::task!(@emit 0u32, $c, $($rest)*);
    };
    ( #[task(retries = $r:expr)] $($rest:tt)* ) => {
        $crate::task!(@emit $r, false, $($rest)*);
    };
    ( #[task] $($rest:tt)* ) => {
        $crate::task!(@emit 0u32, false, $($rest)*);
    };
    // ---- bare function ----------------------------------------------------
    ( $($rest:tt)* ) => {
        $crate::task!(@emit 0u32, false, $($rest)*);
    };
}

/// Normalise a task body's return value into `Result<T, String>`.
///
/// Uses inherent-impl specialization: `Wrap<Result<T, E>>` has an inherent
/// `take_task_result`, everything else falls back to the `PlainOutput` trait.
#[doc(hidden)]
#[macro_export]
macro_rules! __task_result {
    ($call:expr) => {{
        #[allow(unused_imports)]
        use $crate::PlainOutput as _;
        $crate::Wrap(::core::option::Option::Some($call)).take_task_result()
    }};
}

/// Build a [`crate::Pipeline`] from a chain of `#[task]` functions.
///
/// ```text
/// pipeline![ ingest("data/*.parquet") -> clean -> [train, evaluate] ]
/// ```
///
/// Grammar (kept deliberately small):
///
/// * the head may take literal arguments: `ingest("path", 3)`;
/// * `a -> b` passes `a`'s output as `b`'s **first** argument;
/// * extra literals may follow: `a -> b(0.5)` calls `b(a_out, 0.5)`;
/// * `-> [x, y]` fans out; each branch is itself a chain, so
///   `-> [x -> y, z]` nests.
///
/// Not supported (documented scope reduction): fan-*in* (`[a, b] -> c`) and
/// diamonds, because a node receives exactly one upstream value. Node names
/// are the task names; a task reused in one DAG gets a `#n` suffix.
#[macro_export]
macro_rules! pipeline {
    ( $($t:tt)* ) => {{
        let mut __b = $crate::PipelineBuilder::new();
        $crate::__pipe!(@head __b, $($t)*);
        __b.build()
    }};
}

#[doc(hidden)]
#[macro_export]
macro_rules! __pipe {
    // ---- root node (no upstream): returns its node name -------------------
    (@root $b:ident, $name:ident, $($a:expr),*) => {{
        $b.add(
            $name::META,
            ::std::vec::Vec::new(),
            stringify!($name ( $($a),* )),
            move || -> $crate::NodeResult {
                let __v = $crate::__task_result!($name($($a),*))?;
                ::core::result::Result::Ok(::std::sync::Arc::new(__v))
            },
        )
    }};

    // ---- child node (one upstream): returns its node name -----------------
    (@child $b:ident, $parent:expr, $name:ident, $($a:expr),*) => {{
        let __dep: ::std::string::String = ::std::string::ToString::to_string(&$parent);
        let __store = $b.store();
        let __dep2 = __dep.clone();
        $b.add(
            $name::META,
            ::std::vec![__dep],
            stringify!($name (upstream $($a),*)),
            move || -> $crate::NodeResult {
                let __in = $crate::fetch(&__store, &__dep2)?;
                let __v = $crate::__task_result!($name(__in, $($a),*))?;
                ::core::result::Result::Ok(::std::sync::Arc::new(__v))
            },
        )
    }};

    // ---- head of the pipeline ---------------------------------------------
    (@head $b:ident, $name:ident ( $($a:expr),* $(,)? ) -> $($rest:tt)*) => {
        let __parent = $crate::__pipe!(@root $b, $name, $($a),*);
        $crate::__pipe!(@chain $b, __parent, $($rest)*);
    };
    (@head $b:ident, $name:ident ( $($a:expr),* $(,)? )) => {
        let _ = $crate::__pipe!(@root $b, $name, $($a),*);
    };
    (@head $b:ident, $name:ident -> $($rest:tt)*) => {
        let __parent = $crate::__pipe!(@root $b, $name,);
        $crate::__pipe!(@chain $b, __parent, $($rest)*);
    };
    (@head $b:ident, $name:ident) => {
        let _ = $crate::__pipe!(@root $b, $name,);
    };

    // ---- chain continuation -----------------------------------------------
    (@chain $b:ident, $parent:expr, [ $($g:tt)* ]) => {
        $crate::__pipe!(@group $b, $parent, [$($g)*] []);
    };
    (@chain $b:ident, $parent:expr, $name:ident ( $($a:expr),* $(,)? ) -> $($rest:tt)*) => {
        let __parent = $crate::__pipe!(@child $b, $parent, $name, $($a),*);
        $crate::__pipe!(@chain $b, __parent, $($rest)*);
    };
    (@chain $b:ident, $parent:expr, $name:ident ( $($a:expr),* $(,)? )) => {
        let _ = $crate::__pipe!(@child $b, $parent, $name, $($a),*);
    };
    (@chain $b:ident, $parent:expr, $name:ident -> $($rest:tt)*) => {
        let __parent = $crate::__pipe!(@child $b, $parent, $name,);
        $crate::__pipe!(@chain $b, __parent, $($rest)*);
    };
    (@chain $b:ident, $parent:expr, $name:ident) => {
        let _ = $crate::__pipe!(@child $b, $parent, $name,);
    };

    // ---- fan-out: split the bracket group on top-level commas -------------
    (@group $b:ident, $parent:expr, [] []) => {};
    (@group $b:ident, $parent:expr, [] [$($cur:tt)+]) => {
        { $crate::__pipe!(@chain $b, $parent, $($cur)+); }
    };
    (@group $b:ident, $parent:expr, [, $($rest:tt)*] [$($cur:tt)+]) => {
        { $crate::__pipe!(@chain $b, $parent, $($cur)+); }
        $crate::__pipe!(@group $b, $parent, [$($rest)*] []);
    };
    (@group $b:ident, $parent:expr, [$t:tt $($rest:tt)*] [$($cur:tt)*]) => {
        $crate::__pipe!(@group $b, $parent, [$($rest)*] [$($cur)* $t]);
    };
}
