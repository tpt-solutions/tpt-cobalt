//! Pythonic method chaining over [`tpt_omni::Table`].
//!
//! Every method comes in two flavours:
//!
//! * a *script-mode* method (`filter`, `groupby`, `agg`, `plot`, ...) that
//!   raises a [`ScriptError`] as a panic, exactly like a Python exception, so
//!   chains stay clean. [`crate::run`] / [`crate::run_script`] turn that panic
//!   back into a rich traceback.
//! * a `try_*` twin returning [`Res`] for library use with `?`.

use std::sync::Arc;

use tpt_columnar::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray, UInt32Array,
};
use tpt_columnar::datatypes::DataType;
use tpt_omni::table::Scalar;
use tpt_omni::{Expr, OmniFrame, Table};

use crate::error::{Res, ScriptError};

/// A dynamically typed literal accepted by [`TableExt::filter`].
///
/// Integer literals (`18`) are accepted and coerced to the column's dtype, so
/// `t.filter("age", ">", 18)` works on both `Int64` and `Float64` columns.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
}

macro_rules! from_value {
    ($($t:ty => $v:expr),* $(,)?) => {$(
        impl From<$t> for Value { fn from(x: $t) -> Value { #[allow(clippy::redundant_closure_call)] ($v)(x) } }
    )*};
}
from_value! {
    i32 => |x| Value::Int(x as i64),
    i64 => Value::Int,
    usize => |x: usize| Value::Int(x as i64),
    f32 => |x| Value::Float(x as f64),
    f64 => Value::Float,
    bool => Value::Bool,
    &str => |x: &str| Value::Str(x.to_string()),
    String => Value::Str,
}

fn scalar_for(v: &Value, dt: &DataType, col: &str) -> Res<Scalar> {
    Ok(match (dt, v) {
        (DataType::Float64, Value::Float(f)) => Scalar::F64(*f),
        (DataType::Float64, Value::Int(i)) => Scalar::F64(*i as f64),
        (DataType::Int64, Value::Int(i)) => Scalar::I64(*i),
        (DataType::Int64, Value::Float(f)) => Scalar::I64(*f as i64),
        (DataType::Boolean, Value::Bool(b)) => Scalar::Bool(*b),
        (DataType::Utf8, Value::Str(s)) => Scalar::Str(s.clone()),
        _ => {
            return Err(ScriptError::new(
                "TypeError",
                format!("cannot compare column '{col}' ({dt:?}) with {v:?}"),
            ))
        }
    })
}

fn build_expr(col: &str, op: &str, s: Scalar) -> Res<Expr> {
    let name = col.to_string();
    Ok(match op {
        ">" | "gt" => Expr::Gt(name, s),
        "<" | "lt" => Expr::Lt(name, s),
        ">=" | "ge" => Expr::Ge(name, s),
        "<=" | "le" => Expr::Le(name, s),
        "==" | "=" | "eq" => Expr::Eq(name, s),
        "!=" | "<>" | "ne" => Expr::Ne(name, s),
        _ => {
            return Err(ScriptError::new(
                "OperatorError",
                format!("unknown operator '{op}' (use >, <, >=, <=, ==, !=)"),
            ))
        }
    })
}

fn column(t: &Table, name: &str) -> Res<ArrayRef> {
    t.column(name)
        .map_err(|_| ScriptError::new("ColumnError", format!("column '{name}' not found")))
}

/// Numeric values of a column (`Float64`, `Int64` or `Boolean`), nulls as `None`.
fn numeric(arr: &ArrayRef, name: &str) -> Res<Vec<Option<f64>>> {
    let n = arr.len();
    let get = |i: usize, v: f64| if arr.is_null(i) { None } else { Some(v) };
    Ok(match arr.data_type() {
        DataType::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
            (0..n).map(|i| get(i, a.value(i))).collect()
        }
        DataType::Int64 => {
            let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
            (0..n).map(|i| get(i, a.value(i) as f64)).collect()
        }
        DataType::Boolean => {
            let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
            (0..n)
                .map(|i| get(i, if a.value(i) { 1.0 } else { 0.0 }))
                .collect()
        }
        dt => {
            return Err(ScriptError::new(
                "TypeError",
                format!("column '{name}' is not numeric ({dt:?})"),
            ))
        }
    })
}

/// Stable string rendering of one cell, used as a group key.
fn key_at(arr: &ArrayRef, i: usize, name: &str) -> Res<String> {
    if arr.is_null(i) {
        return Ok("null".into());
    }
    Ok(match arr.data_type() {
        DataType::Utf8 => arr
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(i)
            .to_string(),
        DataType::Int64 => arr
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(i)
            .to_string(),
        DataType::Float64 => format!(
            "{}",
            arr.as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(i)
        ),
        DataType::Boolean => arr
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(i)
            .to_string(),
        dt => {
            return Err(ScriptError::new(
                "TypeError",
                format!("column '{name}' ({dt:?}) cannot be used as a group key"),
            ))
        }
    })
}

/// Unwrap a [`Res`], panicking on error.
///
/// This is the script-mode escape hatch: the ergonomic [`TableExt`] methods
/// (`filter`, `groupby`, `col_f64`, `agg`, `plot`, ...) call `raise` so that
/// `.tpt` scripts get Python-like exceptions (turned back into rich tracebacks
/// by [`crate::run`]). **Plain-Rust library callers must not rely on these
/// methods for recoverable error handling** — they will panic.
///
/// For library use, call the `try_*` twins (`try_filter`, `try_groupby`,
/// `try_col_f64`, `try_agg`, ...) directly and propagate the [`Res`] with `?`,
/// or use [`try_raise`] when you only need to convert the error.
///
/// # Panics
/// Panics, formatting the [`ScriptError`], whenever `r` is `Err`.
fn raise<T>(r: Res<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    }
}

/// Non-panicking variant of [`raise`]: returns the error instead of panicking.
///
/// Library callers who want to handle a single fallible step without the
/// `try_*` method family can use this to convert a [`Res<T>`] into a
/// `Result<T, ScriptError>` without aborting the thread.
pub fn try_raise<T>(r: Res<T>) -> Result<T, ScriptError> {
    r
}

/// `f64s("x", vec![1.0, 2.0])` — a `Float64` column for [`table_of`].
pub fn f64s(name: &str, v: impl Into<Vec<f64>>) -> (String, ArrayRef) {
    (name.to_string(), Arc::new(Float64Array::from(v.into())))
}

/// `i64s("n", vec![1, 2])` — an `Int64` column for [`table_of`].
pub fn i64s(name: &str, v: impl Into<Vec<i64>>) -> (String, ArrayRef) {
    (name.to_string(), Arc::new(Int64Array::from(v.into())))
}

/// `strs("g", &["a", "b"])` — a `Utf8` column for [`table_of`].
pub fn strs(name: &str, v: &[&str]) -> (String, ArrayRef) {
    (name.to_string(), Arc::new(StringArray::from(v.to_vec())))
}

/// Build a [`Table`] from columns (via [`OmniFrame::from_columns`]); raises on
/// mismatched column lengths.
pub fn table_of(cols: Vec<(String, ArrayRef)>) -> Table {
    raise(
        OmniFrame::from_columns(cols)
            .map(|f| f.as_table())
            .map_err(ScriptError::from),
    )
}

/// Pythonic chaining methods added to [`tpt_omni::Table`].
///
/// # Script vs. library error handling
/// The non-`try_` methods (`filter`, `groupby`, `col_f64`, `agg`, `plot`, ...)
/// raise any [`ScriptError`] as a **panic** (via [`raise`]) for clean
/// script-style chaining; these are intended for `.tpt` scripts run through
/// [`crate::run`]. **Library callers should use the `try_*` twins**
/// (`try_filter`, `try_groupby`, `try_col_f64`, `try_agg`, ...) and propagate
/// errors with `?`, or use [`try_raise`] for a single fallible step, rather
/// than calling the panicking methods.
///
/// The trait methods take `self`/`&self`; `filter` deliberately takes `self`
/// **by value** so that `table.filter("x", ">", 0.0)` resolves to this method
/// rather than the inherent `Table::filter(&Expr)` (by-value receivers are
/// probed before autoref). Use `(&table).filter(&expr)` for the Arrow-level
/// predicate API.
pub trait TableExt: Sized {
    fn try_filter(&self, column: &str, op: &str, value: Value) -> Res<Table>;
    /// `t.filter("age", ">", 18)` — raises on unknown column/operator.
    fn filter(self, column: &str, op: &str, value: impl Into<Value>) -> Table;
    fn try_groupby(&self, keys: &[&str]) -> Res<Grouped>;
    /// `t.groupby(&["grp"]).agg(&[("val", "mean")])`
    fn groupby(&self, keys: &[&str]) -> Grouped;
    /// First `n` rows.
    fn head(&self, n: usize) -> Table;
    fn try_col_f64(&self, name: &str) -> Res<Vec<f64>>;
    /// Column as `Vec<f64>` (nulls become `NaN`).
    fn col_f64(&self, name: &str) -> Vec<f64>;
    /// Scatter plot of two named columns.
    fn plot_xy(&self, x: &str, y: &str) -> tpt_viz::Plot;
    /// Scatter plot of the first two numeric columns (index vs value if there
    /// is only one). Never raises: an empty table yields an empty plot.
    fn plot(&self) -> tpt_viz::Plot;
    /// OLS of `y` on `xs` with an intercept, via `tpt_stat`.
    fn ols(&self, y: &str, xs: &[&str]) -> tpt_stat::regression::OLSResult;
    /// Compact textual preview (`repr`-ish).
    fn show(&self) -> String;
}

impl TableExt for Table {
    fn try_filter(&self, col: &str, op: &str, value: Value) -> Res<Table> {
        let arr = column(self, col)?;
        let expr = build_expr(
            col,
            op,
            scalar_for(&value, &arr.data_type(), col)?)?;
        Ok(Table::filter(self, &expr)?)
    }

    fn filter(self, col: &str, op: &str, value: impl Into<Value>) -> Table {
        raise(self.try_filter(col, op, value.into()))
    }

    fn try_groupby(&self, keys: &[&str]) -> Res<Grouped> {
        if keys.is_empty() {
            return Err(ScriptError::new("ValueError", "groupby needs >= 1 key"));
        }
        let cols: Vec<ArrayRef> = keys.iter().map(|k| column(self, k)).collect::<Res<_>>()?;
        let mut groups: Vec<Group> = Vec::new();
        for row in 0..self.num_rows() {
            let key: Vec<String> = keys
                .iter()
                .zip(&cols)
                .map(|(n, a)| key_at(a, row, n))
                .collect::<Res<_>>()?;
            match groups.iter_mut().find(|g| g.key == key) {
                Some(g) => g.rows.push(row),
                None => groups.push(Group {
                    key,
                    rows: vec![row],
                }),
            }
        }
        // Deterministic output order: lexicographic by rendered key.
        groups.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(Grouped {
            table: self.clone(),
            keys: keys.iter().map(|k| k.to_string()).collect(),
            groups,
        })
    }

    fn groupby(&self, keys: &[&str]) -> Grouped {
        raise(self.try_groupby(keys))
    }

    fn head(&self, n: usize) -> Table {
        Table::new(self.batch().slice(0, n.min(self.num_rows())))
    }

    fn try_col_f64(&self, name: &str) -> Res<Vec<f64>> {
        let arr = column(self, name)?;
        Ok(numeric(&arr, name)?
            .into_iter()
            .map(|v| v.unwrap_or(f64::NAN))
            .collect())
    }

    fn col_f64(&self, name: &str) -> Vec<f64> {
        raise(self.try_col_f64(name))
    }

    fn plot_xy(&self, x: &str, y: &str) -> tpt_viz::Plot {
        let (xs, ys) = (self.col_f64(x), self.col_f64(y));
        tpt_viz::Plot::new()
            .layer(tpt_viz::Scatter::new(xs, ys).size(4.0))
            .xlabel(x)
            .ylabel(y)
            .title(&format!("{y} vs {x}"))
    }

    fn plot(&self) -> tpt_viz::Plot {
        let nums: Vec<String> = self
            .column_names()
            .into_iter()
            .filter(|n| {
                self.column(n)
                    .map(|a| matches!(a.data_type(), DataType::Float64 | DataType::Int64))
                    .unwrap_or(false)
            })
            .collect();
        match nums.len() {
            0 => tpt_viz::Plot::new().title("empty table"),
            1 => {
                let ys = self.col_f64(&nums[0]);
                let xs: Vec<f64> = (0..ys.len()).map(|i| i as f64).collect();
                tpt_viz::Plot::new()
                    .layer(tpt_viz::Scatter::new(xs, ys).size(4.0))
                    .xlabel("index")
                    .ylabel(&nums[0])
                    .title(&nums[0])
            }
            _ => self.plot_xy(&nums[0], &nums[1]),
        }
    }

    fn ols(&self, y: &str, xs: &[&str]) -> tpt_stat::regression::OLSResult {
        let yv = self.col_f64(y);
        let cols: Vec<Vec<f64>> = xs.iter().map(|c| self.col_f64(c)).collect();
        let design: Vec<Vec<f64>> = (0..yv.len())
            .map(|i| {
                let mut row = vec![1.0];
                row.extend(cols.iter().map(|c| c[i]));
                row
            })
            .collect();
        tpt_stat::regression::ols(&design, &yv)
    }

    fn show(&self) -> String {
        format!(
            "Table[{} rows x {} cols] {:?}",
            self.num_rows(),
            self.num_cols(),
            self.column_names()
        )
    }
}

#[derive(Clone, Debug)]
struct Group {
    key: Vec<String>,
    rows: Vec<usize>,
}

/// Result of [`TableExt::groupby`]; call [`Grouped::agg`] to materialise a table.
pub struct Grouped {
    table: Table,
    keys: Vec<String>,
    groups: Vec<Group>,
}

impl Grouped {
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }
    /// Rendered group keys, in output order.
    pub fn keys(&self) -> Vec<Vec<String>> {
        self.groups.iter().map(|g| g.key.clone()).collect()
    }
    /// Row counts per group, in output order.
    pub fn sizes(&self) -> Vec<usize> {
        self.groups.iter().map(|g| g.rows.len()).collect()
    }

    /// Aggregate: `agg(&[("val", "mean"), ("*", "count")])`.
    ///
    /// Ops: `mean`, `sum`, `count`, `min`, `max`, `std` (sample). Output columns
    /// are the key columns (original dtype preserved) followed by one `Float64`
    /// column per spec, named `col_op` (`count` for the `"*"` pseudo-column).
    /// Nulls are skipped; an empty aggregate yields `NaN`.
    pub fn try_agg(&self, specs: &[(&str, &str)]) -> Res<Table> {
        let first: Vec<u32> = self.groups.iter().map(|g| g.rows[0] as u32).collect();
        let idx = UInt32Array::from(first);
        let mut out: Vec<(String, ArrayRef)> = Vec::new();
        for k in &self.keys {
            let arr = column(&self.table, k)?;
            out.push((k.clone(), tpt_columnar::compute::take(arr.as_ref(), &idx)?));
        }
        for (col_name, op) in specs {
            let values = if *col_name == "*" {
                None
            } else {
                let arr = column(&self.table, col_name)?;
                Some(numeric(&arr, col_name)?)
            };
            let mut agg = Vec::with_capacity(self.groups.len());
            for g in &self.groups {
                let vals: Vec<f64> = match &values {
                    Some(v) => g.rows.iter().filter_map(|&r| v[r]).collect(),
                    None => vec![0.0; g.rows.len()],
                };
                agg.push(apply_op(op, &vals, col_name)?);
            }
            let name = if *col_name == "*" {
                "count".to_string()
            } else {
                format!("{col_name}_{op}")
            };
            out.push((name, Arc::new(Float64Array::from(agg)) as ArrayRef));
        }
        Ok(OmniFrame::from_columns(out)?.as_table())
    }

    /// Script-mode [`Grouped::try_agg`] (raises on error).
    pub fn agg(&self, specs: &[(&str, &str)]) -> Table {
        raise(self.try_agg(specs))
    }
    /// Shorthand for `agg(&[(col, "mean")])`.
    pub fn mean(&self, col: &str) -> Table {
        self.agg(&[(col, "mean")])
    }
    /// Shorthand for `agg(&[("*", "count")])`.
    pub fn count(&self) -> Table {
        self.agg(&[("*", "count")])
    }
}

fn apply_op(op: &str, v: &[f64], col: &str) -> Res<f64> {
    let n = v.len() as f64;
    let sum: f64 = v.iter().sum();
    Ok(match op {
        "count" => n,
        "sum" => sum,
        "mean" => {
            if v.is_empty() {
                f64::NAN
            } else {
                sum / n
            }
        }
        "min" => v.iter().copied().fold(f64::INFINITY, f64::min),
        "max" => v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        "std" => {
            if v.len() < 2 {
                f64::NAN
            } else {
                let m = sum / n;
                (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
            }
        }
        _ => {
            return Err(ScriptError::new(
                "AggError",
                format!("unknown aggregation '{op}' on '{col}' (mean|sum|count|min|max|std)"),
            ))
        }
    })
}
