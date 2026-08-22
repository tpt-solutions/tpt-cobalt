use tpt_columnar::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use tpt_columnar::compute::{and, not, or};
use tpt_columnar::datatypes::{DataType, Field};
use tpt_columnar::record_batch::RecordBatch;
use std::sync::Arc;

use crate::error::OmniError;

/// A relational view over an `OmniFrame`'s Arrow record batch.
#[derive(Clone)]
pub struct Table {
    batch: RecordBatch,
}

#[derive(Clone)]
pub enum Scalar {
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Scalar::I64(v)
    }
}
impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Scalar::F64(v)
    }
}
impl From<bool> for Scalar {
    fn from(v: bool) -> Self {
        Scalar::Bool(v)
    }
}
impl From<&str> for Scalar {
    fn from(v: &str) -> Self {
        Scalar::Str(v.to_string())
    }
}
impl From<String> for Scalar {
    fn from(v: String) -> Self {
        Scalar::Str(v)
    }
}

#[derive(Clone, Copy)]
enum Cmp {
    Gt,
    Lt,
    Ge,
    Le,
    Eq,
    Ne,
}

/// A boolean predicate over a table's columns.
#[derive(Clone)]
pub enum Expr {
    Gt(String, Scalar),
    Lt(String, Scalar),
    Ge(String, Scalar),
    Le(String, Scalar),
    Eq(String, Scalar),
    Ne(String, Scalar),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
}

impl Expr {
    pub fn and(self, other: Expr) -> Expr {
        Expr::And(Box::new(self), Box::new(other))
    }
    pub fn or(self, other: Expr) -> Expr {
        Expr::Or(Box::new(self), Box::new(other))
    }
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Expr {
        Expr::Not(Box::new(self))
    }

    pub fn eval(&self, batch: &RecordBatch) -> Result<BooleanArray, OmniError> {
        match self {
            Expr::Gt(c, s) => compare(batch, c, s, Cmp::Gt),
            Expr::Lt(c, s) => compare(batch, c, s, Cmp::Lt),
            Expr::Ge(c, s) => compare(batch, c, s, Cmp::Ge),
            Expr::Le(c, s) => compare(batch, c, s, Cmp::Le),
            Expr::Eq(c, s) => compare(batch, c, s, Cmp::Eq),
            Expr::Ne(c, s) => compare(batch, c, s, Cmp::Ne),
            Expr::And(a, b) => Ok(and(&a.eval(batch)?, &b.eval(batch)?)?),
            Expr::Or(a, b) => Ok(or(&a.eval(batch)?, &b.eval(batch)?)?),
            Expr::Not(a) => Ok(not(&a.eval(batch)?)?),
        }
    }
}

/// Start building a column predicate, e.g. `col("age").gt(18)`.
pub fn col(name: &str) -> Column {
    Column {
        name: name.to_string(),
    }
}

pub struct Column {
    name: String,
}
impl Column {
    pub fn gt(self, v: impl Into<Scalar>) -> Expr {
        Expr::Gt(self.name, v.into())
    }
    pub fn lt(self, v: impl Into<Scalar>) -> Expr {
        Expr::Lt(self.name, v.into())
    }
    pub fn ge(self, v: impl Into<Scalar>) -> Expr {
        Expr::Ge(self.name, v.into())
    }
    pub fn le(self, v: impl Into<Scalar>) -> Expr {
        Expr::Le(self.name, v.into())
    }
    pub fn eq(self, v: impl Into<Scalar>) -> Expr {
        Expr::Eq(self.name, v.into())
    }
    pub fn ne(self, v: impl Into<Scalar>) -> Expr {
        Expr::Ne(self.name, v.into())
    }
}

fn compare(batch: &RecordBatch, col: &str, s: &Scalar, op: Cmp) -> Result<BooleanArray, OmniError> {
    let i = batch
        .schema()
        .index_of(col)
        .map_err(|_| OmniError::ColumnNotFound(col.to_string()))?;
    let arr = batch.column(i);
    let apply = |a: bool, b: bool| match op {
        Cmp::Gt => a && b,
        Cmp::Lt => !a && b,
        Cmp::Ge => a || b,
        Cmp::Le => !a || b,
        Cmp::Eq => a == b,
        Cmp::Ne => a != b,
    };
    let out: Vec<bool> = match (arr.data_type(), s) {
        (DataType::Float64, Scalar::F64(v)) => {
            let a = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| OmniError::NotPrimitive(col.to_string()))?;
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        false
                    } else {
                        cmp_op(op, a.value(i), *v)
                    }
                })
                .collect()
        }
        (DataType::Int64, Scalar::I64(v)) => {
            let a = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| OmniError::NotPrimitive(col.to_string()))?;
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        false
                    } else {
                        cmp_op(op, a.value(i), *v)
                    }
                })
                .collect()
        }
        (DataType::Boolean, Scalar::Bool(v)) => {
            let a = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| OmniError::NotPrimitive(col.to_string()))?;
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        false
                    } else {
                        apply(a.value(i) == *v, true)
                    }
                })
                .collect()
        }
        (DataType::Utf8, Scalar::Str(v)) => {
            let a = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| OmniError::NotPrimitive(col.to_string()))?;
            (0..a.len())
                .map(|i| {
                    if a.is_null(i) {
                        false
                    } else {
                        apply(a.value(i) == v, true)
                    }
                })
                .collect()
        }
        _ => return Err(OmniError::UnsupportedType(col.to_string())),
    };
    Ok(BooleanArray::from(out))
}

fn cmp_op<T: PartialOrd>(op: Cmp, a: T, b: T) -> bool {
    match op {
        Cmp::Gt => a > b,
        Cmp::Lt => a < b,
        Cmp::Ge => a >= b,
        Cmp::Le => a <= b,
        Cmp::Eq => a == b,
        Cmp::Ne => a != b,
    }
}

impl Table {
    pub fn new(batch: RecordBatch) -> Self {
        Self { batch }
    }
    pub fn batch(&self) -> &RecordBatch {
        &self.batch
    }
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }
    pub fn num_cols(&self) -> usize {
        self.batch.num_columns()
    }
    pub fn column_names(&self) -> Vec<String> {
        self.batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().to_string())
            .collect()
    }
    pub fn column(&self, name: &str) -> Result<ArrayRef, OmniError> {
        let i = self
            .batch
            .schema()
            .index_of(name)
            .map_err(|_| OmniError::ColumnNotFound(name.to_string()))?;
        Ok(self.batch.column(i).clone())
    }

    /// Filter rows by a predicate, returning a new table (Arrow-backed).
    pub fn filter(&self, expr: &Expr) -> Result<Table, OmniError> {
        let mask = expr.eval(&self.batch)?;
        let cols = (0..self.batch.num_columns())
            .map(|i| tpt_columnar::compute::filter(self.batch.column(i).as_ref(), &mask))
            .collect::<Result<Vec<_>, _>>()?;
        let batch = RecordBatch::try_new(self.batch.schema(), cols)?;
        Ok(Table::new(batch))
    }

    /// Project a subset of columns.
    pub fn select(&self, names: &[&str]) -> Result<Table, OmniError> {
        let idxs = names
            .iter()
            .map(|n| self.batch.schema().index_of(n))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| OmniError::ColumnNotFound("select".into()))?;
        let cols = idxs.iter().map(|&i| self.batch.column(i).clone()).collect();
        let fields: Vec<Field> = idxs
            .iter()
            .map(|&i| self.batch.schema().field(i).clone())
            .collect();
        let schema = Arc::new(tpt_columnar::datatypes::Schema::new(fields));
        let batch = RecordBatch::try_new(schema, cols)?;
        Ok(Table::new(batch))
    }

    /// Number of rows (PyTorch `.len()`-style ergonomics).
    pub fn len(&self) -> usize {
        self.batch.num_rows()
    }
    pub fn is_empty(&self) -> bool {
        self.batch.num_rows() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_columnar::datatypes::Schema;

    fn sample() -> Table {
        let score = Float64Array::from(vec![0.1, 0.5, 0.9, 0.4]);
        let age = Int64Array::from(vec![10, 25, 33, 8]);
        let ok = BooleanArray::from(vec![true, false, true, true]);
        let name = StringArray::from(vec!["a", "b", "c", "d"]);
        let schema = Arc::new(Schema::new(vec![
            Field::new("score", DataType::Float64, true),
            Field::new("age", DataType::Int64, true),
            Field::new("ok", DataType::Boolean, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(score), Arc::new(age), Arc::new(ok), Arc::new(name)],
        )
        .unwrap();
        Table::new(batch)
    }

    #[test]
    fn filter_numeric_comparisons() {
        let t = sample();
        assert_eq!(t.filter(&col("age").gt(10)).unwrap().num_rows(), 2);
        assert_eq!(t.filter(&col("age").ge(10)).unwrap().num_rows(), 3);
        assert_eq!(t.filter(&col("age").lt(10)).unwrap().num_rows(), 1);
        assert_eq!(t.filter(&col("age").le(10)).unwrap().num_rows(), 2);
        assert_eq!(t.filter(&col("age").eq(10i64)).unwrap().num_rows(), 1);
        assert_eq!(t.filter(&col("age").ne(10i64)).unwrap().num_rows(), 3);
    }

    #[test]
    fn filter_and_or_not_combinators() {
        let t = sample();
        let and_out = t
            .filter(&col("age").ge(10).and(col("ok").eq(true)))
            .unwrap();
        assert_eq!(and_out.num_rows(), 2);

        let or_out = t.filter(&col("age").lt(9).or(col("age").gt(30))).unwrap();
        assert_eq!(or_out.num_rows(), 2);

        let not_out = t.filter(&col("ok").eq(true).not()).unwrap();
        assert_eq!(not_out.num_rows(), 1);
    }

    #[test]
    fn filter_string_and_bool_columns() {
        let t = sample();
        assert_eq!(t.filter(&col("name").eq("c")).unwrap().num_rows(), 1);
        assert_eq!(t.filter(&col("ok").eq(true)).unwrap().num_rows(), 3);
    }

    #[test]
    fn select_projects_columns_in_order() {
        let t = sample();
        let out = t.select(&["age", "name"]).unwrap();
        assert_eq!(out.num_cols(), 2);
        assert_eq!(
            out.column_names(),
            vec!["age".to_string(), "name".to_string()]
        );
    }

    #[test]
    fn select_missing_column_errors() {
        let t = sample();
        assert!(t.select(&["nope"]).is_err());
    }

    #[test]
    fn filter_missing_column_errors() {
        let t = sample();
        assert!(matches!(
            t.filter(&col("nope").gt(1i64)),
            Err(OmniError::ColumnNotFound(_))
        ));
    }

    #[test]
    fn filter_type_mismatch_errors() {
        let t = sample();
        // "age" is Int64 but the scalar is a float - type dispatch has no arm for it.
        assert!(matches!(
            t.filter(&col("age").gt(1.5)),
            Err(OmniError::UnsupportedType(_))
        ));
    }

    #[test]
    fn len_and_is_empty_helpers() {
        let t = sample();
        assert!(!t.is_empty());
        assert_eq!(t.len(), 4);
        let empty = t.filter(&col("age").gt(1000)).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn column_returns_named_array() {
        let t = sample();
        let c = t.column("age").unwrap();
        assert_eq!(c.len(), 4);
        assert!(t.column("missing").is_err());
    }
}
