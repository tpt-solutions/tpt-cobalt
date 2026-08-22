use tpt_columnar::array::{ArrayRef, BooleanArray, Float64Array, Int64Array, StringArray};
use serde_json::Value;
use std::sync::Arc;
use tpt_omni::OmniFrame;

use crate::format::IoError;

/// Read a JSON array of objects into an `OmniFrame`, inferring column types.
pub fn read_json(path: &str) -> Result<OmniFrame, IoError> {
    let text = std::fs::read_to_string(path)?;
    let v: Value = serde_json::from_str(&text)?;
    let rows = match v {
        Value::Array(a) => a,
        _ => {
            return Err(IoError::Schema(
                "top-level JSON must be an array of objects".into(),
            ))
        }
    };
    if rows.is_empty() {
        return Err(IoError::Schema("empty JSON array".into()));
    }

    // Collect keys in first-seen order.
    let mut key_order: Vec<String> = Vec::new();
    for row in &rows {
        if let Value::Object(map) = row {
            for k in map.keys() {
                if !key_order.contains(k) {
                    key_order.push(k.clone());
                }
            }
        }
    }

    let mut columns: Vec<(String, Vec<Value>)> = key_order
        .iter()
        .map(|k| (k.clone(), vec![Value::Null; rows.len()]))
        .collect();
    for (r, row) in rows.iter().enumerate() {
        if let Value::Object(map) = row {
            for (k, val) in map {
                if let Some(col) = columns.iter_mut().find(|(name, _)| name == k) {
                    col.1[r] = val.clone();
                }
            }
        }
    }

    let arrays: Vec<ArrayRef> = columns
        .iter()
        .map(|(name, vals)| build_array(name, vals))
        .collect();

    let cols = columns
        .into_iter()
        .map(|(name, _)| name)
        .zip(arrays)
        .collect();
    Ok(OmniFrame::from_columns(cols)?)
}

fn build_array(_name: &str, vals: &[Value]) -> ArrayRef {
    let all_i64 = vals.iter().all(|v| v.is_i64() || v.is_null());
    let all_num = vals.iter().all(|v| v.is_i64() || v.is_f64() || v.is_null());
    let all_bool = vals.iter().all(|v| v.is_boolean() || v.is_null());

    if all_i64 {
        let a: Int64Array = vals.iter().map(|v| v.as_i64()).collect();
        Arc::new(a)
    } else if all_num {
        let a: Float64Array = vals.iter().map(|v| v.as_f64()).collect();
        Arc::new(a)
    } else if all_bool {
        let a: BooleanArray = vals.iter().map(|v| v.as_bool()).collect();
        Arc::new(a)
    } else {
        let a: StringArray = vals
            .iter()
            .map(|v| {
                if v.is_null() {
                    None
                } else {
                    Some(v.to_string())
                }
            })
            .collect();
        Arc::new(a)
    }
}

/// Build a `serde_json::Value` array-of-objects representation of a frame.
pub fn to_json_value(frame: &OmniFrame) -> Value {
    use tpt_columnar::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
    use tpt_columnar::datatypes::DataType;

    let names = frame.column_names();
    let n = frame.batch().num_rows();
    let mut rows = Vec::with_capacity(n);
    for r in 0..n {
        let mut obj = serde_json::Map::new();
        for (ci, name) in names.iter().enumerate() {
            let arr = frame.batch().column(ci);
            let val = match arr.data_type() {
                DataType::Int64 => {
                    let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
                    if a.is_null(r) {
                        Value::Null
                    } else {
                        Value::from(a.value(r))
                    }
                }
                DataType::Float64 => {
                    let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                    if a.is_null(r) {
                        Value::Null
                    } else {
                        Value::from(a.value(r))
                    }
                }
                DataType::Boolean => {
                    let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                    if a.is_null(r) {
                        Value::Null
                    } else {
                        Value::from(a.value(r))
                    }
                }
                DataType::Utf8 => {
                    let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
                    if a.is_null(r) {
                        Value::Null
                    } else {
                        Value::from(a.value(r))
                    }
                }
                _ => Value::Null,
            };
            obj.insert(name.clone(), val);
        }
        rows.push(Value::Object(obj));
    }
    Value::Array(rows)
}
