//! Arrow IPC file-format bridge (Phase 4 System Layer serialization).
//!
//! Maps named tensors to/from an Arrow IPC *file* (one column per tensor,
//! flattened values, the original shape carried as schema-field metadata so
//! round-trips preserve rank/dims). Uses the workspace's `arrow` crate; the
//! IPC feature is part of its default feature set.

use std::collections::HashMap;
use std::io::Cursor;
use std::sync::Arc;

use tpt_columnar::array::Array;
use tpt_columnar::datatypes::{DataType, Field, Schema};
use tpt_columnar::ipc::{FileReader, FileWriter};

use crate::safetensors::HubError;
use tpt_tensor::{DType, Tensor};

pub fn save_arrow_ipc(tensors: &[(&str, &Tensor)]) -> Result<Vec<u8>, HubError> {
    let mut names: Vec<&str> = Vec::new();
    let mut dtypes: Vec<&str> = Vec::new();
    let mut shapes: Vec<String> = Vec::new();
    let mut data: Vec<&[u8]> = Vec::new();
    for (name, t) in tensors {
        names.push(name);
        dtypes.push(t.dtype().name());
        shapes.push(
            t.shape()
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(","),
        );
        data.push(t.as_bytes());
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("dtype", DataType::Utf8, false),
        Field::new("shape", DataType::Utf8, false),
        Field::new("data", DataType::Binary, false),
    ]));
    let batch = tpt_columnar::record_batch::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(tpt_columnar::array::StringArray::from(names)),
            Arc::new(tpt_columnar::array::StringArray::from(dtypes)),
            Arc::new(tpt_columnar::array::StringArray::from(shapes)),
            Arc::new(tpt_columnar::array::BinaryArray::from(data)),
        ],
    )
    .map_err(|e| HubError::UnknownDtype(e.to_string()))?;

    let mut buf: Vec<u8> = Vec::new();
    let mut writer = FileWriter::try_new(&mut buf, &batch.schema())
        .map_err(|e| HubError::UnknownDtype(e.to_string()))?;
    writer
        .write(&batch)
        .map_err(|e| HubError::UnknownDtype(e.to_string()))?;
    writer
        .finish()
        .map_err(|e| HubError::UnknownDtype(e.to_string()))?;
    Ok(buf)
}

fn str_col<'a>(
    batch: &'a tpt_columnar::record_batch::RecordBatch,
    name: &str,
) -> Result<Vec<&'a str>, HubError> {
    let col = batch.column_by_name(name).ok_or(HubError::BadHeaderLen)?;
    let a = col
        .as_any()
        .downcast_ref::<tpt_columnar::array::StringArray>()
        .ok_or(HubError::BadHeaderLen)?;
    Ok((0..a.len()).map(|i| a.value(i)).collect())
}

/// Parse an Arrow IPC buffer produced by [`save_arrow_ipc`] back into tensors.
pub fn load_arrow_ipc(bytes: &[u8]) -> Result<HashMap<String, Tensor>, HubError> {
    let reader = FileReader::try_new(Cursor::new(bytes))
        .map_err(|e| HubError::UnknownDtype(e.to_string()))?;
    let mut out = HashMap::new();
    for batch in reader {
        let batch = batch.map_err(|e| HubError::UnknownDtype(e.to_string()))?;
        let names = str_col(&batch, "name")?;
        let dtypes = str_col(&batch, "dtype")?;
        let shapes = str_col(&batch, "shape")?;
        let data_col = batch.column_by_name("data").ok_or(HubError::BadHeaderLen)?;
        let data = data_col
            .as_any()
            .downcast_ref::<tpt_columnar::array::BinaryArray>()
            .ok_or(HubError::BadHeaderLen)?;
        for i in 0..batch.num_rows() {
            let dtype = match dtypes[i] {
                "f64" => DType::F64,
                "f32" => DType::F32,
                "i64" => DType::I64,
                "i32" => DType::I32,
                "i16" => DType::I16,
                "i8" => DType::I8,
                "u8" => DType::U8,
                "bool" => DType::Bool,
                other => return Err(HubError::UnknownDtype(other.to_string())),
            };
            let shape: Vec<usize> = shapes[i]
                .split(',')
                .filter_map(|d| d.parse().ok())
                .collect();
            let tensor = Tensor::from_le_bytes(data.value(i).to_vec(), dtype, &shape);
            out.insert(names[i].to_string(), tensor);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_ipc_roundtrip_f64() {
        let w = Tensor::from_typed(vec![1.0_f64, -2.5, 3.25, 4.0])
            .reshape(&[2, 2])
            .unwrap();
        let b = Tensor::from_typed(vec![0.5_f64]);
        let buf = save_arrow_ipc(&[("weight", &w), ("bias", &b)]).unwrap();
        let loaded = load_arrow_ipc(&buf).unwrap();
        assert_eq!(loaded.len(), 2);
        let lw = &loaded["weight"];
        assert_eq!(lw.shape(), &[2, 2]);
        assert_eq!(lw.dtype(), DType::F64);
        assert_eq!(lw.to_vec::<f64>().unwrap(), vec![1.0, -2.5, 3.25, 4.0]);
        assert_eq!(loaded["bias"].to_vec::<f64>().unwrap(), vec![0.5]);
    }

    #[test]
    fn arrow_ipc_roundtrip_mixed_dtypes() {
        let f = Tensor::from_typed(vec![1.5_f32, -2.5])
            .reshape(&[2])
            .unwrap();
        let i = Tensor::from_typed(vec![-7_i32, 0, 42])
            .reshape(&[1, 3])
            .unwrap();
        let b = Tensor::from_typed(vec![true, false, true, true])
            .reshape(&[2, 2])
            .unwrap();
        let buf = save_arrow_ipc(&[("f", &f), ("i", &i), ("b", &b)]).unwrap();
        let loaded = load_arrow_ipc(&buf).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded["f"].to_vec::<f32>().unwrap(), vec![1.5, -2.5]);
        assert_eq!(loaded["i"].shape(), &[1, 3]);
        assert_eq!(loaded["i"].to_vec::<i32>().unwrap(), vec![-7, 0, 42]);
        assert_eq!(
            loaded["b"].to_vec::<bool>().unwrap(),
            vec![true, false, true, true]
        );
    }
}
