//! HDF5 reading. The real reader is enabled by the `hdf5` feature (which pulls in
//! the `hdf5` crate and a system libhdf5). Without it, `read_hdf5` returns a
//! clear error so callers know to rebuild with `--features hdf5`.

use crate::format::IoError;
use tpt_omni::OmniFrame;

#[cfg(feature = "hdf5")]
pub fn read_hdf5(path: &str) -> Result<OmniFrame, IoError> {
    use hdf5::File;
    let file = File::open(path).map_err(|e| IoError::Schema(format!("hdf5: {e}")))?;
    // Read the first dataset under the root group and view it as an OmniFrame.
    let dataset = file
        .dataset("/data")
        .or_else(|_| {
            file.group("/")?
                .datasets()
                .next()
                .ok_or_else(|| hdf5::Error::Internal("no dataset found".into()))
        })
        .map_err(|e| IoError::Schema(format!("hdf5: {e}")))?;
    // Materialize as a 2D f64 array -> OmniFrame of columns.
    let arr: ndarray::ArrayD<f64> = dataset
        .read_2d::<f64>()
        .map_err(|e| IoError::Schema(format!("hdf5 read: {e}")))?
        .into_dyn();
    let shape = arr.shape().to_vec();
    let cols = shape.get(1).copied().unwrap_or(0);
    let mut columns = Vec::with_capacity(cols);
    let names: Vec<String> = (0..cols).map(|c| format!("col{c}")).collect();
    for c in 0..cols {
        let col: Vec<f64> = (0..shape[0]).map(|r| arr[[r, c]]).collect();
        let array: tpt_columnar::array::ArrayRef =
            std::sync::Arc::new(tpt_columnar::array::Float64Array::from(col));
        columns.push((names[c].clone(), array));
    }
    Ok(tpt_omni::OmniFrame::from_columns(columns)?)
}

#[cfg(not(feature = "hdf5"))]
pub fn read_hdf5(_path: &str) -> Result<OmniFrame, IoError> {
    Err(IoError::UnsupportedFormat(
        "HDF5 support is gated behind the `hdf5` feature: rebuild with --features hdf5".into(),
    ))
}

#[cfg(all(test, feature = "hdf5"))]
mod tests {
    use super::*;

    #[test]
    fn hdf5_roundtrip() {
        let path = std::env::temp_dir().join("tpt_hdf5_roundtrip.h5");
        let _ = std::fs::remove_file(&path);
        {
            let file = hdf5::File::create(&path).expect("create hdf5 file");
            let data = hdf5::ndarray::Array2::<f64>::from_shape_vec(
                (3, 2),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .unwrap();
            let ds = file
                .new_dataset::<f64>()
                .shape((3, 2))
                .create("data")
                .expect("create dataset");
            ds.write(&data).expect("write dataset");
        }
        let frame = read_hdf5(path.to_str().unwrap()).expect("read_hdf5");
        assert_eq!(frame.num_rows(), 3);
        assert_eq!(frame.column_names().len(), 2);
        std::fs::remove_file(&path).ok();
    }
}
