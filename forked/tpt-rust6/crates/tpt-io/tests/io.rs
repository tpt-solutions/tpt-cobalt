use std::fs;

use tpt_io::prelude::*;
use tpt_omni::prelude::*;

fn write_tmp(name: &str, content: &str) -> String {
    let p = std::env::temp_dir().join(name);
    fs::write(&p, content).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn csv_inference_and_filter() {
    let path = write_tmp(
        "tpt_io_test.csv",
        "name,age,score\nalice,34,0.91\nbob,12,0.42\ncarol,29,0.73\n",
    );
    let frame = read(&path).unwrap();
    assert_eq!(frame.num_rows(), 3);
    assert_eq!(frame.column_names().len(), 3);

    let adults = frame.as_table().filter(&col("age").ge(18)).unwrap();
    assert_eq!(adults.num_rows(), 2);

    let tall = frame.as_tensor::<f64>("score", &[3]).unwrap();
    assert!((tall.mean() - (0.91 + 0.42 + 0.73) / 3.0).abs() < 1e-9);
    fs::remove_file(&path).ok();
}

#[test]
fn glob_read_concatenates() {
    let d = std::env::temp_dir();
    let p1 = d.join("tpt_glob1.csv");
    let p2 = d.join("tpt_glob2.csv");
    fs::write(&p1, "a,b\n1,2\n").unwrap();
    fs::write(&p2, "a,b\n3,4\n").unwrap();
    let pattern = d.join("tpt_glob*.csv").to_string_lossy().into_owned();
    let frame = read_glob(&pattern).unwrap();
    assert_eq!(frame.num_rows(), 2);
    fs::remove_file(&p1).ok();
    fs::remove_file(&p2).ok();
}

#[test]
fn csv_roundtrip() {
    let p = write_tmp("tpt_rt.csv", "a,b\n1,2\n3,4\n");
    let frame = read(&p).unwrap();
    let out = std::env::temp_dir().join("tpt_rt_out.csv");
    write(&frame, out.to_str().unwrap()).unwrap();
    let frame2 = match read(out.to_str().unwrap()) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("READ ERROR: {:?}", e);
            panic!("read failed")
        }
    };
    assert_eq!(frame2.num_rows(), 2);
    fs::remove_file(&p).ok();
    fs::remove_file(&out).ok();
}

#[test]
fn json_read() {
    let p = write_tmp("tpt_j.json", "[{\"a\":1,\"b\":2.5},{\"a\":3,\"b\":4.0}]");
    let frame = read(&p).unwrap();
    assert_eq!(frame.column_names().len(), 2);
    assert_eq!(frame.num_rows(), 2);
    fs::remove_file(&p).ok();
}

#[test]
fn detect_format_by_magic() {
    let p = std::env::temp_dir().join("tpt_magic.parquet");
    fs::write(&p, b"PAR1xxxx").unwrap();
    assert_eq!(detect_format(p.to_str().unwrap()), Format::Parquet);
    fs::remove_file(&p).ok();
}

#[test]
fn detect_format_short_file_does_not_panic() {
    // 5-byte files exercise the magic-byte probes that slice past 4 bytes.
    let p = std::env::temp_dir().join("tpt_short_5.bin");
    fs::write(&p, b"PAR1x").unwrap();
    assert_eq!(detect_format(p.to_str().unwrap()), Format::Parquet);
    let p6 = std::env::temp_dir().join("tpt_short_6.bin");
    fs::write(&p6, b"SIMPLE").unwrap();
    assert_eq!(detect_format(p6.to_str().unwrap()), Format::Fits);
    fs::remove_file(&p).ok();
    fs::remove_file(&p6).ok();
}

#[test]
fn parse_line_preserves_multibyte_utf8() {
    let line = tpt_io::csv::parse_line("name,city", b',');
    assert_eq!(line, vec!["name", "city"]);
    let line = tpt_io::csv::parse_line("café,北京", b',');
    assert_eq!(line, vec!["café", "北京"]);
    // Escaped quotes inside a quoted field.
    let line = tpt_io::csv::parse_line("\"a\"\"b\",c", b',');
    assert_eq!(line, vec!["a\"b", "c"]);
}

#[test]
fn write_csv_errors_on_unsupported_column() {
    use arrow::array::{ArrayRef, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1i32, 2, 3])) as ArrayRef],
    )
    .unwrap();
    let frame = tpt_omni::OmniFrame::from_record_batch(batch);
    let out = std::env::temp_dir().join("tpt_unsupported.csv");
    let res = tpt_io::write::write_csv(&frame, out.to_str().unwrap());
    assert!(
        res.is_err(),
        "writing an Int32 column must error, not blank"
    );
    fs::remove_file(&out).ok();
}
