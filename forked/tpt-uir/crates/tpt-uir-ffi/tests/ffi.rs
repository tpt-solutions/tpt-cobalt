use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;

use tpt_uir_core::{Block, OpName, Operation, Region};
use tpt_uir_serde::serialize_region;

#[test]
fn ffi_load_and_query() {
    let region = Region {
        blocks: vec![Block {
            arguments: vec![
                (0, tpt_uir_core::Type::Index),
                (1, tpt_uir_core::Type::Index),
            ],
            operations: vec![Operation {
                id: 1,
                op_name: OpName::parse("tpt_gpu.launch").unwrap(),
                operands: vec![0, 1],
                results: vec![],
                regions: vec![],
                attributes: vec![],
            }],
        }],
    };
    let bytes = serialize_region(&region).unwrap();

    let mut handle: *mut c_void = ptr::null_mut();
    let rc = unsafe { tpt_uir_ffi::tpt_uir_load(bytes.as_ptr(), bytes.len(), &mut handle) };
    assert_eq!(rc, 0);
    assert!(!handle.is_null());

    let mut blocks: usize = 0;
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_block_count(handle, &mut blocks) },
        0
    );
    assert_eq!(blocks, 1);

    let mut ops: usize = 0;
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_op_count(handle, 0, &mut ops) },
        0
    );
    assert_eq!(ops, 1);

    let mut dialect: *const c_char = ptr::null();
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_op_dialect(handle, 0, 0, &mut dialect) },
        0
    );
    let d = unsafe { CStr::from_ptr(dialect) }.to_str().unwrap();
    assert_eq!(d, "tpt_gpu");

    let mut op: *const c_char = ptr::null();
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_op_op(handle, 0, 0, &mut op) },
        0
    );
    assert_eq!(unsafe { CStr::from_ptr(op) }.to_str().unwrap(), "launch");

    let mut oc: usize = 0;
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_op_operand_count(handle, 0, 0, &mut oc) },
        0
    );
    assert_eq!(oc, 2);

    let mut operand: u32 = 99;
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_op_operand(handle, 0, 0, 1, &mut operand) },
        0
    );
    assert_eq!(operand, 1);

    assert_eq!(unsafe { tpt_uir_ffi::tpt_uir_validate_ssa(handle) }, 0);
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_validate_dialect(handle, c"tpt_gpu".as_ptr()) },
        0
    );
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_validate_dialect(handle, c"crucible".as_ptr()) },
        1
    );

    unsafe { tpt_uir_ffi::tpt_uir_free(handle) };
}

#[test]
fn ffi_load_bad_pointer() {
    let mut handle: *mut c_void = ptr::null_mut();
    assert_eq!(
        unsafe { tpt_uir_ffi::tpt_uir_load(ptr::null(), 0, &mut handle) },
        1
    );
}
