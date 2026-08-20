//! Trivial cargo-test loopback check.

#[test]
fn trivial_loopback() {
    // Imported inside the test: the example's `main` is empty, so top-level
    // imports would be dead code in the non-test build (`clippy -D warnings`).
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::thread;

    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    let srv = thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let mut b = [0u8; 4];
        s.read_exact(&mut b).unwrap();
        s.write_all(&[9, 9, 9]).unwrap();
    });
    let mut c = std::net::TcpStream::connect(addr).unwrap();
    c.write_all(&[1, 2, 3, 4]).unwrap();
    let mut r = [0u8; 3];
    c.read_exact(&mut r).expect("read ack");
    assert_eq!(&r[..], &[9, 9, 9]);
    srv.join().unwrap();
}

fn main() {}
