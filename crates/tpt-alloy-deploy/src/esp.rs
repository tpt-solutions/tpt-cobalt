//! ESP32 ROM-UART download protocol: SLIP framing, command packets, and the
//! flash-plan driver.
//!
//! Framing per the ESP32 ROM loader protocol: each packet is SLIP-encoded
//! (`0xC0` … `0xC0`, `0xDB 0xDC` for `0xC0`, `0xDB 0xDD` for `0xDB`). The
//! packet body is `[direction u8][command u16 LE][size u16 LE][checksum
//! u32 LE][data…]`; requests use direction `0x00`, responses `0x01` with a
//! status byte pair (`0 = OK`) at the end of the data. The data checksum is
//! the protocol's XOR-`0xEF` form (used when MD5 is not requested).
//!
//! The byte-level layers are pure; I/O only touches the [`Transport`] trait,
//! so tests drive the whole flash sequence through an in-memory transport.

use std::fmt;
use std::io;

// ------------------------------- SLIP framing -------------------------------

/// SLIP-encode `frame` (leading and trailing `0xC0` markers).
pub fn slip_encode(frame: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(frame.len() + 2);
    out.push(0xC0);
    for &b in frame {
        match b {
            0xC0 => out.extend_from_slice(&[0xDB, 0xDC]),
            0xDB => out.extend_from_slice(&[0xDB, 0xDD]),
            _ => out.push(b),
        }
    }
    out.push(0xC0);
    out
}

/// Decode the first SLIP frame from `buf`; returns `(frame, bytes_consumed)`.
pub fn slip_decode(buf: &[u8]) -> Result<(Vec<u8>, usize), EspError> {
    if buf.first() != Some(&0xC0) {
        return Err(EspError::Framing("expected leading 0xC0".into()));
    }
    let mut out = Vec::with_capacity(buf.len());
    let mut i = 1;
    while i < buf.len() {
        match buf[i] {
            0xC0 => return Ok((out, i + 1)),
            0xDB => {
                let esc = buf
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| EspError::Framing("truncated escape sequence".into()))?;
                match esc {
                    0xDC => out.push(0xC0),
                    0xDD => out.push(0xDB),
                    other => {
                        return Err(EspError::Framing(format!("bad escape 0xDB 0x{other:02X}")));
                    }
                }
                i += 2;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    Err(EspError::Incomplete)
}

// ------------------------------ command frames ------------------------------

/// ROM loader commands used by the flash sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum Command {
    Sync = 0x08,
    FlashBegin = 0x02,
    FlashData = 0x03,
    FlashEnd = 0x04,
}

impl Command {
    fn from_u16(v: u16) -> Option<Self> {
        Some(match v {
            0x08 => Command::Sync,
            0x02 => Command::FlashBegin,
            0x03 => Command::FlashData,
            0x04 => Command::FlashEnd,
            _ => return None,
        })
    }
}

/// A request or response packet (direction, command, payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packet {
    pub is_response: bool,
    pub command: Command,
    pub data: Vec<u8>,
}

impl Packet {
    /// Build a request packet with the protocol XOR-0xEF data checksum.
    pub fn request(command: Command, data: &[u8]) -> Vec<u8> {
        Self::request_with_checksum(command, data, checksum(data))
    }

    /// Like [`Packet::request`] with an explicit checksum (`0` where the
    /// protocol ignores it).
    pub fn request_with_checksum(command: Command, data: &[u8], sum: u32) -> Vec<u8> {
        let mut body = vec![0x00u8]; // direction: request
        body.extend_from_slice(&(command as u16).to_le_bytes());
        body.extend_from_slice(&(data.len() as u16).to_le_bytes());
        body.extend_from_slice(&sum.to_le_bytes());
        body.extend_from_slice(data);
        slip_encode(&body)
    }

    /// Parse a response packet from a SLIP-decoded body. Header: direction
    /// (1) + command (2) + size (2) + checksum (4) = 9 bytes, then data.
    pub fn parse_response(body: &[u8]) -> Result<Self, EspError> {
        if body.len() < 9 {
            return Err(EspError::Framing("packet shorter than header".into()));
        }
        if body[0] != 0x01 {
            return Err(EspError::Framing("expected response direction 0x01".into()));
        }
        let cmd = u16::from_le_bytes([body[1], body[2]]);
        let size = u16::from_le_bytes([body[3], body[4]]) as usize;
        let command = Command::from_u16(cmd)
            .ok_or_else(|| EspError::Framing(format!("unknown command 0x{cmd:04X}")))?;
        let data = body[9..9 + size].to_vec();
        Ok(Packet {
            is_response: true,
            command,
            data,
        })
    }

    /// The trailing status byte pair of a response (`0x00 0x00` = success).
    pub fn status(&self) -> Result<(), EspError> {
        match self.data.last().copied() {
            Some(0x00) => Ok(()),
            Some(code) => Err(EspError::DeviceStatus(code)),
            None => Err(EspError::Framing("response without status".into())),
        }
    }
}

/// Protocol XOR-0xEF checksum over a data chunk.
pub fn checksum(data: &[u8]) -> u32 {
    let mut sum: u8 = 0xEF;
    for &b in data {
        sum ^= b;
    }
    sum as u32
}

// --------------------------------- transport --------------------------------

/// Byte transport to the device (real serial impl plugs in here).
pub trait Transport {
    fn write_all(&mut self, buf: &[u8]) -> io::Result<()>;
    /// Read available bytes (non-blocking or short reads are fine; the
    /// driver re-polls).
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>;
}

// --------------------------------- flasher ----------------------------------

const FLASH_PACKET_SIZE: usize = 0x800;

/// Error paths of the ESP32 deploy layer.
#[derive(Debug, Clone, PartialEq)]
pub enum EspError {
    Framing(String),
    /// A SLIP frame started but has not fully arrived yet (read more bytes).
    Incomplete,
    DeviceStatus(u8),
    Io(String),
    Timeout,
}

impl fmt::Display for EspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EspError::Framing(m) => write!(f, "framing error: {m}"),
            EspError::Incomplete => write!(f, "frame incomplete"),
            EspError::DeviceStatus(c) => write!(f, "device returned error status 0x{c:02X}"),
            EspError::Io(m) => write!(f, "io error: {m}"),
            EspError::Timeout => write!(f, "timed out waiting for device response"),
        }
    }
}

impl std::error::Error for EspError {}

impl From<io::Error> for EspError {
    fn from(e: io::Error) -> Self {
        EspError::Io(e.to_string())
    }
}

/// Drives the ROM-UART flash sequence over any [`Transport`].
pub struct EspFlasher<'t> {
    transport: &'t mut dyn Transport,
}

impl<'t> EspFlasher<'t> {
    pub fn new(transport: &'t mut dyn Transport) -> Self {
        EspFlasher { transport }
    }

    /// Send one request and wait for the matching response.
    fn transact(&mut self, command: Command, data: &[u8]) -> Result<Packet, EspError> {
        let frame = Packet::request(command, data);
        self.transport.write_all(&frame)?;
        let mut buf = [0u8; 1024];
        let mut pending = Vec::new();
        loop {
            let n = self.transport.read(&mut buf)?;
            if n == 0 {
                return Err(EspError::Timeout);
            }
            pending.extend_from_slice(&buf[..n]);
            // skip garbage before the frame marker, then decode one frame
            while !pending.is_empty() {
                if pending.first() != Some(&0xC0) {
                    pending.remove(0);
                    continue;
                }
                let (body, consumed) = match slip_decode(&pending) {
                    Ok(r) => r,
                    Err(EspError::Incomplete) => break, // wait for more bytes
                    Err(e) => return Err(e),
                };
                pending.drain(..consumed);
                let resp = Packet::parse_response(&body)?;
                if resp.command == command {
                    resp.status()?;
                    return Ok(resp);
                }
                // unrelated response byte stream: keep scanning
            }
        }
    }

    /// ROM-sync handshake (8 sync attempts is the ROM-loader convention).
    pub fn sync(&mut self) -> Result<(), EspError> {
        let mut data = vec![0x07, 0x07, 0x12, 0x20];
        data.extend(std::iter::repeat_n(0x55, 32));
        for _ in 0..8 {
            let mut frame = vec![0x00u8];
            frame.extend_from_slice(&(Command::Sync as u16).to_le_bytes());
            frame.extend_from_slice(&(data.len() as u16).to_le_bytes());
            frame.extend_from_slice(&0u32.to_le_bytes());
            frame.extend_from_slice(&data);
            self.transport.write_all(&slip_encode(&frame))?;
            let mut buf = [0u8; 128];
            let n = self.transport.read(&mut buf)?;
            if n > 0 {
                let (body, _) = slip_decode(&buf[..n])?;
                Packet::parse_response(&body)?.status()?;
                return Ok(());
            }
        }
        Err(EspError::Timeout)
    }

    /// Flash `image` at `offset`: `flash_begin` → `flash_data` per
    /// 0x800-byte packet (with sequence numbers and XOR checksums) →
    /// `flash_end` (reboot into the new image).
    pub fn flash(&mut self, image: &[u8], offset: u32) -> Result<(), EspError> {
        let total = image.len();
        let num_packets = total.div_ceil(FLASH_PACKET_SIZE).max(1);
        let mut begin = Vec::with_capacity(16);
        begin.extend_from_slice(&(total as u32).to_le_bytes());
        begin.extend_from_slice(&(num_packets as u32).to_le_bytes());
        begin.extend_from_slice(&(FLASH_PACKET_SIZE as u32).to_le_bytes());
        begin.extend_from_slice(&offset.to_le_bytes());
        self.transact(Command::FlashBegin, &begin)?;

        for (seq, chunk) in image.chunks(FLASH_PACKET_SIZE).enumerate() {
            let mut data = Vec::with_capacity(8 + chunk.len());
            data.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            data.extend_from_slice(&(seq as u32).to_le_bytes());
            data.extend_from_slice(chunk);
            // checksum rides in the header, not the data section
            let frame = Packet::request_with_checksum(Command::FlashData, &data, checksum(chunk));
            self.transport.write_all(&frame)?;
            let mut buf = [0u8; 64];
            let n = self.transport.read(&mut buf)?;
            if n == 0 {
                return Err(EspError::Timeout);
            }
            let (body, _) = slip_decode(&buf[..n])?;
            Packet::parse_response(&body)?.status()?;
        }

        let end = 1u32.to_le_bytes(); // 1 = reboot after flashing
        self.transact(Command::FlashEnd, &end)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------ SLIP tests ------------------------------

    #[test]
    fn slip_round_trip_with_escape_bytes() {
        let frame: Vec<u8> = vec![0x00, 0xC0, 0x01, 0x02, 0xDB, 0x03, 0xC0, 0xDD, 0xFF];
        let encoded = slip_encode(&frame);
        assert_eq!(encoded.first(), Some(&0xC0));
        assert_eq!(encoded.last(), Some(&0xC0));
        // interior: every 0xC0/0xDB must be escaped (0xDB always paired)
        let mut i = 1;
        while i < encoded.len() - 1 {
            match encoded[i] {
                0xC0 => panic!("raw 0xC0 inside frame at {i}"),
                0xDB => {
                    assert!(
                        matches!(encoded.get(i + 1), Some(0xDC) | Some(0xDD)),
                        "bad escape at {i}"
                    );
                    i += 2;
                }
                _ => i += 1,
            }
        }
        let (decoded, consumed) = slip_decode(&encoded).unwrap();
        assert_eq!(decoded, frame);
        assert_eq!(consumed, encoded.len());
    }

    #[test]
    fn slip_decode_rejects_truncated_and_bad_escapes() {
        assert_eq!(
            slip_decode(&[]).unwrap_err(),
            EspError::Framing("expected leading 0xC0".into())
        );
        assert_eq!(
            slip_decode(&[0xC0, 0x01, 0x02]).unwrap_err(),
            EspError::Incomplete
        );
        assert!(matches!(
            slip_decode(&[0xC0, 0xDB, 0x99, 0xC0]).unwrap_err(),
            EspError::Framing(_)
        ));
    }

    // --------------------------- packet layout test -------------------------

    #[test]
    fn sync_packet_matches_protocol_layout() {
        let mut data = vec![0x07, 0x07, 0x12, 0x20];
        data.extend(std::iter::repeat(0x55).take(32));
        let frame = Packet::request_with_checksum(Command::Sync, &data, 0);
        // SLIP marker + [dir 1][cmd 2][size 2][checksum 4] + data + marker
        assert_eq!(frame.len(), 1 + 9 + data.len() + 1);
        assert_eq!(frame[0], 0xC0);
        assert_eq!(frame[1], 0x00); // request direction
        assert_eq!(&frame[2..4], &0x08u16.to_le_bytes()); // Sync
        assert_eq!(&frame[4..6], &(data.len() as u16).to_le_bytes());
        assert_eq!(&frame[6..10], &0u32.to_le_bytes()); // checksum
        assert_eq!(&frame[10..14], &[0x07, 0x07, 0x12, 0x20]);
        assert_eq!(frame.last(), Some(&0xC0));
    }

    #[test]
    fn response_parsing_and_status() {
        // OK response for a Sync
        let body: Vec<u8> = [
            vec![0x01],
            (0x08u16).to_le_bytes().to_vec(),
            (2u16).to_le_bytes().to_vec(),
            0u32.to_le_bytes().to_vec(),
            vec![0x00, 0x00],
        ]
        .concat();
        let p = Packet::parse_response(&body).unwrap();
        assert_eq!(p.command, Command::Sync);
        assert!(p.status().is_ok());
        // failure status
        let bad: Vec<u8> = [
            vec![0x01],
            (0x03u16).to_le_bytes().to_vec(),
            (2u16).to_le_bytes().to_vec(),
            0u32.to_le_bytes().to_vec(),
            vec![0x05, 0x05],
        ]
        .concat();
        let p = Packet::parse_response(&bad).unwrap();
        assert_eq!(p.status().unwrap_err(), EspError::DeviceStatus(0x05));
    }

    #[test]
    fn checksum_is_xor_ef() {
        assert_eq!(checksum(&[]), 0xEF);
        assert_eq!(checksum(&[0x00]), 0xEF);
        assert_eq!(checksum(&[0xFF]), 0x10);
        assert_eq!(checksum(&[0x01, 0x02]), 0xEC);
    }

    // ------------------------------ flash tests ------------------------------

    /// In-memory device: records every request, answers with a matching OK
    /// response (or a configurable failure status on packet `fail_at_seq`).
    struct MockEsp {
        log: Vec<Vec<u8>>,
        rx: Vec<u8>,
        fail_at_seq: Option<u32>,
        seq_seen: u32,
    }

    impl MockEsp {
        fn new(fail_at_seq: Option<u32>) -> Self {
            MockEsp {
                log: Vec::new(),
                rx: Vec::new(),
                fail_at_seq,
                seq_seen: 0,
            }
        }
    }

    impl Transport for MockEsp {
        fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
            let (body, _) = slip_decode(buf).expect("driver sends clean SLIP");
            let cmd = u16::from_le_bytes([body[1], body[2]]);
            // craft the response for this request
            let mut resp: Vec<u8> = vec![0x01];
            resp.extend_from_slice(&cmd.to_le_bytes());
            resp.extend_from_slice(&2u16.to_le_bytes());
            resp.extend_from_slice(&0u32.to_le_bytes());
            let fail = cmd == Command::FlashData as u16 && self.fail_at_seq == Some(self.seq_seen);
            if cmd == Command::FlashData as u16 {
                self.seq_seen += 1;
            }
            let status = if fail { 0x05 } else { 0x00 };
            resp.extend_from_slice(&[status, status]);
            self.rx.extend_from_slice(&slip_encode(&resp));
            self.log.push(body);
            Ok(())
        }

        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = buf.len().min(self.rx.len());
            buf[..n].copy_from_slice(&self.rx[..n]);
            self.rx.drain(..n);
            Ok(n)
        }
    }

    fn parse_requests(mock: &MockEsp) -> Vec<(Command, Vec<u8>, u32)> {
        mock.log
            .iter()
            .map(|body| {
                let cmd = Command::from_u16(u16::from_le_bytes([body[1], body[2]])).unwrap();
                let size = u16::from_le_bytes([body[3], body[4]]) as usize;
                let sum = u32::from_le_bytes([body[5], body[6], body[7], body[8]]);
                (cmd, body[9..9 + size].to_vec(), sum)
            })
            .collect()
    }

    #[test]
    fn flash_sequence_and_payload_chunks_match_the_protocol() {
        let image: Vec<u8> = (0..0x1400u32).map(|i| (i % 253) as u8).collect(); // 2.5 packets
        let mut mock = MockEsp::new(None);
        {
            let mut f = EspFlasher::new(&mut mock);
            f.flash(&image, 0x1_0000).unwrap();
        }
        let reqs = parse_requests(&mock);
        // begin + 3 data packets + end
        assert_eq!(reqs.len(), 5);
        assert_eq!(reqs[0].0, Command::FlashBegin);
        let (total, packets, pkt_size, offset) = {
            let d = &reqs[0].1;
            (
                u32::from_le_bytes(d[0..4].try_into().unwrap()),
                u32::from_le_bytes(d[4..8].try_into().unwrap()),
                u32::from_le_bytes(d[8..12].try_into().unwrap()),
                u32::from_le_bytes(d[12..16].try_into().unwrap()),
            )
        };
        assert_eq!(total, image.len() as u32);
        assert_eq!(packets, 3);
        assert_eq!(pkt_size, 0x800);
        assert_eq!(offset, 0x1_0000);

        for (i, req) in reqs[1..4].iter().enumerate() {
            assert_eq!(req.0, Command::FlashData);
            let d = &req.1;
            assert_eq!(
                u32::from_le_bytes(d[4..8].try_into().unwrap()),
                i as u32,
                "seq"
            );
            let chunk = &d[8..];
            let expected = &image[i * 0x800..((i + 1) * 0x800).min(image.len())];
            assert_eq!(chunk, expected);
            assert_eq!(req.2, checksum(expected), "header checksum");
        }
        assert_eq!(reqs[4].0, Command::FlashEnd);
        assert_eq!(&reqs[4].1, &1u32.to_le_bytes());
    }

    #[test]
    fn device_error_status_fails_the_flash() {
        let image = vec![7u8; 0x800 * 2];
        let mut mock = MockEsp::new(Some(1)); // second data packet fails
        let mut f = EspFlasher::new(&mut mock);
        let err = f.flash(&image, 0).unwrap_err();
        assert_eq!(err, EspError::DeviceStatus(0x05));
    }

    #[test]
    fn sync_handshake_over_mock_transport() {
        let mut mock = MockEsp::new(None);
        let mut f = EspFlasher::new(&mut mock);
        f.sync().unwrap();
        let reqs = parse_requests(&mock);
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].0, Command::Sync);
        assert_eq!(reqs[0].1[0..4], [0x07, 0x07, 0x12, 0x20]);
    }
}
