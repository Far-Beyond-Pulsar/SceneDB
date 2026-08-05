//! Fuzzes `ReplicationRegistry::from_handshake` — parsed once per
//! connection, but still untrusted-peer-controlled input.
#![no_main]

use libfuzzer_sys::fuzz_target;
use pulsar_scenedb::ReplicationRegistry;

fuzz_target!(|data: &[u8]| {
    let _ = ReplicationRegistry::from_handshake(data);
});
