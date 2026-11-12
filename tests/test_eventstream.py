import struct
import unittest
import zlib

from oneharness.errors import ProviderError
from oneharness.eventstream import decode_frames, encode_frame, frame_for

# A frame built by hand from the layout in the spec, not by the encoder in the
# module under test, so a bug in that encoder cannot hide itself. The bytes were
# independently decoded by botocore's `EventStreamBuffer` -- if this hex and our
# decoder ever disagree, one of them is wrong and it is not botocore.
#
#   length 0x72 = 114, headers 0x4b = 75, prelude CRC 073ef4de,
#   message CRC 3bc4134c, payload {"bytes": "Inference!"}
CAPTURED_CHUNK = bytes.fromhex(
    "000000720000004b073ef4de"
    "0d3a636f6e74656e742d747970650700106170706c69636174696f6e2f6a736f6e"
    "0d3a6d6573736167652d747970650700056576656e74"
    "0b3a6576656e742d747970650700056368756e6b"
    "7b226279746573223a2022496e666572656e636521227d"
    "3bc4134c"
)


class CapturedFrameTests(unittest.TestCase):
    def test_decodes_the_real_frame(self):
        frames = list(decode_frames(CAPTURED_CHUNK))
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].event, "chunk")
        self.assertEqual(frames[0].message_type, "event")
        self.assertEqual(frames[0].json(), {"bytes": "Inference!"})

    def test_survives_being_split_anywhere(self):
        # A socket hands over whatever it happens to have; the frame boundary
        # has nothing to do with chunk boundaries.
        for cut in range(1, len(CAPTURED_CHUNK)):
            frames = list(decode_frames([CAPTURED_CHUNK[:cut], CAPTURED_CHUNK[cut:]]))
            self.assertEqual(len(frames), 1, msg=f"split at {cut}")
            self.assertEqual(frames[0].json(), {"bytes": "Inference!"})

    def test_corrupt_payload_is_rejected(self):
        broken = bytearray(CAPTURED_CHUNK)
        broken[60] ^= 0xFF
        with self.assertRaises(ProviderError):
            list(decode_frames(bytes(broken)))

    def test_corrupt_length_is_rejected_before_reading(self):
        broken = bytearray(CAPTURED_CHUNK)
        broken[3] ^= 0xFF  # the length itself; only the prelude CRC covers it
        with self.assertRaises(ProviderError) as caught:
            list(decode_frames(bytes(broken)))
        self.assertIn("prelude", str(caught.exception))

    def test_truncated_stream_is_an_error_not_a_short_answer(self):
        with self.assertRaises(ProviderError):
            list(decode_frames(CAPTURED_CHUNK[:-6]))


class RoundTripTests(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        blob = encode_frame(b'{"a":1}', {":event-type": "chunk"})
        frame = list(decode_frames(blob))[0]
        self.assertEqual(frame.headers[":event-type"], "chunk")
        self.assertEqual(frame.json(), {"a": 1})

    def test_many_frames_in_one_chunk(self):
        blob = frame_for({"n": 1}, "chunk") + frame_for({"n": 2}, "chunk")
        self.assertEqual([f.json() for f in decode_frames(blob)], [{"n": 1}, {"n": 2}])

    def test_exception_frames_are_marked(self):
        blob = frame_for({"message": "no"}, "ValidationException", exception=True)
        frame = list(decode_frames(blob))[0]
        self.assertEqual(frame.message_type, "exception")
        self.assertEqual(frame.event, "ValidationException")

    def test_header_types_round_trip(self):
        headers = {":event-type": "chunk", ":content-type": "application/json"}
        frame = list(decode_frames(encode_frame(b"{}", headers)))[0]
        self.assertEqual(frame.headers, headers)


class UnitTests(unittest.TestCase):
    def test_empty_input_yields_nothing(self):
        self.assertEqual(list(decode_frames(b"")), [])

    def test_impossible_length_is_rejected(self):
        with self.assertRaises(ProviderError):
            list(decode_frames(struct.pack(">II", 4, 0) + struct.pack(">I", 0)))

    def test_frame_with_no_headers(self):
        blob = encode_frame(b"payload")
        frame = list(decode_frames(blob))[0]
        self.assertEqual(frame.payload, b"payload")
        self.assertEqual(frame.headers, {})

    def test_crc_is_a_real_crc(self):
        blob = encode_frame(b"x", {":event-type": "chunk"})
        (total,) = struct.unpack(">I", blob[:4])
        self.assertEqual(total, len(blob))
        (message_crc,) = struct.unpack(">I", blob[-4:])
        self.assertEqual(message_crc, zlib.crc32(blob[:-4]) & 0xFFFFFFFF)


if __name__ == "__main__":
    unittest.main()
