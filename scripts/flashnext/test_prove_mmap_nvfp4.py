# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU-only helper checks; run directly to avoid repository pytest/TRT imports.

    OMP_NUM_THREADS=1 python test_prove_mmap_nvfp4.py

The GPU integration test is prove_mmap_nvfp4.py itself and is explicitly opt-in.
"""

import tempfile
import unittest
from pathlib import Path

import torch
from prove_mmap_nvfp4 import _GLOBAL_SCALE, _Fixture, _id_cases, _mapping_regions

__all__ = []


class MmapProofTests(unittest.TestCase):
    def test_reference_low_nibble_order_and_block_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), (1,), 32)
            packed = torch.tensor(
                [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8
            )
            scales = torch.tensor([[0.5, 2.0]]).to(torch.float8_e4m3fn)
            fixture.shards = [(packed, scales)]
            decoded = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
            )
            expected = torch.cat((decoded * 0.5, decoded * 2.0)) * _GLOBAL_SCALE
            torch.testing.assert_close(
                fixture.reference(torch.tensor([0])),
                expected.bfloat16().unsqueeze(0),
                rtol=0,
                atol=0,
            )

    def test_retained_mmap_survives_closed_safe_open_and_reclaim(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), (7, 11), 48)
            ids = torch.tensor([0, 6, 7, 17, 0])
            expected = fixture.reference(ids)
            pointers = [tensor.data_ptr() for shard in fixture.shards for tensor in shard]
            fixture.reclaim(evict_file_cache=False)
            self.assertTrue(all(region["Rss"] == "0 kB" for region in fixture.smaps()))
            self.assertEqual(
                pointers, [tensor.data_ptr() for shard in fixture.shards for tensor in shard]
            )
            torch.testing.assert_close(fixture.reference(ids), expected, rtol=0, atol=0)
            self.assertTrue(all(region["Locked"] == "0 kB" for region in fixture.smaps()))

    def test_id_cases_cover_unequal_shards_duplicates_and_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), (7, 11, 3), 16)
            cases = _id_cases(fixture, 16)
            for ids in cases:
                self.assertTrue({0, 6, 7, 17, 18, 20}.issubset(set(ids.tolist())))
                self.assertEqual(ids[-1], ids[0])
                self.assertTrue(bool(((ids >= 0) & (ids < 21)).all()))
            self.assertFalse(torch.equal(cases[0], cases[1]))
            self.assertFalse(torch.equal(fixture.reference(cases[0]), fixture.reference(cases[1])))

    def test_unmapped_external_file_is_not_accepted_as_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "No file-backed mapping"):
                _mapping_regions(Path(directory) / "not-a-fixture")

    def test_read_only_protection_preserves_tensor_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory), (7, 11), 48)
            ids = torch.tensor([0, 6, 7, 17, 0])
            expected = fixture.reference(ids)
            fixture.protect_read_only()
            self.assertTrue(all(region["permissions"] == "r--p" for region in fixture.smaps()))
            fixture.reclaim(evict_file_cache=False)
            torch.testing.assert_close(fixture.reference(ids), expected, rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
