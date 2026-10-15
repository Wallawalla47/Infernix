"""V3 entry/continuation headers and file payload alignment shared by reader and writer."""

import struct

MAGIC = b"NINFER\x00\x03"
PART_MAGIC = b"NINPRT\x00\x03"
HEADER = struct.Struct("<8sQ16s")
PAYLOAD_ALIGNMENT = 4096

# Writer policy (artifact-container.md §5.3): placement may exceed the §6.2 minimums. The
# converter starts streamed routed-expert bank objects on 2 MiB boundaries — the entry payload
# start and the file segment capacities are aligned so this holds in file space across parts —
# so the SSD expert tier can use huge-page mappings and wide O_DIRECT reads. Readers keep
# enforcing only the layout-mandated 4 KiB minima.
EXPERT_STREAM_ALIGNMENT = 2 * 1024 * 1024
