"""Generate small PNG fixture images for tests."""
import struct, zlib, os

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)

def write_png(path: str, width: int, height: int, pixels: list[tuple[int,int,int]]) -> None:
    raw = b""
    for y in range(height):
        raw += b"\x00"
        for x in range(width):
            r, g, b = pixels[y * width + x]
            raw += bytes([r, g, b])
    compressed = zlib.compress(raw, 9)

    data = b"\x89PNG\r\n\x1a\n"
    data += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += _png_chunk(b"IDAT", compressed)
    data += _png_chunk(b"IEND", b"")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

if __name__ == "__main__":
    fixture_dir = os.path.join(os.path.dirname(__file__), "fixtures")

    # 16x16 four-quadrant image (red/green/blue/yellow)
    W, H = 16, 16
    pixels = []
    for y in range(H):
        for x in range(W):
            if x < W // 2 and y < H // 2:
                pixels.append((255, 0, 0))
            elif x >= W // 2 and y < H // 2:
                pixels.append((0, 255, 0))
            elif x < W // 2 and y >= H // 2:
                pixels.append((0, 0, 255))
            else:
                pixels.append((255, 255, 0))
    write_png(os.path.join(fixture_dir, "test_4colors.png"), W, H, pixels)

    # 8x8 solid gray
    write_png(os.path.join(fixture_dir, "test_gray.png"), 8, 8,
              [(128, 128, 128)] * 64)

    print("Fixtures written to", fixture_dir)
