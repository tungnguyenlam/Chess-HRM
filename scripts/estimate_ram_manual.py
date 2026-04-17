def estimate():
    # 1. Encoded Board Tensor [8, 8, 119] in float32
    # 8 * 8 * 119 * 4 bytes = 30,464 bytes (~30 KB)
    tensor_size = 8 * 8 * 119 * 4

    # 2. Raw JSON string estimate (approx 600 characters for FEN + 7 history FENs + move)
    # ~600 bytes
    json_str_size = 600

    # 3. Python dict/chess overhead (estimated)
    # 20 KB to be extremely conservative
    overhead = 20 * 1024

    total_per_move = tensor_size + json_str_size + overhead

    print("--- Per Move Estimates ---")
    print(f"Tensor [8,8,119]: {tensor_size / 1024:.2f} KB")
    print(f"Estimated overhead (JSON, dict, board): {overhead / 1024:.2f} KB")
    print(f"Total per move: {total_per_move / 1024:.2f} KB")

    # 4. Buffering Strategy (buffer_size=10,000 moves)
    # This is what will actually be in RAM at any moment
    buffer_ram = (10000 * total_per_move) / (1024**2)

    # 5. Shard size on Disk (100k games * 40 moves/game * 600 bytes/move)
    shard_disk = (100000 * 40 * 600) / (1024**3)

    print("\n--- Runtime RAM Usage ---")
    print(f"Active Buffer (10,000 moves): {buffer_ram:.2f} MB")

    print("\n--- Disk Estimate (Uncompressed) ---")
    print(f"100,000 Game Shard: ~{shard_disk:.2f} GB")
    print("Note: Zstd compression will reduce this by ~5x-10x.")


if __name__ == "__main__":
    estimate()
